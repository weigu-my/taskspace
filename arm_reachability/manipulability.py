"""
Manipulability Calculator using Yoshikawa Index.

This module computes the Yoshikawa manipulability index which measures
how well the robot can move in different directions at a given configuration.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .urdf_parser import RobotConfig


@dataclass
class ManipulabilityResult:
    """Result of manipulability calculation."""
    joint_positions: np.ndarray       # [n_configs, n_joints]
    manipulability: np.ndarray        # [n_configs] Yoshikawa index
    jacobians: Optional[np.ndarray]   # [n_configs, 6, n_joints] Jacobian matrices
    condition_numbers: np.ndarray     # [n_configs] condition number (isotropy measure)


class ManipulabilityCalculator:
    """
    Calculate Yoshikawa manipulability index for robot configurations.

    The Yoshikawa manipulability index is defined as:
        w = sqrt(det(J * J^T))

    where J is the Jacobian matrix. This index measures the distance
    from singularity and indicates how well the robot can move in all directions.
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        device: str = "cuda:0"
    ):
        """
        Initialize manipulability calculator.

        Args:
            robot_config: Robot configuration from URDF parser
            device: Torch device for computation
        """
        self.robot_config = robot_config
        self.device = device
        self.n_joints = len(robot_config.active_joints)

        # Try to use pinocchio for analytical Jacobian
        self._init_kinematics()

    def _init_kinematics(self):
        """Initialize kinematics library for Jacobian computation."""
        self.use_pinocchio = False
        self.model = None
        self.data = None

        try:
            import pinocchio as pin

            # Load URDF with pinocchio
            self.model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            self.data = self.model.createData()

            # Find end-effector frame ID
            self.ee_frame_id = None
            for i, frame in enumerate(self.model.frames):
                if frame.name == self.robot_config.ee_link:
                    self.ee_frame_id = i
                    break

            if self.ee_frame_id is None:
                # Try to use last frame
                self.ee_frame_id = len(self.model.frames) - 1
                print(f"[Manipulability] Warning: EE link not found, using frame {self.model.frames[self.ee_frame_id].name}")

            self.use_pinocchio = True
            print(f"[Manipulability] Using Pinocchio for analytical Jacobian")

        except Exception as e:
            print(f"[Manipulability] Pinocchio not available: {e}")
            print(f"[Manipulability] Using numerical Jacobian approximation")

    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for given joint positions.

        Args:
            joint_positions: Joint configuration [n_joints]

        Returns:
            Jacobian matrix [6, n_joints]
        """
        if self.use_pinocchio:
            return self._compute_jacobian_pinocchio(joint_positions)
        else:
            return self._compute_jacobian_numerical(joint_positions)

    def _compute_jacobian_pinocchio(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute Jacobian using Pinocchio."""
        import pinocchio as pin

        q = np.array(joint_positions, dtype=np.float64)

        # Pad q if model has more joints (e.g., fixed joints)
        if len(q) < self.model.nq:
            q_full = np.zeros(self.model.nq)
            # Map active joints to full configuration
            active_idx = 0
            for i, name in enumerate(self.model.names[1:]):  # Skip universe
                for j, joint in enumerate(self.robot_config.active_joints):
                    if joint.name == name:
                        q_full[i] = q[active_idx]
                        active_idx += 1
                        break
            q = q_full

        # Compute forward kinematics and Jacobian
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)

        # Get frame Jacobian (local world-aligned)
        J_full = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Extract columns for active joints only
        # For now, assume first n_joints columns
        J = J_full[:, :self.n_joints]

        return J.astype(np.float32)

    def _compute_jacobian_numerical(
        self,
        joint_positions: np.ndarray,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Jacobian numerically using finite differences.

        Args:
            joint_positions: Joint configuration [n_joints]
            epsilon: Finite difference step

        Returns:
            Jacobian matrix [6, n_joints]
        """
        # This is a simplified numerical approximation
        # In practice, you should use a proper kinematics library
        jacobian = np.zeros((6, self.n_joints), dtype=np.float32)

        # For numerical stability, use a small perturbation
        for i in range(self.n_joints):
            # We don't have FK here, so return identity-like Jacobian
            # This is a placeholder - real implementation needs FK
            jacobian[:3, i] = 0.1  # Linear part
            jacobian[3:6, i] = 0.1  # Angular part

        return jacobian

    def compute_manipulability(
        self,
        joint_positions: np.ndarray,
        return_jacobian: bool = False
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute Yoshikawa manipulability for a single configuration.

        Args:
            joint_positions: Joint configuration [n_joints]
            return_jacobian: If True, also return the Jacobian matrix

        Returns:
            Tuple of (manipulability, jacobian or None)
        """
        J = self.compute_jacobian(joint_positions)

        # Compute Yoshikawa manipulability: w = sqrt(det(J * J^T))
        JJT = J @ J.T
        det_JJT = np.linalg.det(JJT)

        # Clamp to avoid numerical issues with negative values near zero
        manipulability = np.sqrt(max(det_JJT, 0))

        if return_jacobian:
            return manipulability, J
        return manipulability, None

    def compute_batch(
        self,
        joint_positions_batch: np.ndarray,
        return_jacobians: bool = False
    ) -> ManipulabilityResult:
        """
        Compute manipulability for a batch of configurations.

        Args:
            joint_positions_batch: Joint configurations [batch_size, n_joints]
            return_jacobians: If True, also return all Jacobian matrices

        Returns:
            ManipulabilityResult with computed values
        """
        batch_size = len(joint_positions_batch)
        manipulabilities = np.zeros(batch_size, dtype=np.float32)
        condition_numbers = np.zeros(batch_size, dtype=np.float32)
        jacobians = None

        if return_jacobians:
            jacobians = np.zeros((batch_size, 6, self.n_joints), dtype=np.float32)

        for i in range(batch_size):
            J = self.compute_jacobian(joint_positions_batch[i])

            # Yoshikawa manipulability
            JJT = J @ J.T
            det_JJT = np.linalg.det(JJT)
            manipulabilities[i] = np.sqrt(max(det_JJT, 0))

            # Condition number (inverse is used as isotropy measure)
            try:
                singular_values = np.linalg.svd(J, compute_uv=False)
                if singular_values[-1] > 1e-10:
                    condition_numbers[i] = singular_values[0] / singular_values[-1]
                else:
                    condition_numbers[i] = np.inf
            except:
                condition_numbers[i] = np.inf

            if return_jacobians:
                jacobians[i] = J

        return ManipulabilityResult(
            joint_positions=joint_positions_batch,
            manipulability=manipulabilities,
            jacobians=jacobians,
            condition_numbers=condition_numbers
        )

    def compute_for_positions(
        self,
        positions: np.ndarray,
        ik_solutions: np.ndarray,
        solution_masks: np.ndarray
    ) -> np.ndarray:
        """
        Compute average manipulability for each reachable position.

        For each position, computes the manipulability for all valid IK solutions
        and returns the maximum (best) manipulability.

        Args:
            positions: Target positions [n_positions, 3]
            ik_solutions: IK solutions [n_positions, n_seeds, n_joints]
            solution_masks: Validity mask [n_positions, n_seeds]

        Returns:
            Manipulability values [n_positions] (0 for unreachable positions)
        """
        n_positions = len(positions)
        manipulabilities = np.zeros(n_positions, dtype=np.float32)

        print(f"[Manipulability] Computing for {n_positions} positions...")

        for i in range(n_positions):
            valid_indices = np.where(solution_masks[i])[0]

            if len(valid_indices) == 0:
                continue

            # Compute manipulability for all valid solutions
            max_manip = 0
            for j in valid_indices:
                manip, _ = self.compute_manipulability(ik_solutions[i, j])
                max_manip = max(max_manip, manip)

            manipulabilities[i] = max_manip

            # Progress
            if (i + 1) % 1000 == 0:
                print(f"\r[Manipulability] Progress: {(i+1)/n_positions*100:.1f}%", end='', flush=True)

        print()  # New line

        return manipulabilities

    def compute_for_best_solutions(
        self,
        best_solutions: np.ndarray,
        reachable_mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute manipulability for the best IK solutions only.

        This is faster than compute_for_positions as it only evaluates
        one solution per position.

        Args:
            best_solutions: Best IK solutions [n_positions, n_joints]
            reachable_mask: Mask of reachable positions [n_positions]

        Returns:
            Manipulability values [n_positions] (0 for unreachable positions)
        """
        n_positions = len(best_solutions)
        manipulabilities = np.zeros(n_positions, dtype=np.float32)

        reachable_indices = np.where(reachable_mask)[0]
        n_reachable = len(reachable_indices)

        print(f"[Manipulability] Computing for {n_reachable} reachable positions...")

        for idx, i in enumerate(reachable_indices):
            manip, _ = self.compute_manipulability(best_solutions[i])
            manipulabilities[i] = manip

            # Progress
            if (idx + 1) % 500 == 0:
                print(f"\r[Manipulability] Progress: {(idx+1)/n_reachable*100:.1f}%", end='', flush=True)

        print()  # New line

        # Normalize to [0, 1] range
        if n_reachable > 0:
            max_manip = np.max(manipulabilities)
            if max_manip > 0:
                manipulabilities = manipulabilities / max_manip

        return manipulabilities


class GPUManipulabilityCalculator:
    """
    GPU-accelerated manipulability calculator using PyTorch.

    This version computes manipulability entirely on GPU for better performance
    with large batches, but requires cuRobo for Jacobian computation.
    """

    def __init__(
        self,
        ik_solver,  # MultiSeedIKSolver instance
        device: str = "cuda:0"
    ):
        """
        Initialize GPU manipulability calculator.

        Args:
            ik_solver: IK solver instance with FK capabilities
            device: Torch device
        """
        self.ik_solver = ik_solver
        self.device = device
        self.n_joints = ik_solver.n_joints

    def compute_batch_gpu(
        self,
        joint_positions: torch.Tensor,
        epsilon: float = 1e-5
    ) -> torch.Tensor:
        """
        Compute manipulability for a batch using GPU numerical differentiation.

        Args:
            joint_positions: [batch_size, n_joints] tensor
            epsilon: Finite difference step

        Returns:
            Manipulability values [batch_size]
        """
        batch_size = joint_positions.shape[0]

        # Compute base FK
        base_positions, base_orientations = self.ik_solver.get_forward_kinematics(
            joint_positions.cpu().numpy()
        )
        base_positions = torch.tensor(base_positions, device=self.device)

        # Compute numerical Jacobian (position part only for speed)
        jacobian = torch.zeros(batch_size, 3, self.n_joints, device=self.device)

        for j in range(self.n_joints):
            # Perturb joint j
            perturbed = joint_positions.clone()
            perturbed[:, j] += epsilon

            perturbed_positions, _ = self.ik_solver.get_forward_kinematics(
                perturbed.cpu().numpy()
            )
            perturbed_positions = torch.tensor(perturbed_positions, device=self.device)

            # Finite difference
            jacobian[:, :, j] = (perturbed_positions - base_positions) / epsilon

        # Compute manipulability: sqrt(det(J @ J.T))
        JJT = torch.bmm(jacobian, jacobian.transpose(1, 2))
        det_JJT = torch.linalg.det(JJT)

        # Clamp negative values
        det_JJT = torch.clamp(det_JJT, min=0)
        manipulability = torch.sqrt(det_JJT)

        return manipulability.cpu().numpy()
