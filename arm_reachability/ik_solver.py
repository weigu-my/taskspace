"""
Multi-seed IK Solver using cuRobo with GPU acceleration.

This module provides IK solving capabilities with multiple seeds to find
multiple solutions for redundant manipulators.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from .config import IKConfig, ReachabilityConfig
from .urdf_parser import RobotConfig, CuroboConfigGenerator


@dataclass
class IKResult:
    """Result of IK solving for a single target pose."""
    target_position: np.ndarray       # [3]
    target_orientation: np.ndarray    # [4] quaternion (w, x, y, z)
    success: bool                     # Whether at least one solution was found
    num_solutions: int                # Number of valid IK solutions found
    solutions: np.ndarray             # [num_seeds, n_joints] joint configurations
    solution_mask: np.ndarray         # [num_seeds] boolean mask of valid solutions
    position_errors: np.ndarray       # [num_seeds] position errors
    rotation_errors: np.ndarray       # [num_seeds] rotation errors


@dataclass
class BatchIKResult:
    """Result of batch IK solving."""
    success_mask: np.ndarray          # [batch_size] whether each pose has at least one solution
    num_solutions: np.ndarray         # [batch_size] number of solutions for each pose
    best_solutions: np.ndarray        # [batch_size, n_joints] best solution for each pose
    all_solutions: np.ndarray         # [batch_size, num_seeds, n_joints] all solutions
    solution_masks: np.ndarray        # [batch_size, num_seeds] validity mask
    solve_time: float                 # Total solving time


class MultiSeedIKSolver:
    """
    Multi-seed IK solver using cuRobo for GPU-accelerated computation.

    This solver runs IK from multiple initial seeds to find multiple solutions,
    which is essential for analyzing reachability of redundant manipulators.
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        ik_config: IKConfig,
        device: str = "cuda:0"
    ):
        """
        Initialize the multi-seed IK solver.

        Args:
            robot_config: Robot configuration from URDF parser
            ik_config: IK solver configuration
            device: Torch device for computation
        """
        self.robot_config = robot_config
        self.ik_config = ik_config
        self.device = device
        self.ik_solver = None
        self.tensor_args = None

        self._initialize_solver()

    def _initialize_solver(self):
        """Initialize cuRobo IK solver."""
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import RobotConfig as CuroboRobotConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

            # Setup tensor arguments
            self.tensor_args = TensorDeviceType(
                device=torch.device(self.device),
                dtype=torch.float32
            )

            # Generate cuRobo configuration
            config_generator = CuroboConfigGenerator(self.robot_config)
            curobo_config = config_generator.generate()

            # Create robot configuration for cuRobo
            robot_cfg = self._create_curobo_robot_config(curobo_config)

            # Create IK solver configuration
            ik_solver_config = IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=None,
                tensor_args=self.tensor_args,
                num_seeds=self.ik_config.num_seeds,
                position_threshold=self.ik_config.position_threshold,
                rotation_threshold=self.ik_config.rotation_threshold,
                use_cuda_graph=True,
                self_collision_check=self.ik_config.self_collision_check,
                self_collision_opt=self.ik_config.self_collision_check,
            )

            self.ik_solver = IKSolver(ik_solver_config)
            self.n_joints = len(self.robot_config.active_joints)

            print(f"[IK Solver] Initialized with {self.ik_config.num_seeds} seeds")
            print(f"[IK Solver] Robot: {self.robot_config.name}, DOF: {self.n_joints}")
            print(f"[IK Solver] Device: {self.device}")

        except ImportError as e:
            print(f"[Warning] cuRobo not available: {e}")
            print("[Warning] Using fallback dummy solver")
            self._initialize_dummy_solver()

    def _create_curobo_robot_config(self, config_dict: Dict) -> Any:
        """Create cuRobo robot configuration from dictionary."""
        from curobo.types.robot import RobotConfig as CuroboRobotConfig

        # Get active joints
        active_joints = self.robot_config.active_joints
        joint_names = [j.name for j in active_joints]

        # Build joint limits
        joint_limits = {
            'position': [
                [j.lower_limit for j in active_joints],
                [j.upper_limit for j in active_joints]
            ],
            'velocity': [[j.velocity_limit for j in active_joints]],
        }

        # Create configuration dictionary
        robot_cfg_dict = {
            'kinematics': {
                'urdf_path': self.robot_config.urdf_path,
                'asset_root_path': self.robot_config.urdf_dir,
                'base_link': self.robot_config.base_link,
                'ee_link': self.robot_config.ee_link,
                'cspace': {
                    'joint_names': joint_names,
                    'retract_config': [(j.lower_limit + j.upper_limit) / 2 for j in active_joints],
                    'null_space_weight': [1.0] * len(joint_names),
                    'cspace_distance_weight': [1.0] * len(joint_names),
                },
            },
        }

        robot_cfg = CuroboRobotConfig.from_dict(
            robot_cfg_dict,
            self.tensor_args
        )

        return robot_cfg

    def _initialize_dummy_solver(self):
        """Initialize a dummy solver for testing without cuRobo."""
        self.ik_solver = None
        self.n_joints = len(self.robot_config.active_joints)
        print("[Dummy Solver] Initialized for testing")

    def solve_single(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """
        Solve IK for a single target pose with multiple seeds.

        Args:
            position: Target position [3]
            orientation: Target orientation quaternion [4] (w, x, y, z)

        Returns:
            IKResult with all solutions found
        """
        if self.ik_solver is None:
            return self._solve_single_dummy(position, orientation)

        from curobo.types.math import Pose

        # Create goal pose
        goal_pose = Pose(
            position=torch.tensor(
                position.reshape(1, 3),
                dtype=torch.float32,
                device=self.device
            ),
            quaternion=torch.tensor(
                orientation.reshape(1, 4),
                dtype=torch.float32,
                device=self.device
            )
        )

        # Solve IK
        result = self.ik_solver.solve_batch(goal_pose)

        # Extract results
        success = result.success.cpu().numpy()[0]
        solutions = result.solution.cpu().numpy()[0]  # [num_seeds, n_joints]

        # Get position and rotation errors for all seeds
        if hasattr(result, 'position_error') and result.position_error is not None:
            position_errors = result.position_error.cpu().numpy().flatten()
        else:
            position_errors = np.zeros(self.ik_config.num_seeds)

        if hasattr(result, 'rotation_error') and result.rotation_error is not None:
            rotation_errors = result.rotation_error.cpu().numpy().flatten()
        else:
            rotation_errors = np.zeros(self.ik_config.num_seeds)

        # Create solution mask based on error thresholds
        solution_mask = (
            (position_errors < self.ik_config.position_threshold) &
            (rotation_errors < self.ik_config.rotation_threshold)
        )

        num_solutions = int(np.sum(solution_mask))

        return IKResult(
            target_position=position,
            target_orientation=orientation,
            success=success or num_solutions > 0,
            num_solutions=num_solutions,
            solutions=solutions,
            solution_mask=solution_mask,
            position_errors=position_errors,
            rotation_errors=rotation_errors
        )

    def _solve_single_dummy(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """Dummy solver for testing."""
        # Simulate random results
        success = np.random.random() > 0.3
        num_solutions = np.random.randint(0, self.ik_config.num_seeds // 2) if success else 0

        solutions = np.random.uniform(
            -np.pi, np.pi,
            (self.ik_config.num_seeds, self.n_joints)
        )

        solution_mask = np.zeros(self.ik_config.num_seeds, dtype=bool)
        solution_mask[:num_solutions] = True

        return IKResult(
            target_position=position,
            target_orientation=orientation,
            success=success,
            num_solutions=num_solutions,
            solutions=solutions.astype(np.float32),
            solution_mask=solution_mask,
            position_errors=np.random.uniform(0, 0.01, self.ik_config.num_seeds).astype(np.float32),
            rotation_errors=np.random.uniform(0, 0.1, self.ik_config.num_seeds).astype(np.float32)
        )

    def solve_batch(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        return_all_solutions: bool = True
    ) -> BatchIKResult:
        """
        Solve IK for a batch of target poses with GPU acceleration.

        Args:
            positions: Target positions [batch_size, 3]
            orientations: Target orientation quaternions [batch_size, 4] (w, x, y, z)
            return_all_solutions: If True, return all solutions from all seeds

        Returns:
            BatchIKResult with solutions for all poses
        """
        start_time = time.time()
        batch_size = len(positions)

        if self.ik_solver is None:
            return self._solve_batch_dummy(positions, orientations, return_all_solutions)

        from curobo.types.math import Pose

        # Create goal poses
        goal_pose = Pose(
            position=torch.tensor(
                positions,
                dtype=torch.float32,
                device=self.device
            ),
            quaternion=torch.tensor(
                orientations,
                dtype=torch.float32,
                device=self.device
            )
        )

        # Solve IK batch
        result = self.ik_solver.solve_batch(goal_pose)

        # Extract results
        success_mask = result.success.cpu().numpy()
        solutions = result.solution.cpu().numpy()  # [batch_size, num_seeds, n_joints]

        # Best solution for each pose (first successful seed)
        best_solutions = solutions[:, 0, :]  # Take first seed as best

        # Count solutions per pose
        num_solutions = np.zeros(batch_size, dtype=np.int32)

        # Solution validity mask
        solution_masks = np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool)

        # For positions that succeeded, count all valid seeds
        if hasattr(result, 'position_error') and result.position_error is not None:
            position_errors = result.position_error.cpu().numpy()
            rotation_errors = result.rotation_error.cpu().numpy() if hasattr(result, 'rotation_error') else np.zeros_like(position_errors)

            for i in range(batch_size):
                valid = (
                    (position_errors[i] < self.ik_config.position_threshold) &
                    (rotation_errors[i] < self.ik_config.rotation_threshold)
                )
                solution_masks[i] = valid
                num_solutions[i] = np.sum(valid)
        else:
            # Use success mask to estimate solutions
            num_solutions = success_mask.astype(np.int32)
            solution_masks[:, 0] = success_mask

        solve_time = time.time() - start_time

        return BatchIKResult(
            success_mask=success_mask,
            num_solutions=num_solutions,
            best_solutions=best_solutions.astype(np.float32),
            all_solutions=solutions.astype(np.float32) if return_all_solutions else None,
            solution_masks=solution_masks,
            solve_time=solve_time
        )

    def _solve_batch_dummy(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        return_all_solutions: bool
    ) -> BatchIKResult:
        """Dummy batch solver for testing."""
        start_time = time.time()
        batch_size = len(positions)

        # Simulate results based on distance from origin
        distances = np.linalg.norm(positions, axis=1)
        success_prob = np.clip(1.0 - distances / 2.0, 0.1, 0.9)
        success_mask = np.random.random(batch_size) < success_prob

        num_solutions = np.where(
            success_mask,
            np.random.randint(1, self.ik_config.num_seeds // 2, batch_size),
            0
        ).astype(np.int32)

        best_solutions = np.random.uniform(
            -np.pi, np.pi,
            (batch_size, self.n_joints)
        ).astype(np.float32)

        if return_all_solutions:
            all_solutions = np.random.uniform(
                -np.pi, np.pi,
                (batch_size, self.ik_config.num_seeds, self.n_joints)
            ).astype(np.float32)
        else:
            all_solutions = None

        solution_masks = np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool)
        for i in range(batch_size):
            solution_masks[i, :num_solutions[i]] = True

        solve_time = time.time() - start_time

        return BatchIKResult(
            success_mask=success_mask,
            num_solutions=num_solutions,
            best_solutions=best_solutions,
            all_solutions=all_solutions,
            solution_masks=solution_masks,
            solve_time=solve_time
        )

    def solve_batch_multi_orientation(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        batch_size: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve IK for multiple positions with multiple orientations.

        This is the main method for reachability analysis, testing each
        position with multiple orientations.

        Args:
            positions: Grid positions [n_positions, 3]
            orientations: Orientations to test [n_orientations, 4]
            batch_size: Batch size for GPU processing

        Returns:
            Tuple of:
                - reachable_mask: [n_positions] True if any orientation succeeds
                - dexterity: [n_positions] number of reachable orientations
                - num_solutions: [n_positions, n_orientations] solutions per pose
        """
        n_positions = len(positions)
        n_orientations = len(orientations)

        print(f"[IK Solver] Testing {n_positions} positions × {n_orientations} orientations")
        print(f"[IK Solver] Total poses: {n_positions * n_orientations}")

        # Create all position-orientation combinations
        # positions: [n_positions, 3] -> [n_positions, n_orientations, 3]
        # orientations: [n_orientations, 4] -> [n_positions, n_orientations, 4]
        pos_expanded = np.tile(positions[:, np.newaxis, :], (1, n_orientations, 1))
        ori_expanded = np.tile(orientations[np.newaxis, :, :], (n_positions, 1, 1))

        # Flatten for batch processing
        all_positions = pos_expanded.reshape(-1, 3)
        all_orientations = ori_expanded.reshape(-1, 4)
        total_poses = len(all_positions)

        # Process in batches
        all_success = []
        all_num_solutions = []

        num_batches = (total_poses + batch_size - 1) // batch_size
        print(f"[IK Solver] Processing {num_batches} batches...")

        for i in range(0, total_poses, batch_size):
            end_idx = min(i + batch_size, total_poses)
            batch_positions = all_positions[i:end_idx]
            batch_orientations = all_orientations[i:end_idx]

            result = self.solve_batch(
                batch_positions,
                batch_orientations,
                return_all_solutions=False
            )

            all_success.append(result.success_mask)
            all_num_solutions.append(result.num_solutions)

            # Progress
            progress = (i + batch_size) / total_poses * 100
            print(f"\r[IK Solver] Progress: {min(progress, 100):.1f}%", end='', flush=True)

        print()  # New line

        # Concatenate results
        success = np.concatenate(all_success)
        num_solutions = np.concatenate(all_num_solutions)

        # Reshape to [n_positions, n_orientations]
        success_grid = success.reshape(n_positions, n_orientations)
        num_solutions_grid = num_solutions.reshape(n_positions, n_orientations)

        # Compute per-position metrics
        reachable_mask = np.any(success_grid, axis=1)
        dexterity = np.sum(success_grid, axis=1)

        print(f"[IK Solver] Reachable positions: {np.sum(reachable_mask)} / {n_positions}")
        print(f"[IK Solver] Max dexterity: {np.max(dexterity)}")

        return reachable_mask, dexterity, num_solutions_grid

    def get_forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint positions.

        Args:
            joint_positions: Joint positions [batch_size, n_joints]

        Returns:
            Tuple of:
                - positions: End-effector positions [batch_size, 3]
                - orientations: End-effector orientations [batch_size, 4]
        """
        if self.ik_solver is None:
            # Dummy FK
            batch_size = len(joint_positions)
            positions = np.random.uniform(-1, 1, (batch_size, 3)).astype(np.float32)
            orientations = np.zeros((batch_size, 4), dtype=np.float32)
            orientations[:, 0] = 1.0  # w = 1 for identity quaternion
            return positions, orientations

        # Use cuRobo kinematics
        q = torch.tensor(
            joint_positions,
            dtype=torch.float32,
            device=self.device
        )

        state = self.ik_solver.fk(q)

        positions = state.ee_position.cpu().numpy()
        orientations = state.ee_quaternion.cpu().numpy()  # (w, x, y, z)

        return positions, orientations

    def sample_workspace(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the robot's workspace using random FK sampling.

        Args:
            n_samples: Number of random joint configurations to sample

        Returns:
            Tuple of:
                - positions: Sampled end-effector positions [n_samples, 3]
                - joint_configs: Corresponding joint configurations [n_samples, n_joints]
        """
        # Generate random joint configurations within limits
        active_joints = self.robot_config.active_joints
        joint_configs = np.zeros((n_samples, self.n_joints), dtype=np.float32)

        for i, joint in enumerate(active_joints):
            joint_configs[:, i] = np.random.uniform(
                joint.lower_limit,
                joint.upper_limit,
                n_samples
            )

        # Compute FK
        positions, _ = self.get_forward_kinematics(joint_configs)

        return positions, joint_configs

    def estimate_workspace_bounds(
        self,
        n_samples: int = 10000,
        padding: float = 0.1
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Estimate workspace bounds using FK sampling.

        Args:
            n_samples: Number of samples for estimation
            padding: Padding ratio to add to bounds

        Returns:
            Tuple of (x_range, y_range, z_range) as (min, max) tuples
        """
        print(f"[IK Solver] Estimating workspace bounds with {n_samples} FK samples...")

        positions, _ = self.sample_workspace(n_samples)

        x_min, y_min, z_min = positions.min(axis=0)
        x_max, y_max, z_max = positions.max(axis=0)

        # Add padding
        x_pad = (x_max - x_min) * padding
        y_pad = (y_max - y_min) * padding
        z_pad = (z_max - z_min) * padding

        x_range = (x_min - x_pad, x_max + x_pad)
        y_range = (y_min - y_pad, y_max + y_pad)
        z_range = (z_min - z_pad, z_max + z_pad)

        print(f"[IK Solver] Estimated bounds:")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range
