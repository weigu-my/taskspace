"""
Voxel-based Reachability Analyzer.

This module performs comprehensive reachability analysis using voxel grid
sampling and GPU-accelerated IK solving.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from .config import (
    ReachabilityConfig,
    VoxelConfig,
    IKConfig,
    OrientationConfig,
    OrientationMode
)
from .urdf_parser import URDFParser, RobotConfig
from .ik_solver import MultiSeedIKSolver, BatchIKResult
from .manipulability import ManipulabilityCalculator
from .utils import (
    euler_to_quaternion,
    generate_spherical_orientations,
    generate_random_orientations,
    ProgressBar
)


@dataclass
class ReachabilityResult:
    """Complete results of reachability analysis."""
    # Grid information
    grid_points: np.ndarray           # [n_points, 3] all voxel center points
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz) voxel grid dimensions

    # Basic reachability
    reachable_mask: np.ndarray        # [n_points] bool mask of reachable points
    reachable_points: np.ndarray      # [n_reachable, 3] reachable positions

    # Multi-seed IK results
    num_solutions: np.ndarray         # [n_points, n_orientations] solutions per pose

    # Dexterity (number of reachable orientations per position)
    dexterity: np.ndarray             # [n_points] number of orientations reachable

    # Manipulability (Yoshikawa index)
    manipulability: np.ndarray        # [n_points] manipulability value

    # Best IK solutions
    best_solutions: np.ndarray        # [n_points, n_joints] best joint config

    # Statistics
    total_points: int
    reachable_count: int
    reachability_ratio: float
    max_dexterity: int
    mean_dexterity: float
    max_manipulability: float
    mean_manipulability: float
    computation_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_points': self.total_points,
            'reachable_count': self.reachable_count,
            'reachability_ratio': self.reachability_ratio,
            'max_dexterity': int(self.max_dexterity),
            'mean_dexterity': float(self.mean_dexterity),
            'max_manipulability': float(self.max_manipulability),
            'mean_manipulability': float(self.mean_manipulability),
            'computation_time': self.computation_time,
            'grid_shape': list(self.grid_shape),
        }


class ReachabilityAnalyzer:
    """
    Main class for performing reachability analysis.

    This analyzer:
    1. Parses URDF and creates cuRobo configuration
    2. Generates voxel grid for workspace sampling
    3. Tests reachability using multi-seed IK with GPU acceleration
    4. Computes dexterity (number of reachable orientations per position)
    5. Computes manipulability (Yoshikawa index) for reachable positions
    """

    def __init__(self, config: ReachabilityConfig):
        """
        Initialize reachability analyzer.

        Args:
            config: Reachability analysis configuration
        """
        self.config = config
        self.robot_config: Optional[RobotConfig] = None
        self.ik_solver: Optional[MultiSeedIKSolver] = None
        self.manipulability_calc: Optional[ManipulabilityCalculator] = None

        self._initialize()

    def _initialize(self):
        """Initialize all components."""
        print("=" * 60)
        print("Initializing Reachability Analyzer")
        print("=" * 60)

        # Parse URDF
        print(f"\n[1/4] Parsing URDF: {self.config.urdf_path}")
        parser = URDFParser(self.config.urdf_path)
        self.robot_config = parser.parse(
            base_link=self.config.base_link,
            ee_link=self.config.ee_link
        )

        print(f"  Robot: {self.robot_config.name}")
        print(f"  Base link: {self.robot_config.base_link}")
        print(f"  EE link: {self.robot_config.ee_link}")
        print(f"  DOF: {self.robot_config.num_dof}")
        print(f"  Active joints: {[j.name for j in self.robot_config.active_joints]}")

        # Initialize IK solver
        print(f"\n[2/4] Initializing IK solver...")
        self.ik_solver = MultiSeedIKSolver(
            robot_config=self.robot_config,
            ik_config=self.config.ik,
            device=self.config.device
        )

        # Initialize manipulability calculator
        if self.config.compute_manipulability:
            print(f"\n[3/4] Initializing manipulability calculator...")
            self.manipulability_calc = ManipulabilityCalculator(
                robot_config=self.robot_config,
                device=self.config.device
            )
        else:
            print(f"\n[3/4] Skipping manipulability (disabled)")

        # Auto-detect workspace bounds if needed
        if self.config.voxel.auto_detect:
            print(f"\n[4/4] Auto-detecting workspace bounds...")
            x_range, y_range, z_range = self.ik_solver.estimate_workspace_bounds(
                n_samples=self.config.voxel.fk_samples,
                padding=self.config.voxel.padding
            )
            self.config.voxel.x_range = x_range
            self.config.voxel.y_range = y_range
            self.config.voxel.z_range = z_range
        else:
            print(f"\n[4/4] Using specified workspace bounds")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)

    def _generate_orientations(self) -> np.ndarray:
        """Generate orientation samples based on configuration."""
        mode = self.config.orientation.mode

        if mode == OrientationMode.FIXED:
            # Single fixed orientation
            return np.array([self.config.orientation.fixed_orientation], dtype=np.float32)

        elif mode == OrientationMode.MULTI_FIXED:
            # Multiple predefined orientations from Euler angles
            orientations = []
            for euler in self.config.orientation.multi_orientations_euler:
                quat = euler_to_quaternion(euler[0], euler[1], euler[2], degrees=True)
                orientations.append(quat)
            return np.array(orientations, dtype=np.float32)

        elif mode == OrientationMode.SPHERICAL:
            # Evenly distributed on sphere
            return generate_spherical_orientations(
                self.config.orientation.num_spherical_samples
            )

        elif mode == OrientationMode.RANDOM:
            # Random orientations
            return generate_random_orientations(
                self.config.orientation.num_random_samples
            )

        else:
            raise ValueError(f"Unknown orientation mode: {mode}")

    def analyze(self) -> ReachabilityResult:
        """
        Perform complete reachability analysis.

        Returns:
            ReachabilityResult with all computed metrics
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("Starting Reachability Analysis")
        print("=" * 60)

        # Generate voxel grid
        print("\n[Step 1] Generating voxel grid...")
        grid_points = self.config.voxel.get_grid_points()
        grid_shape = self.config.voxel.get_grid_shape()
        n_points = len(grid_points)

        print(f"  Grid shape: {grid_shape}")
        print(f"  Resolution: {self.config.voxel.resolution}m")
        print(f"  Total voxels: {n_points}")

        # Generate orientations
        print("\n[Step 2] Generating orientations...")
        orientations = self._generate_orientations()
        n_orientations = len(orientations)
        print(f"  Mode: {self.config.orientation.mode.value}")
        print(f"  Number of orientations: {n_orientations}")

        # Perform IK solving
        print("\n[Step 3] Solving IK for all positions and orientations...")
        reachable_mask, dexterity, num_solutions = self.ik_solver.solve_batch_multi_orientation(
            positions=grid_points,
            orientations=orientations,
            batch_size=self.config.ik.batch_size
        )

        reachable_points = grid_points[reachable_mask]
        reachable_count = int(np.sum(reachable_mask))

        print(f"  Reachable points: {reachable_count} / {n_points} ({reachable_count/n_points*100:.1f}%)")
        print(f"  Max dexterity: {np.max(dexterity)} / {n_orientations}")

        # Get best solutions for manipulability calculation
        print("\n[Step 4] Computing best IK solutions...")
        best_solutions = self._compute_best_solutions(grid_points, orientations, reachable_mask)

        # Compute manipulability
        manipulability = np.zeros(n_points, dtype=np.float32)
        if self.config.compute_manipulability and self.manipulability_calc is not None:
            print("\n[Step 5] Computing manipulability...")
            manipulability = self.manipulability_calc.compute_for_best_solutions(
                best_solutions=best_solutions,
                reachable_mask=reachable_mask
            )
            print(f"  Max manipulability: {np.max(manipulability):.4f}")
            print(f"  Mean manipulability (reachable): {np.mean(manipulability[reachable_mask]):.4f}")
        else:
            print("\n[Step 5] Skipping manipulability computation")

        computation_time = time.time() - start_time

        # Compute statistics
        reachable_dexterity = dexterity[reachable_mask]
        reachable_manip = manipulability[reachable_mask]

        result = ReachabilityResult(
            grid_points=grid_points,
            grid_shape=grid_shape,
            reachable_mask=reachable_mask,
            reachable_points=reachable_points,
            num_solutions=num_solutions,
            dexterity=dexterity,
            manipulability=manipulability,
            best_solutions=best_solutions,
            total_points=n_points,
            reachable_count=reachable_count,
            reachability_ratio=reachable_count / n_points,
            max_dexterity=int(np.max(dexterity)) if reachable_count > 0 else 0,
            mean_dexterity=float(np.mean(reachable_dexterity)) if reachable_count > 0 else 0,
            max_manipulability=float(np.max(reachable_manip)) if reachable_count > 0 else 0,
            mean_manipulability=float(np.mean(reachable_manip)) if reachable_count > 0 else 0,
            computation_time=computation_time
        )

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print(f"  Total time: {computation_time:.2f}s")
        print(f"  Reachability: {result.reachability_ratio*100:.1f}%")
        print(f"  Max dexterity: {result.max_dexterity}")
        print(f"  Mean dexterity: {result.mean_dexterity:.2f}")

        return result

    def _compute_best_solutions(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        reachable_mask: np.ndarray
    ) -> np.ndarray:
        """Compute best IK solution for each reachable position."""
        n_positions = len(positions)
        n_joints = self.robot_config.num_dof
        best_solutions = np.zeros((n_positions, n_joints), dtype=np.float32)

        # For each reachable position, get the best solution
        reachable_indices = np.where(reachable_mask)[0]

        # Use first orientation that succeeds
        default_orientation = orientations[0]

        # Batch solve for reachable positions
        if len(reachable_indices) > 0:
            reachable_positions = positions[reachable_indices]
            ori_batch = np.tile(default_orientation, (len(reachable_indices), 1))

            result = self.ik_solver.solve_batch(
                positions=reachable_positions,
                orientations=ori_batch,
                return_all_solutions=False
            )

            best_solutions[reachable_indices] = result.best_solutions

        return best_solutions

    def save_results(self, result: ReachabilityResult, prefix: str = "reachability"):
        """
        Save analysis results to files.

        Args:
            result: ReachabilityResult to save
            prefix: Filename prefix
        """
        output_dir = self.config.output_dir
        print(f"\nSaving results to {output_dir}/")

        # Save numpy arrays
        np.save(os.path.join(output_dir, f"{prefix}_grid_points.npy"), result.grid_points)
        np.save(os.path.join(output_dir, f"{prefix}_reachable_mask.npy"), result.reachable_mask)
        np.save(os.path.join(output_dir, f"{prefix}_reachable_points.npy"), result.reachable_points)
        np.save(os.path.join(output_dir, f"{prefix}_dexterity.npy"), result.dexterity)
        np.save(os.path.join(output_dir, f"{prefix}_manipulability.npy"), result.manipulability)
        np.save(os.path.join(output_dir, f"{prefix}_best_solutions.npy"), result.best_solutions)

        # Save CSV for reachable points with metrics
        import pandas as pd
        reachable_indices = np.where(result.reachable_mask)[0]
        df = pd.DataFrame({
            'x': result.grid_points[reachable_indices, 0],
            'y': result.grid_points[reachable_indices, 1],
            'z': result.grid_points[reachable_indices, 2],
            'dexterity': result.dexterity[reachable_indices],
            'manipulability': result.manipulability[reachable_indices]
        })
        df.to_csv(os.path.join(output_dir, f"{prefix}_data.csv"), index=False)

        # Save statistics JSON
        stats = result.to_dict()
        stats['robot_name'] = self.robot_config.name
        stats['urdf_path'] = self.robot_config.urdf_path
        stats['ee_link'] = self.robot_config.ee_link
        stats['base_link'] = self.robot_config.base_link
        stats['num_dof'] = self.robot_config.num_dof
        stats['voxel_resolution'] = self.config.voxel.resolution
        stats['num_ik_seeds'] = self.config.ik.num_seeds

        with open(os.path.join(output_dir, f"{prefix}_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  Saved: {prefix}_grid_points.npy")
        print(f"  Saved: {prefix}_reachable_points.npy")
        print(f"  Saved: {prefix}_dexterity.npy")
        print(f"  Saved: {prefix}_manipulability.npy")
        print(f"  Saved: {prefix}_data.csv")
        print(f"  Saved: {prefix}_stats.json")

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        ee_link: str = "",
        base_link: str = "",
        resolution: float = 0.05,
        num_seeds: int = 32,
        device: str = "cuda:0",
        output_dir: str = "./reachability_output"
    ) -> "ReachabilityAnalyzer":
        """
        Convenience constructor from URDF path.

        Args:
            urdf_path: Path to URDF file
            ee_link: End-effector link name (auto-detect if empty)
            base_link: Base link name (auto-detect if empty)
            resolution: Voxel resolution in meters
            num_seeds: Number of IK seeds
            device: Computation device
            output_dir: Output directory

        Returns:
            Configured ReachabilityAnalyzer
        """
        config = ReachabilityConfig(
            urdf_path=urdf_path,
            ee_link=ee_link,
            base_link=base_link,
            device=device,
            output_dir=output_dir
        )
        config.voxel.resolution = resolution
        config.voxel.auto_detect = True
        config.ik.num_seeds = num_seeds

        return cls(config)


def analyze_urdf(
    urdf_path: str,
    ee_link: str = "",
    base_link: str = "",
    resolution: float = 0.05,
    num_seeds: int = 32,
    output_dir: str = "./reachability_output",
    device: str = "cuda:0"
) -> ReachabilityResult:
    """
    Convenience function to analyze a URDF file.

    Args:
        urdf_path: Path to URDF file
        ee_link: End-effector link name
        base_link: Base link name
        resolution: Voxel resolution
        num_seeds: Number of IK seeds
        output_dir: Output directory
        device: Computation device

    Returns:
        ReachabilityResult
    """
    analyzer = ReachabilityAnalyzer.from_urdf(
        urdf_path=urdf_path,
        ee_link=ee_link,
        base_link=base_link,
        resolution=resolution,
        num_seeds=num_seeds,
        device=device,
        output_dir=output_dir
    )

    result = analyzer.analyze()
    analyzer.save_results(result)

    return result
