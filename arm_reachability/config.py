"""
Configuration classes for reachability analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class OrientationMode(Enum):
    """Orientation sampling modes for reachability analysis."""
    FIXED = "fixed"                # Single fixed orientation
    MULTI_FIXED = "multi_fixed"    # Multiple predefined orientations
    SPHERICAL = "spherical"        # Spherical sampling
    RANDOM = "random"              # Random orientations


@dataclass
class VoxelConfig:
    """Configuration for voxel-based workspace sampling."""

    # Workspace bounds [min, max] for x, y, z (meters)
    x_range: Tuple[float, float] = (-1.0, 1.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    z_range: Tuple[float, float] = (0.0, 1.5)

    # Voxel resolution (meters)
    resolution: float = 0.05

    # Auto-detect workspace from FK sampling
    auto_detect: bool = True

    # FK sampling parameters for auto-detection
    fk_samples: int = 10000
    padding: float = 0.1  # Padding ratio for detected bounds

    def get_grid_points(self) -> np.ndarray:
        """Generate voxel grid points."""
        x = np.arange(self.x_range[0], self.x_range[1] + self.resolution, self.resolution)
        y = np.arange(self.y_range[0], self.y_range[1] + self.resolution, self.resolution)
        z = np.arange(self.z_range[0], self.z_range[1] + self.resolution, self.resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

        return points.astype(np.float32)

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the voxel grid."""
        nx = int(np.ceil((self.x_range[1] - self.x_range[0]) / self.resolution)) + 1
        ny = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.resolution)) + 1
        nz = int(np.ceil((self.z_range[1] - self.z_range[0]) / self.resolution)) + 1
        return (nx, ny, nz)


@dataclass
class IKConfig:
    """Configuration for multi-seed IK solving."""

    # Number of IK seeds (for finding multiple solutions)
    num_seeds: int = 32

    # IK solver parameters
    position_threshold: float = 0.005  # meters
    rotation_threshold: float = 0.05   # radians

    # cuRobo IK solver settings
    num_graph_seeds: int = 12
    num_trajopt_seeds: int = 8

    # Batch size for GPU processing
    batch_size: int = 1024

    # Maximum iterations
    max_iterations: int = 100

    # Use parallel environment for batch solving
    use_parallel_env: bool = True

    # Collision checking
    collision_check: bool = True
    self_collision_check: bool = True


@dataclass
class OrientationConfig:
    """Configuration for orientation sampling."""

    mode: OrientationMode = OrientationMode.MULTI_FIXED

    # For FIXED mode: single quaternion [w, x, y, z]
    fixed_orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    # For MULTI_FIXED mode: list of Euler angles [roll, pitch, yaw] in degrees
    multi_orientations_euler: List[List[float]] = field(default_factory=lambda: [
        [0, 0, 0],       # Pointing down (Z-)
        [180, 0, 0],     # Pointing up (Z+)
        [90, 0, 0],      # Pointing forward (X+)
        [-90, 0, 0],     # Pointing backward (X-)
        [0, 90, 0],      # Pointing left (Y+)
        [0, -90, 0],     # Pointing right (Y-)
    ])

    # For SPHERICAL mode: number of orientations to sample
    num_spherical_samples: int = 26  # Icosahedron vertices + face centers

    # For RANDOM mode: number of random orientations
    num_random_samples: int = 10


@dataclass
class ReachabilityConfig:
    """Main configuration for reachability analysis."""

    # URDF file path
    urdf_path: str = ""

    # End-effector link name (auto-detect if empty)
    ee_link: str = ""

    # Base link name (auto-detect if empty)
    base_link: str = ""

    # Voxel configuration
    voxel: VoxelConfig = field(default_factory=VoxelConfig)

    # IK configuration
    ik: IKConfig = field(default_factory=IKConfig)

    # Orientation configuration
    orientation: OrientationConfig = field(default_factory=OrientationConfig)

    # Output directory
    output_dir: str = "./reachability_output"

    # Device for computation
    device: str = "cuda:0"

    # Whether to compute manipulability
    compute_manipulability: bool = True

    # Verbose output
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReachabilityConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        if 'urdf_path' in data:
            config.urdf_path = data['urdf_path']
        if 'ee_link' in data:
            config.ee_link = data['ee_link']
        if 'base_link' in data:
            config.base_link = data['base_link']
        if 'output_dir' in data:
            config.output_dir = data['output_dir']
        if 'device' in data:
            config.device = data['device']
        if 'compute_manipulability' in data:
            config.compute_manipulability = data['compute_manipulability']
        if 'verbose' in data:
            config.verbose = data['verbose']

        # Voxel config
        if 'voxel' in data:
            v = data['voxel']
            if 'x_range' in v:
                config.voxel.x_range = tuple(v['x_range'])
            if 'y_range' in v:
                config.voxel.y_range = tuple(v['y_range'])
            if 'z_range' in v:
                config.voxel.z_range = tuple(v['z_range'])
            if 'resolution' in v:
                config.voxel.resolution = v['resolution']
            if 'auto_detect' in v:
                config.voxel.auto_detect = v['auto_detect']
            if 'fk_samples' in v:
                config.voxel.fk_samples = v['fk_samples']
            if 'padding' in v:
                config.voxel.padding = v['padding']

        # IK config
        if 'ik' in data:
            ik = data['ik']
            if 'num_seeds' in ik:
                config.ik.num_seeds = ik['num_seeds']
            if 'position_threshold' in ik:
                config.ik.position_threshold = ik['position_threshold']
            if 'rotation_threshold' in ik:
                config.ik.rotation_threshold = ik['rotation_threshold']
            if 'batch_size' in ik:
                config.ik.batch_size = ik['batch_size']
            if 'max_iterations' in ik:
                config.ik.max_iterations = ik['max_iterations']
            if 'collision_check' in ik:
                config.ik.collision_check = ik['collision_check']
            if 'self_collision_check' in ik:
                config.ik.self_collision_check = ik['self_collision_check']

        # Orientation config
        if 'orientation' in data:
            o = data['orientation']
            if 'mode' in o:
                config.orientation.mode = OrientationMode(o['mode'])
            if 'fixed_orientation' in o:
                config.orientation.fixed_orientation = o['fixed_orientation']
            if 'multi_orientations_euler' in o:
                config.orientation.multi_orientations_euler = o['multi_orientations_euler']
            if 'num_spherical_samples' in o:
                config.orientation.num_spherical_samples = o['num_spherical_samples']
            if 'num_random_samples' in o:
                config.orientation.num_random_samples = o['num_random_samples']

        return config

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        import yaml

        data = {
            'urdf_path': self.urdf_path,
            'ee_link': self.ee_link,
            'base_link': self.base_link,
            'output_dir': self.output_dir,
            'device': self.device,
            'compute_manipulability': self.compute_manipulability,
            'verbose': self.verbose,
            'voxel': {
                'x_range': list(self.voxel.x_range),
                'y_range': list(self.voxel.y_range),
                'z_range': list(self.voxel.z_range),
                'resolution': self.voxel.resolution,
                'auto_detect': self.voxel.auto_detect,
                'fk_samples': self.voxel.fk_samples,
                'padding': self.voxel.padding,
            },
            'ik': {
                'num_seeds': self.ik.num_seeds,
                'position_threshold': self.ik.position_threshold,
                'rotation_threshold': self.ik.rotation_threshold,
                'batch_size': self.ik.batch_size,
                'max_iterations': self.ik.max_iterations,
                'collision_check': self.ik.collision_check,
                'self_collision_check': self.ik.self_collision_check,
            },
            'orientation': {
                'mode': self.orientation.mode.value,
                'fixed_orientation': self.orientation.fixed_orientation,
                'multi_orientations_euler': self.orientation.multi_orientations_euler,
                'num_spherical_samples': self.orientation.num_spherical_samples,
                'num_random_samples': self.orientation.num_random_samples,
            }
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
