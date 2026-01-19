"""
Utility functions for reachability analysis.
"""

import numpy as np
from typing import List, Tuple, Optional
import torch


def euler_to_quaternion(roll: float, pitch: float, yaw: float,
                        degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles to quaternion (w, x, y, z format).

    Args:
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
        degrees: If True, angles are in degrees

    Returns:
        Quaternion as numpy array [w, x, y, z]
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrix = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ], dtype=np.float32)

    return matrix


def generate_spherical_orientations(n: int) -> np.ndarray:
    """
    Generate evenly distributed orientations on a sphere using Fibonacci lattice.

    Args:
        n: Number of orientations to generate

    Returns:
        Array of quaternions [n, 4] in (w, x, y, z) format
    """
    orientations = []

    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)

        # Convert spherical to direction vector
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Create rotation that aligns Z-axis with this direction
        # Using rodrigues formula
        z_axis = np.array([0, 0, 1])
        direction = np.array([x, y, z])

        if np.allclose(direction, z_axis):
            quat = np.array([1, 0, 0, 0])
        elif np.allclose(direction, -z_axis):
            quat = np.array([0, 1, 0, 0])  # 180 degree rotation around X
        else:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

            # Axis-angle to quaternion
            w = np.cos(angle / 2)
            xyz = axis * np.sin(angle / 2)
            quat = np.array([w, xyz[0], xyz[1], xyz[2]])

        orientations.append(quat)

    return np.array(orientations, dtype=np.float32)


def generate_random_orientations(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random orientations uniformly distributed on SO(3).

    Args:
        n: Number of orientations
        seed: Random seed

    Returns:
        Array of quaternions [n, 4] in (w, x, y, z) format
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random quaternions using method from Shoemake
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 2 * np.pi, n)
    u3 = np.random.uniform(0, 2 * np.pi, n)

    w = np.sqrt(1 - u1) * np.cos(u2)
    x = np.sqrt(1 - u1) * np.sin(u2)
    y = np.sqrt(u1) * np.cos(u3)
    z = np.sqrt(u1) * np.sin(u3)

    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def batched_iterator(data: np.ndarray, batch_size: int):
    """
    Iterate over data in batches.

    Args:
        data: Array to iterate over
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    n = len(data)
    for i in range(0, n, batch_size):
        yield data[i:min(i + batch_size, n)]


def compute_jacobian_numerical(
    forward_kinematics_fn,
    joint_positions: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute Jacobian matrix numerically using finite differences.

    Args:
        forward_kinematics_fn: Function that takes joint positions and returns [x, y, z, qw, qx, qy, qz]
        joint_positions: Current joint positions [n_joints]
        epsilon: Finite difference step size

    Returns:
        Jacobian matrix [6, n_joints]
    """
    n_joints = len(joint_positions)
    jacobian = np.zeros((6, n_joints))

    # Get base pose
    base_pose = forward_kinematics_fn(joint_positions)
    base_pos = base_pose[:3]
    base_quat = base_pose[3:7]

    for i in range(n_joints):
        # Perturb joint
        perturbed = joint_positions.copy()
        perturbed[i] += epsilon

        perturbed_pose = forward_kinematics_fn(perturbed)
        perturbed_pos = perturbed_pose[:3]
        perturbed_quat = perturbed_pose[3:7]

        # Position Jacobian (linear velocity)
        jacobian[:3, i] = (perturbed_pos - base_pos) / epsilon

        # Orientation Jacobian (angular velocity)
        # Using quaternion difference to approximate angular velocity
        dq = quaternion_multiply(perturbed_quat, quaternion_inverse(base_quat))
        # For small rotations, angular velocity ≈ 2 * imaginary part of quaternion / dt
        jacobian[3:6, i] = 2 * dq[1:4] / epsilon

    return jacobian


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def create_pose_tensor(
    positions: np.ndarray,
    orientations: np.ndarray,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    Create pose tensor for cuRobo.

    Args:
        positions: Position array [n, 3]
        orientations: Quaternion array [n, 4] in (w, x, y, z) format

    Returns:
        Pose tensor [n, 7] with (x, y, z, qw, qx, qy, qz)
    """
    poses = np.concatenate([positions, orientations], axis=-1)
    return torch.tensor(poses, dtype=torch.float32, device=device)


def colormap_dexterity(dexterity: np.ndarray, vmin: float = 0, vmax: float = None) -> np.ndarray:
    """
    Map dexterity values to RGB colors.

    Low dexterity -> Blue
    Medium dexterity -> Green/Yellow
    High dexterity -> Red

    Args:
        dexterity: Dexterity values
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization

    Returns:
        RGB colors [n, 3]
    """
    if vmax is None:
        vmax = np.max(dexterity) if len(dexterity) > 0 else 1

    # Normalize to [0, 1]
    if vmax > vmin:
        normalized = np.clip((dexterity - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(dexterity)

    # Create colormap (blue -> cyan -> green -> yellow -> red)
    colors = np.zeros((len(dexterity), 3))

    for i, val in enumerate(normalized):
        if val < 0.25:
            t = val / 0.25
            colors[i] = [0, t, 1]  # Blue to Cyan
        elif val < 0.5:
            t = (val - 0.25) / 0.25
            colors[i] = [0, 1, 1 - t]  # Cyan to Green
        elif val < 0.75:
            t = (val - 0.5) / 0.25
            colors[i] = [t, 1, 0]  # Green to Yellow
        else:
            t = (val - 0.75) / 0.25
            colors[i] = [1, 1 - t, 0]  # Yellow to Red

    return colors


def size_from_manipulability(
    manipulability: np.ndarray,
    min_size: float = 2.0,
    max_size: float = 20.0
) -> np.ndarray:
    """
    Map manipulability values to point sizes.

    Args:
        manipulability: Manipulability values
        min_size: Minimum point size
        max_size: Maximum point size

    Returns:
        Point sizes
    """
    if len(manipulability) == 0:
        return np.array([])

    m_min = np.min(manipulability)
    m_max = np.max(manipulability)

    if m_max > m_min:
        normalized = (manipulability - m_min) / (m_max - m_min)
    else:
        normalized = np.ones_like(manipulability) * 0.5

    return min_size + normalized * (max_size - min_size)


class ProgressBar:
    """Simple progress bar for console output."""

    def __init__(self, total: int, prefix: str = "", width: int = 50):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._display()

    def _display(self):
        """Display the progress bar."""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{self.prefix} |{bar}| {percent*100:.1f}% ({self.current}/{self.total})', end='', flush=True)

        if self.current >= self.total:
            print()  # New line when complete

    def close(self):
        """Close the progress bar."""
        if self.current < self.total:
            print()
