"""
可达性分析工具函数
"""

import numpy as np
from typing import List, Tuple, Optional
import torch


def euler_to_quaternion(roll: float, pitch: float, yaw: float,
                        degrees: bool = True) -> np.ndarray:
    """
    将欧拉角转换为四元数 (w, x, y, z 格式)

    参数:
        roll: 绕 X 轴旋转
        pitch: 绕 Y 轴旋转
        yaw: 绕 Z 轴旋转
        degrees: 如果为 True，角度单位为度

    返回:
        四元数数组 [w, x, y, z]
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
    将四元数转换为 3x3 旋转矩阵

    参数:
        quat: 四元数 [w, x, y, z]

    返回:
        3x3 旋转矩阵
    """
    w, x, y, z = quat

    # 归一化四元数
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
    使用斐波那契格点生成球面上均匀分布的姿态

    参数:
        n: 要生成的姿态数量

    返回:
        四元数数组 [n, 4]，格式为 (w, x, y, z)
    """
    orientations = []

    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)

        # 将球面坐标转换为方向向量
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # 创建将 Z 轴对齐到该方向的旋转
        # 使用罗德里格斯公式
        z_axis = np.array([0, 0, 1])
        direction = np.array([x, y, z])

        if np.allclose(direction, z_axis):
            quat = np.array([1, 0, 0, 0])
        elif np.allclose(direction, -z_axis):
            quat = np.array([0, 1, 0, 0])  # 绕 X 轴旋转 180 度
        else:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

            # 轴角转四元数
            w = np.cos(angle / 2)
            xyz = axis * np.sin(angle / 2)
            quat = np.array([w, xyz[0], xyz[1], xyz[2]])

        orientations.append(quat)

    return np.array(orientations, dtype=np.float32)


def generate_random_orientations(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    在 SO(3) 上生成均匀分布的随机姿态

    参数:
        n: 姿态数量
        seed: 随机种子

    返回:
        四元数数组 [n, 4]，格式为 (w, x, y, z)
    """
    if seed is not None:
        np.random.seed(seed)

    # 使用 Shoemake 方法生成随机四元数
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
    按批次迭代数据

    参数:
        data: 要迭代的数组
        batch_size: 每个批次的大小

    生成:
        数据批次
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
    使用有限差分数值计算雅可比矩阵

    参数:
        forward_kinematics_fn: 接受关节位置并返回 [x, y, z, qw, qx, qy, qz] 的函数
        joint_positions: 当前关节位置 [n_joints]
        epsilon: 有限差分步长

    返回:
        雅可比矩阵 [6, n_joints]
    """
    n_joints = len(joint_positions)
    jacobian = np.zeros((6, n_joints))

    # 获取基准位姿
    base_pose = forward_kinematics_fn(joint_positions)
    base_pos = base_pose[:3]
    base_quat = base_pose[3:7]

    for i in range(n_joints):
        # 扰动关节
        perturbed = joint_positions.copy()
        perturbed[i] += epsilon

        perturbed_pose = forward_kinematics_fn(perturbed)
        perturbed_pos = perturbed_pose[:3]
        perturbed_quat = perturbed_pose[3:7]

        # 位置雅可比（线速度）
        jacobian[:3, i] = (perturbed_pos - base_pos) / epsilon

        # 姿态雅可比（角速度）
        # 使用四元数差分近似角速度
        dq = quaternion_multiply(perturbed_quat, quaternion_inverse(base_quat))
        # 对于小旋转，角速度 ≈ 2 * 四元数虚部 / dt
        jacobian[3:6, i] = 2 * dq[1:4] / epsilon

    return jacobian


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """计算四元数的逆（对于单位四元数即共轭）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def create_pose_tensor(
    positions: np.ndarray,
    orientations: np.ndarray,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    为 cuRobo 创建位姿张量

    参数:
        positions: 位置数组 [n, 3]
        orientations: 四元数数组 [n, 4]，格式为 (w, x, y, z)

    返回:
        位姿张量 [n, 7]，格式为 (x, y, z, qw, qx, qy, qz)
    """
    poses = np.concatenate([positions, orientations], axis=-1)
    return torch.tensor(poses, dtype=torch.float32, device=device)


def colormap_dexterity(dexterity: np.ndarray, vmin: float = 0, vmax: float = None) -> np.ndarray:
    """
    将灵活度值映射为 RGB 颜色

    低灵活度 -> 蓝色
    中灵活度 -> 绿色/黄色
    高灵活度 -> 红色

    参数:
        dexterity: 灵活度值
        vmin: 归一化最小值
        vmax: 归一化最大值

    返回:
        RGB 颜色 [n, 3]
    """
    if vmax is None:
        vmax = np.max(dexterity) if len(dexterity) > 0 else 1

    # 归一化到 [0, 1]
    if vmax > vmin:
        normalized = np.clip((dexterity - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(dexterity)

    # 创建色图（蓝 -> 青 -> 绿 -> 黄 -> 红）
    colors = np.zeros((len(dexterity), 3))

    for i, val in enumerate(normalized):
        if val < 0.25:
            t = val / 0.25
            colors[i] = [0, t, 1]  # 蓝色到青色
        elif val < 0.5:
            t = (val - 0.25) / 0.25
            colors[i] = [0, 1, 1 - t]  # 青色到绿色
        elif val < 0.75:
            t = (val - 0.5) / 0.25
            colors[i] = [t, 1, 0]  # 绿色到黄色
        else:
            t = (val - 0.75) / 0.25
            colors[i] = [1, 1 - t, 0]  # 黄色到红色

    return colors


def size_from_manipulability(
    manipulability: np.ndarray,
    min_size: float = 2.0,
    max_size: float = 20.0
) -> np.ndarray:
    """
    将可操作度值映射为点大小

    参数:
        manipulability: 可操作度值
        min_size: 最小点大小
        max_size: 最大点大小

    返回:
        点大小
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
    """简单的控制台进度条"""

    def __init__(self, total: int, prefix: str = "", width: int = 50):
        """
        初始化进度条

        参数:
            total: 总步数
            prefix: 前缀文本
            width: 进度条宽度
        """
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0

    def update(self, n: int = 1):
        """更新进度（增加 n 步）"""
        self.current += n
        self._display()

    def _display(self):
        """显示进度条"""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{self.prefix} |{bar}| {percent*100:.1f}% ({self.current}/{self.total})', end='', flush=True)

        if self.current >= self.total:
            print()  # 完成时换行

    def close(self):
        """关闭进度条"""
        if self.current < self.total:
            print()
