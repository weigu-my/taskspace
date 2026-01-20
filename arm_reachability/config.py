"""
可达性分析配置类

定义了体素网格、IK求解器、姿态采样等配置参数。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class OrientationMode(Enum):
    """姿态采样模式"""
    FIXED = "fixed"                # 单一固定姿态
    MULTI_FIXED = "multi_fixed"    # 多个预定义姿态
    SPHERICAL = "spherical"        # 球面均匀采样
    RANDOM = "random"              # 随机姿态


@dataclass
class VoxelConfig:
    """体素网格配置 - 用于工作空间采样"""

    # 工作空间边界 [最小值, 最大值]，单位：米
    x_range: Tuple[float, float] = (-1.0, 1.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    z_range: Tuple[float, float] = (0.0, 1.5)

    # 体素分辨率（米）
    resolution: float = 0.05

    # 是否自动检测工作空间（通过FK采样）
    auto_detect: bool = True

    # FK采样参数（用于自动检测）
    fk_samples: int = 10000
    padding: float = 0.1  # 边界填充比例

    def get_grid_points(self) -> np.ndarray:
        """生成体素网格中心点"""
        x = np.arange(self.x_range[0], self.x_range[1] + self.resolution, self.resolution)
        y = np.arange(self.y_range[0], self.y_range[1] + self.resolution, self.resolution)
        z = np.arange(self.z_range[0], self.z_range[1] + self.resolution, self.resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

        return points.astype(np.float32)

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """获取体素网格形状"""
        nx = int(np.ceil((self.x_range[1] - self.x_range[0]) / self.resolution)) + 1
        ny = int(np.ceil((self.y_range[1] - self.y_range[0]) / self.resolution)) + 1
        nz = int(np.ceil((self.z_range[1] - self.z_range[0]) / self.resolution)) + 1
        return (nx, ny, nz)


@dataclass
class IKConfig:
    """IK求解器配置 - 用于多种子IK求解"""

    # IK种子数量（用于寻找多个解）
    num_seeds: int = 32

    # IK求解精度阈值
    position_threshold: float = 0.005  # 位置误差阈值（米）
    rotation_threshold: float = 0.05   # 旋转误差阈值（弧度）

    # cuRobo IK求解器设置
    num_graph_seeds: int = 12
    num_trajopt_seeds: int = 8

    # GPU批处理大小
    batch_size: int = 1024

    # 最大迭代次数
    max_iterations: int = 100

    # 是否使用并行环境批量求解
    use_parallel_env: bool = True

    # 碰撞检测
    collision_check: bool = True
    self_collision_check: bool = True


@dataclass
class OrientationConfig:
    """姿态采样配置"""

    mode: OrientationMode = OrientationMode.MULTI_FIXED

    # FIXED模式：单个四元数 [w, x, y, z]
    fixed_orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    # MULTI_FIXED模式：多个欧拉角 [roll, pitch, yaw]，单位：度
    multi_orientations_euler: List[List[float]] = field(default_factory=lambda: [
        [0, 0, 0],       # Z轴向下
        [180, 0, 0],     # Z轴向上
        [90, 0, 0],      # X轴向前
        [-90, 0, 0],     # X轴向后
        [0, 90, 0],      # Y轴向左
        [0, -90, 0],     # Y轴向右
    ])

    # SPHERICAL模式：球面采样数量
    num_spherical_samples: int = 26  # 二十面体顶点+面心

    # RANDOM模式：随机姿态数量
    num_random_samples: int = 10


@dataclass
class ReachabilityConfig:
    """可达性分析主配置"""

    # URDF文件路径
    urdf_path: str = ""

    # 末端执行器链接名（为空则自动检测）
    ee_link: str = ""

    # 基座链接名（为空则自动检测）
    base_link: str = ""

    # 体素配置
    voxel: VoxelConfig = field(default_factory=VoxelConfig)

    # IK配置
    ik: IKConfig = field(default_factory=IKConfig)

    # 姿态配置
    orientation: OrientationConfig = field(default_factory=OrientationConfig)

    # 输出目录
    output_dir: str = "./reachability_output"

    # 计算设备
    device: str = "cuda:0"

    # 是否计算可操作度
    compute_manipulability: bool = True

    # 详细输出
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReachabilityConfig":
        """从YAML文件加载配置"""
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

        # 体素配置
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

        # IK配置
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

        # 姿态配置
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
        """保存配置到YAML文件"""
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
