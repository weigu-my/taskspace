"""
可达性分析配置类

定义了体素网格、IK求解器、姿态采样等配置参数。
支持单臂和多臂配置。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class OrientationMode(Enum):
    """姿态采样模式"""
    FIXED = "fixed"                # 单一固定姿态
    MULTI_FIXED = "multi_fixed"    # 多个预定义姿态
    SPHERICAL = "spherical"        # 球面均匀采样
    RANDOM = "random"              # 随机姿态


class ArmMode(Enum):
    """机械臂分析模式"""
    AUTO = "auto"          # 自动检测（单臂）
    LEFT = "left"          # 仅左臂
    RIGHT = "right"        # 仅右臂
    BOTH = "both"          # 双臂同时分析
    ALL = "all"            # 所有检测到的臂
    SINGLE = "single"      # 指定单臂（需配合 arm_name 使用）


@dataclass
class VoxelConfig:
    """体素网格配置 - 用于工作空间采样"""

    # 工作空间边界 [最小值, 最大值]，单位：米
    # 默认范围较大，适合大多数机器人；auto_detect=True 时会自动调整
    x_range: Tuple[float, float] = (-2.5, 2.5)
    y_range: Tuple[float, float] = (-2.5, 2.5)
    z_range: Tuple[float, float] = (-0.5, 3.0)

    # 体素分辨率（米）
    resolution: float = 0.05

    # 是否自动检测工作空间（通过FK采样）
    auto_detect: bool = True

    # FK采样参数（用于自动检测）
    fk_samples: int = 10000
    padding: float = 0.5  # 边界填充比例（50% 确保覆盖完整可达空间）

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

    def copy(self) -> "VoxelConfig":
        """创建配置的副本"""
        return VoxelConfig(
            x_range=self.x_range,
            y_range=self.y_range,
            z_range=self.z_range,
            resolution=self.resolution,
            auto_detect=self.auto_detect,
            fk_samples=self.fk_samples,
            padding=self.padding
        )


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

    def copy(self) -> "IKConfig":
        """创建配置的副本"""
        return IKConfig(
            num_seeds=self.num_seeds,
            position_threshold=self.position_threshold,
            rotation_threshold=self.rotation_threshold,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            batch_size=self.batch_size,
            max_iterations=self.max_iterations,
            use_parallel_env=self.use_parallel_env,
            collision_check=self.collision_check,
            self_collision_check=self.self_collision_check
        )


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

    def copy(self) -> "OrientationConfig":
        """创建配置的副本"""
        return OrientationConfig(
            mode=self.mode,
            fixed_orientation=self.fixed_orientation.copy(),
            multi_orientations_euler=[o.copy() for o in self.multi_orientations_euler],
            num_spherical_samples=self.num_spherical_samples,
            num_random_samples=self.num_random_samples
        )


@dataclass
class ArmConfig:
    """单臂配置"""
    name: str = ""              # 臂名称（如 'left', 'right'）
    base_link: str = ""         # 臂基座链接
    ee_link: str = ""           # 末端执行器链接
    color: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # 可视化颜色 (R, G, B)
    output_prefix: str = ""     # 输出文件前缀


@dataclass
class MultiArmConfig:
    """多臂分析配置"""

    # 分析模式
    mode: ArmMode = ArmMode.AUTO

    # 指定的臂名称（当 mode=SINGLE 时使用）
    arm_name: str = ""

    # 各臂的自定义配置（可选）
    arm_configs: Dict[str, ArmConfig] = field(default_factory=dict)

    # 左臂默认颜色
    left_color: Tuple[float, float, float] = (0.0, 0.8, 0.2)   # 绿色

    # 右臂默认颜色
    right_color: Tuple[float, float, float] = (0.8, 0.2, 0.0)  # 红色

    # 其他臂的默认颜色
    default_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.0, 0.5, 1.0),   # 蓝色
        (1.0, 0.5, 0.0),   # 橙色
        (0.5, 0.0, 1.0),   # 紫色
        (1.0, 1.0, 0.0),   # 黄色
    ])

    def get_arm_color(self, arm_name: str, index: int = 0) -> Tuple[float, float, float]:
        """获取指定臂的颜色"""
        # 检查自定义配置
        if arm_name in self.arm_configs:
            return self.arm_configs[arm_name].color

        # 使用默认颜色
        if arm_name.lower() == 'left':
            return self.left_color
        elif arm_name.lower() == 'right':
            return self.right_color
        else:
            return self.default_colors[index % len(self.default_colors)]


@dataclass
class ReachabilityConfig:
    """可达性分析主配置"""

    # URDF文件路径
    urdf_path: str = ""

    # 末端执行器链接名（为空则自动检测）
    ee_link: str = ""

    # 基座链接名（为空则自动检测）
    base_link: str = ""

    # 结果点云坐标系: "base" 或 "world"
    points_frame: str = "base"

    # 体素配置
    voxel: VoxelConfig = field(default_factory=VoxelConfig)

    # IK配置
    ik: IKConfig = field(default_factory=IKConfig)

    # 姿态配置
    orientation: OrientationConfig = field(default_factory=OrientationConfig)

    # 多臂配置
    multi_arm: MultiArmConfig = field(default_factory=MultiArmConfig)

    # 输出目录
    output_dir: str = "./reachability_output"

    # 计算设备
    device: str = "cuda:0"

    # 是否计算可操作度
    compute_manipulability: bool = True

    # 详细输出
    verbose: bool = True

    def copy(self) -> "ReachabilityConfig":
        """创建配置的副本"""
        return ReachabilityConfig(
            urdf_path=self.urdf_path,
            ee_link=self.ee_link,
            base_link=self.base_link,
            points_frame=self.points_frame,
            voxel=self.voxel.copy(),
            ik=self.ik.copy(),
            orientation=self.orientation.copy(),
            multi_arm=self.multi_arm,
            output_dir=self.output_dir,
            device=self.device,
            compute_manipulability=self.compute_manipulability,
            verbose=self.verbose
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReachabilityConfig":
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
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
        if 'points_frame' in data:
            config.points_frame = data['points_frame']
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

        # 多臂配置
        if 'multi_arm' in data:
            ma = data['multi_arm']
            if 'mode' in ma:
                config.multi_arm.mode = ArmMode(ma['mode'])
            if 'arm_name' in ma:
                config.multi_arm.arm_name = ma['arm_name']
            if 'left_color' in ma:
                config.multi_arm.left_color = tuple(ma['left_color'])
            if 'right_color' in ma:
                config.multi_arm.right_color = tuple(ma['right_color'])

        return config

    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        import yaml

        data = {
            'urdf_path': self.urdf_path,
            'ee_link': self.ee_link,
            'base_link': self.base_link,
            'points_frame': self.points_frame,
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
            },
            'multi_arm': {
                'mode': self.multi_arm.mode.value,
                'arm_name': self.multi_arm.arm_name,
                'left_color': list(self.multi_arm.left_color),
                'right_color': list(self.multi_arm.right_color),
            }
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
