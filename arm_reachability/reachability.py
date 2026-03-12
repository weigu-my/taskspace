"""
基于体素的可达性分析器

本模块使用体素网格采样和GPU加速IK求解进行全面的可达性分析。
支持单臂和多臂同时分析。
"""

import os
import json
import time
import numpy as np
import torch
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .config import (
    ReachabilityConfig,
    VoxelConfig,
    IKConfig,
    OrientationConfig,
    OrientationMode,
    ArmMode,
    MultiArmConfig
)
from .urdf_parser import URDFParser, RobotConfig, ArmChain
from .ik_solver import MultiSeedIKSolver, BatchIKResult
from .manipulability import ManipulabilityCalculator
from .utils import (
    euler_to_quaternion,
    generate_spherical_orientations,
    generate_random_orientations,
    ProgressBar
)


def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class ReachabilityResult:
    """可达性分析的完整结果"""
    # 网格信息
    grid_points: np.ndarray           # [n_points, 3] 所有体素中心点
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz) 体素网格维度

    # 基本可达性
    reachable_mask: np.ndarray        # [n_points] 可达点的布尔掩码
    reachable_points: np.ndarray      # [n_reachable, 3] 可达位置

    # 多种子IK结果
    num_solutions: np.ndarray         # [n_points, n_orientations] 每个位姿的解数量

    # 灵活度（每个位置可达的姿态数量）
    dexterity: np.ndarray             # [n_points] 可达姿态数量

    # 可操作度（Yoshikawa指标）
    manipulability: np.ndarray        # [n_points] 可操作度值

    # 最优IK解
    best_solutions: np.ndarray        # [n_points, n_joints] 最优关节配置

    # 统计信息
    total_points: int
    reachable_count: int
    reachability_ratio: float
    max_dexterity: int
    mean_dexterity: float
    max_manipulability: float
    mean_manipulability: float
    computation_time: float

    # 臂信息（用于多臂分析）
    arm_name: str = ""
    base_link: str = ""
    ee_link: str = ""
    color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    base_transform: Optional[Dict[str, Any]] = None
    points_frame: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        data = {
            'arm_name': self.arm_name,
            'base_link': self.base_link,
            'ee_link': self.ee_link,
            'points_frame': self.points_frame,
            'total_points': self.total_points,
            'reachable_count': self.reachable_count,
            'reachability_ratio': self.reachability_ratio,
            'max_dexterity': int(self.max_dexterity),
            'mean_dexterity': float(self.mean_dexterity),
            'max_manipulability': float(self.max_manipulability),
            'mean_manipulability': float(self.mean_manipulability),
            'computation_time': self.computation_time,
            'grid_shape': list(self.grid_shape),
            'color': list(self.color),
        }
        if self.base_transform is not None:
            data['base_transform'] = {
                'translation': self.base_transform['translation'].tolist(),
                'rotation': self.base_transform['rotation'].tolist()
            }
        return data


@dataclass
class MultiArmReachabilityResult:
    """多臂可达性分析结果"""
    results: Dict[str, ReachabilityResult]  # {臂名称: 结果}
    urdf_path: str
    total_computation_time: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            'urdf_path': self.urdf_path,
            'total_computation_time': self.total_computation_time,
            'arms': {name: result.to_dict() for name, result in self.results.items()}
        }

    @property
    def arm_names(self) -> List[str]:
        """获取所有臂名称"""
        return list(self.results.keys())

    def get_result(self, arm_name: str) -> Optional[ReachabilityResult]:
        """获取指定臂的结果"""
        return self.results.get(arm_name)


class ReachabilityAnalyzer:
    """
    可达性分析主类

    该分析器:
    1. 解析URDF并创建cuRobo配置
    2. 生成体素网格用于工作空间采样
    3. 使用GPU加速的多种子IK测试可达性
    4. 计算灵活度（每个位置可达的姿态数量）
    5. 计算可操作度（Yoshikawa指标）
    """

    def __init__(
        self,
        config: ReachabilityConfig = None,
        robot_config: RobotConfig = None,
        world_config: Optional[Any] = None,
        urdf_path: str = None,
        other_arm_links: Optional[set] = None,
        dynamic_mode: bool = False
    ):
        """
        初始化可达性分析器

        参数:
            config: 可达性分析配置
            robot_config: 可选的预解析机器人配置（用于多臂分析）
            world_config: cuRobo WorldConfig对象，用于环境碰撞检测
                         可以包含机器人躯体、另一臂等障碍物
            urdf_path: URDF文件路径（如果config未提供）
            other_arm_links: 对侧臂的 link 名称集合（动态模式使用）
            dynamic_mode: 是否启用动态可达性模式
        """
        # 支持仅传入urdf_path的简化用法
        if config is None:
            from .config import ReachabilityConfig
            config = ReachabilityConfig()
            if urdf_path:
                config.urdf_path = urdf_path

        self.config = config
        self.robot_config: Optional[RobotConfig] = robot_config
        self.world_config = world_config  # 环境碰撞配置
        self._other_arm_links = other_arm_links
        self._dynamic_mode = dynamic_mode
        self.ik_solver: Optional[MultiSeedIKSolver] = None
        self.manipulability_calc: Optional[ManipulabilityCalculator] = None

        self._initialize()

    def _initialize(self):
        """初始化所有组件"""
        print("=" * 60)
        print("初始化可达性分析器")
        print("=" * 60)

        # 解析URDF（如果没有提供预解析的配置）
        if self.robot_config is None:
            print(f"\n[1/4] 解析URDF: {self.config.urdf_path}")
            parser = URDFParser(self.config.urdf_path)
            self.robot_config = parser.parse(
                base_link=self.config.base_link,
                ee_link=self.config.ee_link
            )
        else:
            print(f"\n[1/4] 使用预解析的机器人配置")

        print(f"  机器人: {self.robot_config.name}")
        print(f"  基座链接: {self.robot_config.base_link}")
        print(f"  末端链接: {self.robot_config.ee_link}")
        print(f"  自由度: {self.robot_config.num_dof}")
        print(f"  活动关节: {[j.name for j in self.robot_config.active_joints]}")

        # 初始化IK求解器
        print(f"\n[2/4] 初始化IK求解器...")
        self.ik_solver = MultiSeedIKSolver(
            robot_config=self.robot_config,
            ik_config=self.config.ik,
            device=self.config.device,
            world_config=self.world_config,
            other_arm_links=self._other_arm_links,
            dynamic_mode=self._dynamic_mode
        )

        # 初始化可操作度计算器
        if self.config.compute_manipulability:
            print(f"\n[3/4] 初始化可操作度计算器...")
            self.manipulability_calc = ManipulabilityCalculator(
                robot_config=self.robot_config,
                device=self.config.device
            )
        else:
            print(f"\n[3/4] 跳过可操作度（已禁用）")

        # 自动检测工作空间边界
        if self.config.voxel.auto_detect:
            print(f"\n[4/4] 自动检测工作空间边界...")
            # 根据 points_frame 选择坐标系
            if self.config.points_frame == "world":
                print(f"  使用 world 坐标系")
                x_range, y_range, z_range = self.ik_solver.estimate_workspace_bounds_world(
                    n_samples=self.config.voxel.fk_samples,
                    padding=self.config.voxel.padding
                )
            else:
                print(f"  使用 base_link 坐标系")
                x_range, y_range, z_range = self.ik_solver.estimate_workspace_bounds(
                    n_samples=self.config.voxel.fk_samples,
                    padding=self.config.voxel.padding
                )
            self.config.voxel.x_range = x_range
            self.config.voxel.y_range = y_range
            self.config.voxel.z_range = z_range
        else:
            print(f"\n[4/4] 使用指定的工作空间边界")

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("初始化完成!")
        print("=" * 60)

    def _generate_orientations(self) -> np.ndarray:
        """根据配置生成姿态样本"""
        mode = self.config.orientation.mode

        if mode == OrientationMode.FIXED:
            # 单一固定姿态
            return np.array([self.config.orientation.fixed_orientation], dtype=np.float32)

        elif mode == OrientationMode.MULTI_FIXED:
            # 多个预定义姿态（从欧拉角转换）
            orientations = []
            for euler in self.config.orientation.multi_orientations_euler:
                quat = euler_to_quaternion(euler[0], euler[1], euler[2], degrees=True)
                orientations.append(quat)
            return np.array(orientations, dtype=np.float32)

        elif mode == OrientationMode.SPHERICAL:
            # 球面均匀分布
            return generate_spherical_orientations(
                self.config.orientation.num_spherical_samples
            )

        elif mode == OrientationMode.RANDOM:
            # 随机姿态
            return generate_random_orientations(
                self.config.orientation.num_random_samples
            )

        else:
            raise ValueError(f"未知的姿态模式: {mode}")

    def analyze(self, arm_name: str = "", color: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> ReachabilityResult:
        """
        执行完整的可达性分析

        参数:
            arm_name: 臂名称（用于结果标识）
            color: 可视化颜色

        返回:
            包含所有计算指标的ReachabilityResult
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        if arm_name:
            print(f"开始可达性分析 - {arm_name}")
        else:
            print("开始可达性分析")
        print("=" * 60)

        # 生成体素网格
        # 当 points_frame="world" 时，voxel 范围已在 world 坐标系下设置
        # 当 points_frame="base" 时，voxel 范围在 base_link 坐标系下
        print("\n[Step 1] 生成体素网格...")
        grid_points = self.config.voxel.get_grid_points()
        grid_shape = self.config.voxel.get_grid_shape()
        n_points = len(grid_points)

        points_frame = self.config.points_frame
        grid_points_for_ik = grid_points

        print(f"  网格形状: {grid_shape}")
        print(f"  分辨率: {self.config.voxel.resolution}m")
        print(f"  总体素数: {n_points}")
        print(f"  点云坐标系: {points_frame}")

        # 生成姿态
        print("\n[Step 2] 生成姿态...")
        orientations = self._generate_orientations()
        n_orientations = len(orientations)
        print(f"  模式: {self.config.orientation.mode.value}")
        print(f"  姿态数量: {n_orientations}")

        # 执行IK求解
        # 当 points_frame="world" 时，grid_points 已经是 world 坐标系
        print("\n[Step 3] 求解所有位置和姿态的IK...")
        reachable_mask, dexterity, num_solutions = self.ik_solver.solve_batch_multi_orientation(
            positions=grid_points_for_ik,
            orientations=orientations,
            batch_size=self.config.ik.batch_size,
            positions_in_world=(points_frame == "world")
        )

        reachable_points = grid_points[reachable_mask]
        reachable_count = int(np.sum(reachable_mask))

        print(f"  可达点: {reachable_count} / {n_points} ({reachable_count/n_points*100:.1f}%)")
        print(f"  最大灵活度: {np.max(dexterity)} / {n_orientations}")

        # 清理显存
        clear_gpu_memory()

        # 获取最优解用于可操作度计算
        print("\n[Step 4] 计算最优IK解...")
        best_solutions = self._compute_best_solutions(
            grid_points_for_ik, orientations, reachable_mask,
            batch_size=self.config.ik.batch_size
        )

        # 清理显存
        clear_gpu_memory()

        # 计算可操作度
        manipulability = np.zeros(n_points, dtype=np.float32)
        if self.config.compute_manipulability and self.manipulability_calc is not None:
            print("\n[Step 5] 计算可操作度...")
            manipulability = self.manipulability_calc.compute_for_best_solutions(
                best_solutions=best_solutions,
                reachable_mask=reachable_mask
            )
            print(f"  最大可操作度: {np.max(manipulability):.4f}")
            print(f"  平均可操作度（可达点）: {np.mean(manipulability[reachable_mask]):.4f}")
        else:
            print("\n[Step 5] 跳过可操作度计算")

        computation_time = time.time() - start_time

        # 计算统计信息
        reachable_dexterity = dexterity[reachable_mask]
        reachable_manip = manipulability[reachable_mask]

        # 当 points_frame="world" 时，grid_points 已经在 world 坐标系下生成
        # 无需额外转换
        if points_frame == "world":
            print(f"  点云已在 world 坐标系下生成")

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
            reachability_ratio=reachable_count / n_points if n_points > 0 else 0,
            max_dexterity=int(np.max(dexterity)) if reachable_count > 0 else 0,
            mean_dexterity=float(np.mean(reachable_dexterity)) if reachable_count > 0 else 0,
            max_manipulability=float(np.max(reachable_manip)) if reachable_count > 0 else 0,
            mean_manipulability=float(np.mean(reachable_manip)) if reachable_count > 0 else 0,
            computation_time=computation_time,
            arm_name=arm_name,
            base_link=self.robot_config.base_link,
            ee_link=self.robot_config.ee_link,
            color=color,
            base_transform=self.ik_solver.get_base_transform() if self.ik_solver else None,
            points_frame=points_frame
        )

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        print(f"  总时间: {computation_time:.2f}秒")
        print(f"  可达性: {result.reachability_ratio*100:.1f}%")
        print(f"  最大灵活度: {result.max_dexterity}")
        print(f"  平均灵活度: {result.mean_dexterity:.2f}")

        return result

    def _compute_best_solutions(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        reachable_mask: np.ndarray,
        batch_size: int = 512
    ) -> np.ndarray:
        """
        为每个可达位置计算最优IK解（分批处理）

        参数:
            positions: 所有位置 [n_positions, 3]
            orientations: 姿态数组 [n_orientations, 4]
            reachable_mask: 可达性掩码 [n_positions]
            batch_size: 批处理大小

        返回:
            最优关节配置 [n_positions, n_joints]
        """
        n_positions = len(positions)

        # 获取可达位置的索引
        reachable_indices = np.where(reachable_mask)[0]
        n_reachable = len(reachable_indices)

        # 使用第一个姿态
        default_orientation = orientations[0]

        # 先用一个小批次探测实际的关节数
        # cuRobo内部可能只使用运动链中的部分关节
        if n_reachable > 0:
            test_positions = positions[reachable_indices[:1]]
            test_orientations = np.tile(default_orientation, (1, 1))
            test_result = self.ik_solver.solve_batch(
                positions=test_positions,
                orientations=test_orientations,
                return_all_solutions=False
            )
            n_joints = test_result.best_solutions.shape[1]
            print(f"  IK求解器实际关节数: {n_joints}")
        else:
            n_joints = self.ik_solver.n_joints

        best_solutions = np.zeros((n_positions, n_joints), dtype=np.float32)

        if n_reachable == 0:
            return best_solutions

        # 分批处理可达位置
        print(f"  可达点数: {n_reachable}")
        print(f"  批处理大小: {batch_size}")

        num_batches = (n_reachable + batch_size - 1) // batch_size
        print(f"  批次数: {num_batches}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_reachable)

            # 获取当前批次的索引
            batch_indices = reachable_indices[start_idx:end_idx]
            batch_positions = positions[batch_indices]
            batch_orientations = np.tile(default_orientation, (len(batch_indices), 1))

            # 求解IK
            result = self.ik_solver.solve_batch(
                positions=batch_positions,
                orientations=batch_orientations,
                return_all_solutions=False
            )

            # 保存结果
            best_solutions[batch_indices] = result.best_solutions

            # 显示进度
            progress = (batch_idx + 1) / num_batches * 100
            print(f"\r  进度: {progress:.1f}%", end='', flush=True)

            # 定期清理显存
            if (batch_idx + 1) % 10 == 0:
                clear_gpu_memory()

        print()  # 换行

        return best_solutions

    def save_results(self, result: ReachabilityResult, prefix: str = "reachability"):
        """
        保存分析结果到文件

        参数:
            result: 要保存的ReachabilityResult
            prefix: 文件名前缀
        """
        output_dir = self.config.output_dir
        print(f"\n保存结果到 {output_dir}/")

        # 保存numpy数组
        np.save(os.path.join(output_dir, f"{prefix}_grid_points.npy"), result.grid_points)
        np.save(os.path.join(output_dir, f"{prefix}_reachable_mask.npy"), result.reachable_mask)
        np.save(os.path.join(output_dir, f"{prefix}_reachable_points.npy"), result.reachable_points)
        np.save(os.path.join(output_dir, f"{prefix}_dexterity.npy"), result.dexterity)
        np.save(os.path.join(output_dir, f"{prefix}_manipulability.npy"), result.manipulability)
        np.save(os.path.join(output_dir, f"{prefix}_best_solutions.npy"), result.best_solutions)

        # 保存可达点的CSV（含指标）
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

        # 保存统计信息JSON
        stats = result.to_dict()
        stats['robot_name'] = self.robot_config.name
        stats['urdf_path'] = self.robot_config.urdf_path
        stats['num_dof'] = self.robot_config.num_dof
        stats['voxel_resolution'] = self.config.voxel.resolution
        stats['num_ik_seeds'] = self.config.ik.num_seeds

        with open(os.path.join(output_dir, f"{prefix}_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        print(f"  已保存: {prefix}_grid_points.npy")
        print(f"  已保存: {prefix}_reachable_points.npy")
        print(f"  已保存: {prefix}_dexterity.npy")
        print(f"  已保存: {prefix}_manipulability.npy")
        print(f"  已保存: {prefix}_data.csv")
        print(f"  已保存: {prefix}_stats.json")

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
        从URDF路径创建分析器的便捷构造函数

        参数:
            urdf_path: URDF文件路径
            ee_link: 末端执行器链接名（为空则自动检测）
            base_link: 基座链接名（为空则自动检测）
            resolution: 体素分辨率（米）
            num_seeds: IK种子数量
            device: 计算设备
            output_dir: 输出目录

        返回:
            配置好的ReachabilityAnalyzer
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


class MultiArmReachabilityAnalyzer:
    """
    多臂可达性分析器

    支持同时分析多条机械臂的可达空间。
    """

    def __init__(self, config: ReachabilityConfig, dynamic_mode: bool = False):
        """
        初始化多臂分析器

        参数:
            config: 可达性分析配置
            dynamic_mode: 是否启用动态可达性模式（对侧臂碰撞实时过滤）
        """
        self.config = config
        self.dynamic_mode = dynamic_mode
        self.parser = URDFParser(config.urdf_path)
        self.arm_chains: List[ArmChain] = []
        self.results: Dict[str, ReachabilityResult] = {}

        self._detect_arms()

    def _detect_arms(self):
        """检测URDF中的所有机械臂"""
        print("=" * 60)
        print("检测机械臂")
        print("=" * 60)

        self.arm_chains = self.parser.detect_arm_chains()

        if not self.arm_chains:
            print("  [警告] 未检测到机械臂链，将尝试使用默认解析")
            return

        print(f"\n检测到 {len(self.arm_chains)} 条机械臂:")
        for chain in self.arm_chains:
            print(f"  - {chain.name}:")
            print(f"      基座: {chain.base_link}")
            print(f"      末端: {chain.ee_link}")
            print(f"      自由度: {chain.num_dof}")
            print(f"      关节: {[j.name for j in chain.active_joints]}")

    def get_arms_to_analyze(self) -> List[ArmChain]:
        """根据配置获取要分析的机械臂列表"""
        mode = self.config.multi_arm.mode

        if mode == ArmMode.AUTO:
            # 自动模式：返回第一条臂
            return self.arm_chains[:1] if self.arm_chains else []

        elif mode == ArmMode.LEFT:
            # 仅左臂
            return [c for c in self.arm_chains if c.name.lower() == 'left']

        elif mode == ArmMode.RIGHT:
            # 仅右臂
            return [c for c in self.arm_chains if c.name.lower() == 'right']

        elif mode == ArmMode.BOTH:
            # 双臂（左右）
            return [c for c in self.arm_chains if c.name.lower() in ['left', 'right']]

        elif mode == ArmMode.ALL:
            # 所有臂
            return self.arm_chains

        elif mode == ArmMode.SINGLE:
            # 指定的单臂
            arm_name = self.config.multi_arm.arm_name.lower()
            return [c for c in self.arm_chains if c.name.lower() == arm_name]

        return []

    def analyze(self) -> MultiArmReachabilityResult:
        """
        执行多臂可达性分析

        返回:
            MultiArmReachabilityResult 包含所有臂的分析结果
        """
        start_time = time.time()

        arms_to_analyze = self.get_arms_to_analyze()

        if not arms_to_analyze:
            print("[错误] 没有找到要分析的机械臂")
            if self.arm_chains:
                print(f"可用的臂: {[c.name for c in self.arm_chains]}")
            return MultiArmReachabilityResult(
                results={},
                urdf_path=self.config.urdf_path,
                total_computation_time=0
            )

        print("\n" + "=" * 60)
        print(f"开始多臂可达性分析")
        print(f"将分析 {len(arms_to_analyze)} 条臂: {[c.name for c in arms_to_analyze]}")
        print("=" * 60)

        results = {}

        # 预计算：收集所有臂的 chain links（用于动态模式下互斥）
        all_arm_chain_links = {}
        if self.dynamic_mode:
            for chain in arms_to_analyze:
                # 收集该臂的所有运动 link（chain + 子孙）
                chain_links = set(chain.chain_links)
                child_map = {}
                for j in self.parser._joints:
                    child_map.setdefault(j.parent_link, []).append(j.child_link)
                stack = list(chain_links)
                all_moving = set(chain_links)
                while stack:
                    current = stack.pop()
                    for child in child_map.get(current, []):
                        if child not in all_moving:
                            all_moving.add(child)
                            stack.append(child)
                all_arm_chain_links[chain.name] = all_moving
            print(f"[动态模式] 各臂运动 link: {{{', '.join(f'{k}: {len(v)}' for k, v in all_arm_chain_links.items())}}}")

        for idx, arm_chain in enumerate(arms_to_analyze):
            print(f"\n{'='*60}")
            print(f"分析臂 {idx+1}/{len(arms_to_analyze)}: {arm_chain.name}")
            print(f"{'='*60}")

            # 创建该臂的机器人配置
            robot_config = RobotConfig(
                name=f"{self.parser.get_robot_name()}_{arm_chain.name}",
                joints=self.parser._joints,
                links=self.parser._links,
                base_link=arm_chain.base_link,
                ee_link=arm_chain.ee_link,
                urdf_path=self.parser.urdf_path,
                urdf_dir=self.parser.urdf_dir,
                chain_joints=arm_chain.chain_joints,
                chain_links=arm_chain.chain_links
            )

            # 动态模式：收集对侧臂的 link 集合
            other_arm_links = None
            if self.dynamic_mode:
                other_arm_links = set()
                for other_name, other_links in all_arm_chain_links.items():
                    if other_name != arm_chain.name:
                        other_arm_links |= other_links

            # 创建该臂的配置副本
            arm_config = self.config.copy()
            arm_config.base_link = arm_chain.base_link
            arm_config.ee_link = arm_chain.ee_link
            arm_config.output_dir = os.path.join(self.config.output_dir, arm_chain.name)

            # 获取该臂的颜色
            color = self.config.multi_arm.get_arm_color(arm_chain.name, idx)

            # 创建分析器并执行分析
            analyzer = ReachabilityAnalyzer(
                arm_config, robot_config,
                other_arm_links=other_arm_links,
                dynamic_mode=self.dynamic_mode
            )
            result = analyzer.analyze(arm_name=arm_chain.name, color=color)

            # 保存结果
            prefix = f"reachability_{arm_chain.name}"
            analyzer.save_results(result, prefix=prefix)

            # 动态模式：保存对侧臂碰撞球配置
            if self.dynamic_mode and hasattr(analyzer.ik_solver, '_other_arm_collision_spheres'):
                other_spheres = analyzer.ik_solver._other_arm_collision_spheres
                if other_spheres:
                    from .dynamic_filter import DynamicReachabilityFilter
                    filter_obj = DynamicReachabilityFilter(
                        urdf_path=self.config.urdf_path,
                        arm_name=arm_chain.name,
                        base_link=arm_chain.base_link,
                        max_reachable_mask=result.reachable_mask,
                        grid_points=result.grid_points,
                        dexterity=result.dexterity,
                        manipulability=result.manipulability,
                        other_arm_spheres_local=other_spheres,
                    )
                    filter_obj.save_config(arm_config.output_dir)

            results[arm_chain.name] = result

            # 清理GPU内存
            clear_gpu_memory()

        total_time = time.time() - start_time

        # 打印汇总
        print("\n" + "=" * 60)
        print("多臂分析完成!")
        print("=" * 60)
        print(f"总时间: {total_time:.2f}秒")
        print("\n各臂统计:")
        for name, result in results.items():
            print(f"  {name}:")
            print(f"    可达点: {result.reachable_count} ({result.reachability_ratio*100:.1f}%)")
            print(f"    最大灵活度: {result.max_dexterity}")
            print(f"    平均可操作度: {result.mean_manipulability:.4f}")

        multi_result = MultiArmReachabilityResult(
            results=results,
            urdf_path=self.config.urdf_path,
            total_computation_time=total_time
        )

        # 保存汇总统计
        self._save_summary(multi_result)

        return multi_result

    def _save_summary(self, result: MultiArmReachabilityResult):
        """保存多臂分析汇总"""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        summary = result.to_dict()
        summary_path = os.path.join(output_dir, "multi_arm_summary.json")

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\n已保存汇总: {summary_path}")


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
    分析URDF文件的便捷函数

    参数:
        urdf_path: URDF文件路径
        ee_link: 末端执行器链接名
        base_link: 基座链接名
        resolution: 体素分辨率
        num_seeds: IK种子数量
        output_dir: 输出目录
        device: 计算设备

    返回:
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


def analyze_multi_arm(
    urdf_path: str,
    arm_mode: ArmMode = ArmMode.BOTH,
    resolution: float = 0.05,
    num_seeds: int = 32,
    output_dir: str = "./reachability_output",
    device: str = "cuda:0"
) -> MultiArmReachabilityResult:
    """
    多臂可达性分析的便捷函数

    参数:
        urdf_path: URDF文件路径
        arm_mode: 臂分析模式 (AUTO/LEFT/RIGHT/BOTH/ALL)
        resolution: 体素分辨率
        num_seeds: IK种子数量
        output_dir: 输出目录
        device: 计算设备

    返回:
        MultiArmReachabilityResult
    """
    config = ReachabilityConfig(
        urdf_path=urdf_path,
        output_dir=output_dir,
        device=device
    )
    config.voxel.resolution = resolution
    config.voxel.auto_detect = True
    config.ik.num_seeds = num_seeds
    config.multi_arm.mode = arm_mode

    analyzer = MultiArmReachabilityAnalyzer(config)
    return analyzer.analyze()
