"""
基于体素的可达性分析器

本模块使用体素网格采样和GPU加速IK求解进行全面的可达性分析。
"""

import os
import json
import time
import numpy as np
import torch
import gc
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

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
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
    可达性分析主类

    该分析器:
    1. 解析URDF并创建cuRobo配置
    2. 生成体素网格用于工作空间采样
    3. 使用GPU加速的多种子IK测试可达性
    4. 计算灵活度（每个位置可达的姿态数量）
    5. 计算可操作度（Yoshikawa指标）
    """

    def __init__(self, config: ReachabilityConfig):
        """
        初始化可达性分析器

        参数:
            config: 可达性分析配置
        """
        self.config = config
        self.robot_config: Optional[RobotConfig] = None
        self.ik_solver: Optional[MultiSeedIKSolver] = None
        self.manipulability_calc: Optional[ManipulabilityCalculator] = None

        self._initialize()

    def _initialize(self):
        """初始化所有组件"""
        print("=" * 60)
        print("初始化可达性分析器")
        print("=" * 60)

        # 解析URDF
        print(f"\n[1/4] 解析URDF: {self.config.urdf_path}")
        parser = URDFParser(self.config.urdf_path)
        self.robot_config = parser.parse(
            base_link=self.config.base_link,
            ee_link=self.config.ee_link
        )

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
            device=self.config.device
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

    def analyze(self) -> ReachabilityResult:
        """
        执行完整的可达性分析

        返回:
            包含所有计算指标的ReachabilityResult
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("开始可达性分析")
        print("=" * 60)

        # 生成体素网格
        print("\n[Step 1] 生成体素网格...")
        grid_points = self.config.voxel.get_grid_points()
        grid_shape = self.config.voxel.get_grid_shape()
        n_points = len(grid_points)

        print(f"  网格形状: {grid_shape}")
        print(f"  分辨率: {self.config.voxel.resolution}m")
        print(f"  总体素数: {n_points}")

        # 生成姿态
        print("\n[Step 2] 生成姿态...")
        orientations = self._generate_orientations()
        n_orientations = len(orientations)
        print(f"  模式: {self.config.orientation.mode.value}")
        print(f"  姿态数量: {n_orientations}")

        # 执行IK求解
        print("\n[Step 3] 求解所有位置和姿态的IK...")
        reachable_mask, dexterity, num_solutions = self.ik_solver.solve_batch_multi_orientation(
            positions=grid_points,
            orientations=orientations,
            batch_size=self.config.ik.batch_size
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
            grid_points, orientations, reachable_mask,
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
        # 使用IK求解器的关节数，而不是URDF中的总关节数
        # cuRobo只使用从基座到末端的运动链中的关节
        n_joints = self.ik_solver.n_joints
        best_solutions = np.zeros((n_positions, n_joints), dtype=np.float32)

        # 获取可达位置的索引
        reachable_indices = np.where(reachable_mask)[0]
        n_reachable = len(reachable_indices)

        if n_reachable == 0:
            return best_solutions

        # 使用第一个姿态
        default_orientation = orientations[0]

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
        stats['ee_link'] = self.robot_config.ee_link
        stats['base_link'] = self.robot_config.base_link
        stats['num_dof'] = self.robot_config.num_dof
        stats['voxel_resolution'] = self.config.voxel.resolution
        stats['num_ik_seeds'] = self.config.ik.num_seeds

        with open(os.path.join(output_dir, f"{prefix}_stats.json"), 'w') as f:
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
