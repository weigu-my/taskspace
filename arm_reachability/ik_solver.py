"""
多种子IK求解器 - 使用cuRobo实现GPU加速

本模块提供基于多种子策略的IK求解功能，用于为冗余机械臂寻找多个IK解。
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
    """单个目标位姿的IK求解结果"""
    target_position: np.ndarray       # [3] 目标位置
    target_orientation: np.ndarray    # [4] 目标姿态四元数 (w, x, y, z)
    success: bool                     # 是否找到至少一个解
    num_solutions: int                # 找到的有效IK解数量
    solutions: np.ndarray             # [num_seeds, n_joints] 关节配置
    solution_mask: np.ndarray         # [num_seeds] 有效解的布尔掩码
    position_errors: np.ndarray       # [num_seeds] 位置误差
    rotation_errors: np.ndarray       # [num_seeds] 旋转误差


@dataclass
class BatchIKResult:
    """批量IK求解结果"""
    success_mask: np.ndarray          # [batch_size] 每个位姿是否有解
    num_solutions: np.ndarray         # [batch_size] 每个位姿的解数量
    best_solutions: np.ndarray        # [batch_size, n_joints] 最优关节配置
    all_solutions: np.ndarray         # [batch_size, num_seeds, n_joints] 所有解
    solution_masks: np.ndarray        # [batch_size, num_seeds] 有效性掩码
    solve_time: float                 # 总求解时间


class MultiSeedIKSolver:
    """
    多种子IK求解器 - 使用cuRobo进行GPU加速计算

    该求解器从多个初始种子运行IK，以找到多个解，
    这对于分析冗余机械臂的可达性至关重要。
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        ik_config: IKConfig,
        device: str = "cuda:0"
    ):
        """
        初始化多种子IK求解器

        参数:
            robot_config: 从URDF解析器获取的机器人配置
            ik_config: IK求解器配置
            device: 计算设备
        """
        self.robot_config = robot_config
        self.ik_config = ik_config
        self.device = device
        self.ik_solver = None
        self.tensor_args = None
        self._current_batch_size = None  # 跟踪当前批次大小

        self._initialize_solver()

    def _initialize_solver(self):
        """初始化cuRobo IK求解器"""
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import RobotConfig as CuroboRobotConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

            # 设置张量参数
            self.tensor_args = TensorDeviceType(
                device=torch.device(self.device),
                dtype=torch.float32
            )

            # 生成cuRobo配置
            config_generator = CuroboConfigGenerator(self.robot_config)
            curobo_config = config_generator.generate()

            # 为cuRobo创建机器人配置
            robot_cfg = self._create_curobo_robot_config(curobo_config)

            # 创建IK求解器配置
            # 注意: 禁用CUDA graph以避免批次大小变化时的错误
            # CUDA graph 要求批次大小固定，但最后一个批次可能较小
            ik_solver_config = IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=None,
                tensor_args=self.tensor_args,
                num_seeds=self.ik_config.num_seeds,
                position_threshold=self.ik_config.position_threshold,
                rotation_threshold=self.ik_config.rotation_threshold,
                use_cuda_graph=False,  # 禁用CUDA graph以支持可变批次大小
                self_collision_check=self.ik_config.self_collision_check,
                self_collision_opt=self.ik_config.self_collision_check,
            )

            self.ik_solver = IKSolver(ik_solver_config)
            self.n_joints = len(self.robot_config.active_joints)

            print(f"[IK求解器] 已初始化，种子数: {self.ik_config.num_seeds}")
            print(f"[IK求解器] 机器人: {self.robot_config.name}, 自由度: {self.n_joints}")
            print(f"[IK求解器] 设备: {self.device}")

        except ImportError as e:
            print(f"[警告] cuRobo不可用: {e}")
            print("[警告] 使用模拟求解器")
            self._initialize_dummy_solver()

    def _create_curobo_robot_config(self, config_dict: Dict) -> Any:
        """从字典创建cuRobo机器人配置"""
        from curobo.types.robot import RobotConfig as CuroboRobotConfig

        # 获取活动关节
        active_joints = self.robot_config.active_joints
        joint_names = [j.name for j in active_joints]

        # 构建关节限位
        joint_limits = {
            'position': [
                [j.lower_limit for j in active_joints],
                [j.upper_limit for j in active_joints]
            ],
            'velocity': [[j.velocity_limit for j in active_joints]],
        }

        # 创建配置字典
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
        """初始化模拟求解器用于测试（无需cuRobo）"""
        self.ik_solver = None
        self.n_joints = len(self.robot_config.active_joints)
        print("[模拟求解器] 已初始化用于测试")

    def solve_single(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """
        为单个目标位姿求解IK（使用多种子）

        参数:
            position: 目标位置 [3]
            orientation: 目标姿态四元数 [4] (w, x, y, z)

        返回:
            包含所有找到解的IKResult
        """
        if self.ik_solver is None:
            return self._solve_single_dummy(position, orientation)

        from curobo.types.math import Pose

        # 创建目标位姿
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

        # 求解IK
        result = self.ik_solver.solve_batch(goal_pose)

        # 提取结果
        success = result.success.cpu().numpy()[0]
        solutions = result.solution.cpu().numpy()[0]  # [num_seeds, n_joints]

        # 获取所有种子的位置和旋转误差
        if hasattr(result, 'position_error') and result.position_error is not None:
            position_errors = result.position_error.cpu().numpy().flatten()
        else:
            position_errors = np.zeros(self.ik_config.num_seeds)

        if hasattr(result, 'rotation_error') and result.rotation_error is not None:
            rotation_errors = result.rotation_error.cpu().numpy().flatten()
        else:
            rotation_errors = np.zeros(self.ik_config.num_seeds)

        # 根据误差阈值创建解的有效性掩码
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
        """用于测试的模拟求解器"""
        # 模拟随机结果
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
        使用GPU加速批量求解IK

        参数:
            positions: 目标位置 [batch_size, 3]
            orientations: 目标姿态四元数 [batch_size, 4] (w, x, y, z)
            return_all_solutions: 如果为True，返回所有种子的解

        返回:
            包含所有位姿解的BatchIKResult
        """
        start_time = time.time()
        batch_size = len(positions)

        if self.ik_solver is None:
            return self._solve_batch_dummy(positions, orientations, return_all_solutions)

        from curobo.types.math import Pose

        # 创建目标位姿
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

        # 批量求解IK
        result = self.ik_solver.solve_batch(goal_pose)

        # 提取结果
        success_mask = result.success.cpu().numpy()
        solutions = result.solution.cpu().numpy()  # [batch_size, num_seeds, n_joints]

        # 每个位姿的最优解（取第一个成功的种子）
        best_solutions = solutions[:, 0, :]  # 取第一个种子作为最优解

        # 统计每个位姿的解数量
        num_solutions = np.zeros(batch_size, dtype=np.int32)

        # 解的有效性掩码
        solution_masks = np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool)

        # 对于成功的位置，统计所有有效种子
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
            # 使用成功掩码估计解数量
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
        """用于测试的模拟批量求解器"""
        start_time = time.time()
        batch_size = len(positions)

        # 根据与原点的距离模拟结果
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
        为多个位置和多个姿态求解IK

        这是可达性分析的主要方法，测试每个位置的多个姿态。

        参数:
            positions: 网格位置 [n_positions, 3]
            orientations: 要测试的姿态 [n_orientations, 4]
            batch_size: GPU批处理大小

        返回:
            元组包含:
                - reachable_mask: [n_positions] 任意姿态成功则为True
                - dexterity: [n_positions] 可达姿态数量
                - num_solutions: [n_positions, n_orientations] 每个位姿的解数量
        """
        n_positions = len(positions)
        n_orientations = len(orientations)

        print(f"[IK求解器] 测试 {n_positions} 个位置 × {n_orientations} 个姿态")
        print(f"[IK求解器] 总位姿数: {n_positions * n_orientations}")

        # 创建所有位置-姿态组合
        # positions: [n_positions, 3] -> [n_positions, n_orientations, 3]
        # orientations: [n_orientations, 4] -> [n_positions, n_orientations, 4]
        pos_expanded = np.tile(positions[:, np.newaxis, :], (1, n_orientations, 1))
        ori_expanded = np.tile(orientations[np.newaxis, :, :], (n_positions, 1, 1))

        # 展平用于批处理
        all_positions = pos_expanded.reshape(-1, 3)
        all_orientations = ori_expanded.reshape(-1, 4)
        total_poses = len(all_positions)

        # 分批处理
        all_success = []
        all_num_solutions = []

        num_batches = (total_poses + batch_size - 1) // batch_size
        print(f"[IK求解器] 处理 {num_batches} 个批次...")

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

            # 显示进度
            progress = (i + batch_size) / total_poses * 100
            print(f"\r[IK求解器] 进度: {min(progress, 100):.1f}%", end='', flush=True)

        print()  # 换行

        # 合并结果
        success = np.concatenate(all_success)
        num_solutions = np.concatenate(all_num_solutions)

        # 重塑为 [n_positions, n_orientations]
        success_grid = success.reshape(n_positions, n_orientations)
        num_solutions_grid = num_solutions.reshape(n_positions, n_orientations)

        # 计算每个位置的指标
        reachable_mask = np.any(success_grid, axis=1)
        dexterity = np.sum(success_grid, axis=1)

        print(f"[IK求解器] 可达位置: {np.sum(reachable_mask)} / {n_positions}")
        print(f"[IK求解器] 最大灵活度: {np.max(dexterity)}")

        return reachable_mask, dexterity, num_solutions_grid

    def get_forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算给定关节位置的正运动学

        参数:
            joint_positions: 关节位置 [batch_size, n_joints]

        返回:
            元组包含:
                - positions: 末端执行器位置 [batch_size, 3]
                - orientations: 末端执行器姿态 [batch_size, 4]
        """
        if self.ik_solver is None:
            # 模拟FK
            batch_size = len(joint_positions)
            positions = np.random.uniform(-1, 1, (batch_size, 3)).astype(np.float32)
            orientations = np.zeros((batch_size, 4), dtype=np.float32)
            orientations[:, 0] = 1.0  # w = 1 表示单位四元数
            return positions, orientations

        # 使用cuRobo运动学
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
        使用随机FK采样机器人的工作空间

        参数:
            n_samples: 要采样的随机关节配置数量

        返回:
            元组包含:
                - positions: 采样的末端执行器位置 [n_samples, 3]
                - joint_configs: 对应的关节配置 [n_samples, n_joints]
        """
        # 在限位范围内生成随机关节配置
        active_joints = self.robot_config.active_joints
        joint_configs = np.zeros((n_samples, self.n_joints), dtype=np.float32)

        for i, joint in enumerate(active_joints):
            joint_configs[:, i] = np.random.uniform(
                joint.lower_limit,
                joint.upper_limit,
                n_samples
            )

        # 计算FK
        positions, _ = self.get_forward_kinematics(joint_configs)

        return positions, joint_configs

    def estimate_workspace_bounds(
        self,
        n_samples: int = 10000,
        padding: float = 0.1
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        使用FK采样估计工作空间边界

        参数:
            n_samples: 用于估计的样本数量
            padding: 添加到边界的填充比例

        返回:
            (x_range, y_range, z_range) 元组，每个为 (min, max) 元组
        """
        print(f"[IK求解器] 使用 {n_samples} 个FK样本估计工作空间边界...")

        positions, _ = self.sample_workspace(n_samples)

        x_min, y_min, z_min = positions.min(axis=0)
        x_max, y_max, z_max = positions.max(axis=0)

        # ============================
        # 边界策略说明
        # - x/y：以 base_link 为中心(0,0)对称更直观，也避免出现“原点在 base 但范围偏一边”
        # - z：通常机械臂主要在 base 上方工作，保留非对称边界，但至少覆盖 z=0
        # ============================
        x_abs = float(max(abs(x_min), abs(x_max)))
        y_abs = float(max(abs(y_min), abs(y_max)))

        # 添加填充（按半径比例）
        x_half = x_abs * (1.0 + padding)
        y_half = y_abs * (1.0 + padding)

        # Z：也按 0 对称（覆盖下半部分），更适合“以 base 为中心”的可达空间展示
        z_abs = float(max(abs(z_min), abs(z_max)))
        z_half = z_abs * (1.0 + padding)

        # 给一个最小范围，避免采样不足导致范围过小
        min_xy_half = 0.5  # meters
        min_z_half = 0.5   # meters
        x_half = max(x_half, min_xy_half)
        y_half = max(y_half, min_xy_half)
        z_half = max(z_half, min_z_half)

        x_range = (-x_half, x_half)
        y_range = (-y_half, y_half)
        z_range = (-z_half, z_half)

        print(f"[IK求解器] 估计边界:")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range
