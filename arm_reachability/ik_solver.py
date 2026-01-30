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

        # 用于坐标变换的 Pinocchio 模型
        self.pin_model = None
        self.pin_data = None
        self.base_link_frame_id = None
        self._base_transform = None  # 世界坐标系到 base_link 的变换

        self._initialize_solver()
        self._initialize_transforms()

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

    def _initialize_transforms(self):
        """初始化坐标变换，用于将世界坐标转换为 base_link 坐标"""
        try:
            import pinocchio as pin

            # 加载 URDF 模型
            self.pin_model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            self.pin_data = self.pin_model.createData()

            # 查找 base_link 帧 ID
            self.base_link_frame_id = None
            for i, frame in enumerate(self.pin_model.frames):
                if frame.name == self.robot_config.base_link:
                    self.base_link_frame_id = i
                    break

            if self.base_link_frame_id is None:
                raise RuntimeError(f"未找到 base_link '{self.robot_config.base_link}' 的帧")

            # 计算零位姿下 base_link 在世界坐标系中的位置
            q_neutral = np.zeros(self.pin_model.nq)
            for i in range(self.pin_model.njoints):
                joint = self.pin_model.joints[i]
                if joint.nq > 0:
                    idx_q = joint.idx_q
                    q_neutral[idx_q:idx_q+joint.nq] = 0.0

            # 前向运动学
            pin.forwardKinematics(self.pin_model, self.pin_data, q_neutral)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            # 获取 base_link 在世界坐标系中的变换
            base_transform = self.pin_data.oMf[self.base_link_frame_id]

            # 存储 base_link 的位置和旋转
            self._base_transform = {
                'translation': base_transform.translation.copy(),  # [3]
                'rotation': base_transform.rotation.copy()         # [3, 3]
            }

            print(f"[IK求解器] base_link '{self.robot_config.base_link}' 在世界坐标系中的位置:")
            print(f"  位置: {self._base_transform['translation']}")
            print(f"[IK求解器] 将 FK 结果转换为相对于 base_link 的坐标")
            return

        except Exception as e:
            print(f"[IK求解器] Pinocchio 变换初始化失败: {e}")

        # 回退方案：使用 URDF 关节树估算 base_link 在世界坐标系中的变换
        fallback = self._compute_base_transform_from_urdf()
        if fallback is not None:
            self._base_transform = fallback
            print(f"[IK求解器] 使用URDF树估算 base_link '{self.robot_config.base_link}' 在世界坐标系中的位置:")
            print(f"  位置: {self._base_transform['translation']}")
            print(f"[IK求解器] 将 FK 结果转换为相对于 base_link 的坐标")
        else:
            print(f"[IK求解器] 将使用世界坐标系（不进行坐标变换）")

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    def _compute_base_transform_from_urdf(self) -> Optional[dict]:
        """在未安装 Pinocchio 时，使用 URDF 关节树估算 base_link 变换"""
        try:
            link_names = {l.name for l in self.robot_config.links}
            child_links = {j.child_link for j in self.robot_config.joints}
            roots = list(link_names - child_links)
            if not roots:
                roots = [self.robot_config.base_link]

            world_like = {"world", "map", "odom"}
            roots.sort(key=lambda x: (0 if x.lower() in world_like else 1, x))

            by_parent = {}
            for j in self.robot_config.joints:
                by_parent.setdefault(j.parent_link, []).append(j)

            T = {r: np.eye(4, dtype=np.float64) for r in roots}
            queue = list(roots)
            visited = set(roots)

            while queue:
                parent = queue.pop(0)
                for j in by_parent.get(parent, []):
                    child = j.child_link
                    if child in visited:
                        continue

                    T_origin = np.eye(4, dtype=np.float64)
                    T_origin[:3, 3] = np.array(j.origin_xyz, dtype=np.float64)
                    T_origin[:3, :3] = self._rpy_to_matrix(
                        j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2]
                    )

                    # 在零位姿下，关节运动为单位变换
                    T_child = T[parent] @ T_origin
                    T[child] = T_child
                    visited.add(child)
                    queue.append(child)

            if self.robot_config.base_link not in T:
                return None

            base_T = T[self.robot_config.base_link]
            return {
                'translation': base_T[:3, 3].copy(),
                'rotation': base_T[:3, :3].copy()
            }
        except Exception as e:
            print(f"[IK求解器] 无法使用URDF计算 base_link 变换: {e}")
            return None

    def get_base_transform(self) -> Optional[dict]:
        """获取 base_link 在世界坐标系中的变换（translation, rotation）"""
        return self._base_transform

    def solve_single(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """
        为单个目标位姿求解IK（使用多种子）

        参数:
            position: 目标位置 [3]（相对于 base_link）
            orientation: 目标姿态四元数 [4] (w, x, y, z)

        返回:
            包含所有找到解的IKResult
        """
        if self.ik_solver is None:
            return self._solve_single_dummy(position, orientation)

        from curobo.types.math import Pose

        # 将目标位置从 base_link 坐标系转换到世界坐标系（cuRobo 需要）
        world_position = self._transform_to_world_frame(position)

        # 创建目标位姿
        goal_pose = Pose(
            position=torch.tensor(
                world_position.reshape(1, 3),
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
        return_all_solutions: bool = True,
        positions_in_world: bool = False
    ) -> BatchIKResult:
        """
        使用GPU加速批量求解IK

        参数:
            positions: 目标位置 [batch_size, 3]
            orientations: 目标姿态四元数 [batch_size, 4] (w, x, y, z)
            return_all_solutions: 如果为True，返回所有种子的解
            positions_in_world: 如果为True，positions已经是world坐标系，不需要转换

        返回:
            包含所有位姿解的BatchIKResult
        """
        start_time = time.time()
        batch_size = len(positions)

        if self.ik_solver is None:
            return self._solve_batch_dummy(positions, orientations, return_all_solutions)

        from curobo.types.math import Pose

        # 如果位置已经是 world 坐标系，直接使用；否则从 base_link 转换
        if positions_in_world:
            world_positions = positions
        else:
            world_positions = self._transform_to_world_frame(positions)

        # 创建目标位姿
        goal_pose = Pose(
            position=torch.tensor(
                world_positions,
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
        batch_size: int = 1024,
        positions_in_world: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        为多个位置和多个姿态求解IK

        这是可达性分析的主要方法，测试每个位置的多个姿态。

        参数:
            positions: 网格位置 [n_positions, 3]
            orientations: 要测试的姿态 [n_orientations, 4]
            batch_size: GPU批处理大小
            positions_in_world: 如果为 True，positions 已经是 world 坐标系

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
                return_all_solutions=False,
                positions_in_world=positions_in_world
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

    def _transform_to_base_frame(self, world_positions: np.ndarray) -> np.ndarray:
        """
        将世界坐标系中的位置转换为 base_link 坐标系

        参数:
            world_positions: 世界坐标系中的位置 [batch_size, 3] 或 [3]

        返回:
            base_positions: base_link 坐标系中的位置 [batch_size, 3] 或 [3]
        """
        if self._base_transform is None:
            # 没有变换信息，直接返回原始位置
            return world_positions

        # 提取 base_link 的位置和旋转
        base_translation = self._base_transform['translation']
        base_rotation = self._base_transform['rotation']

        # 将位置从世界坐标系转换到 base_link 坐标系
        # p_base = R^T * (p_world - t_base)
        if world_positions.ndim == 1:
            # 单个位置 [3]
            relative_pos = world_positions - base_translation
            base_positions = base_rotation.T @ relative_pos
        else:
            # 批量位置 [batch_size, 3]
            relative_pos = world_positions - base_translation[np.newaxis, :]
            # 使用 einsum 保持数组连续性
            base_positions = np.einsum('ji,nj->ni', base_rotation, relative_pos)
            # 确保数组在内存中是连续的
            base_positions = np.ascontiguousarray(base_positions)

        return base_positions

    def _transform_to_world_frame(self, base_positions: np.ndarray) -> np.ndarray:
        """
        将 base_link 坐标系中的位置转换为世界坐标系

        参数:
            base_positions: base_link 坐标系中的位置 [batch_size, 3] 或 [3]

        返回:
            world_positions: 世界坐标系中的位置 [batch_size, 3] 或 [3]
        """
        if self._base_transform is None:
            # 没有变换信息，直接返回原始位置
            return base_positions

        # 提取 base_link 的位置和旋转
        base_translation = self._base_transform['translation']
        base_rotation = self._base_transform['rotation']

        # 将位置从 base_link 坐标系转换到世界坐标系
        # p_world = R * p_base + t_base
        if base_positions.ndim == 1:
            # 单个位置 [3]
            world_positions = base_rotation @ base_positions + base_translation
        else:
            # 批量位置 [batch_size, 3]
            # 使用 einsum 保持数组连续性，避免 cuRobo 的张量视图问题
            world_positions = np.einsum('ij,nj->ni', base_rotation, base_positions) + base_translation[np.newaxis, :]
            # 确保数组在内存中是连续的
            world_positions = np.ascontiguousarray(world_positions)

        return world_positions

    def get_forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算给定关节位置的正运动学

        参数:
            joint_positions: 关节位置 [batch_size, n_joints]

        返回:
            元组包含:
                - positions: 末端执行器位置（相对于 base_link）[batch_size, 3]
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

        world_positions = state.ee_position.cpu().numpy()
        orientations = state.ee_quaternion.cpu().numpy()  # (w, x, y, z)

        # 转换到 base_link 坐标系
        positions = self._transform_to_base_frame(world_positions)

        return positions, orientations

    def sample_workspace(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用随机FK采样机器人的工作空间

        参数:
            n_samples: 要采样的随机关节配置数量

        返回:
            元组包含:
                - positions: 采样的末端执行器位置（相对于 base_link）[n_samples, 3]
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
        padding: float = 0.5
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        使用FK采样估计工作空间边界（相对于 base_link）

        该方法通过随机采样关节配置并计算 FK，估计机械臂的工作空间边界。
        返回的边界是相对于 base_link 坐标系的，即 base_link 位于原点 (0, 0, 0)。

        重要：为了显示完整的可达空间边界（球形/环形），我们需要将采样范围
        扩展到 FK 可达范围之外。默认 padding=0.5 (50%) 可以较好地显示边界。

        参数:
            n_samples: 用于估计的样本数量
            padding: 添加到边界的填充比例（默认 0.5 = 50%，确保能看到可达边界）

        返回:
            (x_range, y_range, z_range) 元组，每个为 (min, max) 元组
            边界相对于 base_link 坐标系
        """
        print(f"[IK求解器] 使用 {n_samples} 个FK样本估计工作空间边界...")
        print(f"[IK求解器] 边界将相对于 base_link '{self.robot_config.base_link}' 计算")

        positions, _ = self.sample_workspace(n_samples)

        x_min, y_min, z_min = positions.min(axis=0)
        x_max, y_max, z_max = positions.max(axis=0)

        # ============================
        # 边界策略说明
        # FK 采样得到的是实际可达位置，为了显示完整的可达空间边界（类球形），
        # 需要将采样范围扩展到可达范围之外。
        #
        # - x/y：以 base_link 为中心对称
        # - z：同样对称，覆盖上下空间
        # - padding=0.5：扩展 50%，确保能看到可达边界
        # ============================
        x_abs = float(max(abs(x_min), abs(x_max)))
        y_abs = float(max(abs(y_min), abs(y_max)))
        z_abs = float(max(abs(z_min), abs(z_max)))

        # 添加填充（扩展范围以显示可达边界）
        x_half = x_abs * (1.0 + padding)
        y_half = y_abs * (1.0 + padding)
        z_half = z_abs * (1.0 + padding)

        # 给一个最小范围，避免采样不足导致范围过小
        min_half = 0.5  # meters
        x_half = max(x_half, min_half)
        y_half = max(y_half, min_half)
        z_half = max(z_half, min_half)

        x_range = (-x_half, x_half)
        y_range = (-y_half, y_half)
        z_range = (-z_half, z_half)

        print(f"[IK求解器] FK 可达范围:")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
        print(f"[IK求解器] 扩展后采样边界 (padding={padding*100:.0f}%):")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range

    def estimate_workspace_bounds_world(
        self,
        n_samples: int = 10000,
        padding: float = 0.5
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        在 world 坐标系下估计工作空间边界

        与 estimate_workspace_bounds 不同，此方法直接在 world 坐标系下计算边界，
        考虑了 base_link 相对于 world 的旋转。这对于双臂机器人特别重要，
        因为每条臂的 base_link 可能有不同的旋转角度。

        参数:
            n_samples: 用于估计的样本数量
            padding: 添加到边界的填充比例（默认 0.5 = 50%）

        返回:
            (x_range, y_range, z_range) 元组，每个为 (min, max) 元组
            边界在 world 坐标系下
        """
        print(f"[IK求解器] 使用 {n_samples} 个FK样本估计工作空间边界（world 坐标系）...")
        print(f"[IK求解器] base_link: '{self.robot_config.base_link}'")

        if self.ik_solver is None:
            print("[IK求解器] 求解器未初始化，回退到 base_link 坐标系")
            return self.estimate_workspace_bounds(n_samples, padding)

        # 在关节限位范围内生成随机关节配置
        active_joints = self.robot_config.active_joints
        joint_configs = np.zeros((n_samples, self.n_joints), dtype=np.float32)

        for i, joint in enumerate(active_joints):
            joint_configs[:, i] = np.random.uniform(
                joint.lower_limit,
                joint.upper_limit,
                n_samples
            )

        # 计算 FK，直接获取 world 坐标系的位置（不转换到 base_link）
        q = torch.tensor(joint_configs, dtype=torch.float32, device=self.device)
        state = self.ik_solver.fk(q)
        world_positions = state.ee_position.cpu().numpy()  # world 坐标系

        # 计算 world 坐标系下的边界
        x_min, y_min, z_min = world_positions.min(axis=0)
        x_max, y_max, z_max = world_positions.max(axis=0)

        # 计算范围并添加 padding
        x_range_val = x_max - x_min
        y_range_val = y_max - y_min
        z_range_val = z_max - z_min

        x_pad = x_range_val * padding / 2
        y_pad = y_range_val * padding / 2
        z_pad = z_range_val * padding / 2

        # 最小 padding
        min_pad = 0.25  # meters
        x_pad = max(x_pad, min_pad)
        y_pad = max(y_pad, min_pad)
        z_pad = max(z_pad, min_pad)

        x_range = (float(x_min - x_pad), float(x_max + x_pad))
        y_range = (float(y_min - y_pad), float(y_max + y_pad))
        z_range = (float(z_min - z_pad), float(z_max + z_pad))

        print(f"[IK求解器] FK 可达范围 (world 坐标系):")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
        print(f"[IK求解器] 扩展后采样边界 (padding={padding*100:.0f}%):")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range
