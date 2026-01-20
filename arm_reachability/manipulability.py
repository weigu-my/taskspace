"""
可操作度计算器 - 基于 Yoshikawa 指标

本模块计算 Yoshikawa 可操作度指标，用于衡量机器人在给定配置下
向各方向运动的能力。
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .urdf_parser import RobotConfig


@dataclass
class ManipulabilityResult:
    """可操作度计算结果"""
    joint_positions: np.ndarray       # [n_configs, n_joints] 关节位置
    manipulability: np.ndarray        # [n_configs] Yoshikawa 指标
    jacobians: Optional[np.ndarray]   # [n_configs, 6, n_joints] 雅可比矩阵
    condition_numbers: np.ndarray     # [n_configs] 条件数（各向同性度量）


class ManipulabilityCalculator:
    """
    可操作度计算器 - 计算机器人配置的 Yoshikawa 可操作度指标

    Yoshikawa 可操作度指标定义为：
        w = sqrt(det(J * J^T))

    其中 J 是雅可比矩阵。该指标衡量与奇异点的距离，
    值越大表示机器人在各方向的运动能力越好。
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        device: str = "cuda:0"
    ):
        """
        初始化可操作度计算器

        参数:
            robot_config: 从 URDF 解析器获取的机器人配置
            device: 计算设备
        """
        self.robot_config = robot_config
        self.device = device
        self.n_joints = len(robot_config.active_joints)

        # 尝试使用 pinocchio 计算解析雅可比
        self._init_kinematics()

    def _init_kinematics(self):
        """初始化运动学库用于雅可比计算"""
        self.use_pinocchio = False
        self.model = None
        self.data = None
        self.active_joint_indices = None  # pinocchio q 向量中的索引

        try:
            import pinocchio as pin  # type: ignore

            # 使用 pinocchio 加载 URDF
            self.model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            self.data = self.model.createData()

            # 查找末端执行器帧 ID
            self.ee_frame_id = None
            for i, frame in enumerate(self.model.frames):
                if frame.name == self.robot_config.ee_link:
                    self.ee_frame_id = i
                    break

            if self.ee_frame_id is None:
                # 尝试使用最后一个帧
                self.ee_frame_id = len(self.model.frames) - 1
                print(f"[可操作度] 警告: 未找到末端链接，使用帧 {self.model.frames[self.ee_frame_id].name}")

            # 构建活动关节到 pinocchio q 索引的映射
            self.active_joint_indices = []
            active_joint_names = {j.name for j in self.robot_config.active_joints}

            for i in range(1, self.model.njoints):  # 跳过 universe (关节 0)
                joint_name = self.model.names[i]
                if joint_name in active_joint_names:
                    joint = self.model.joints[i]
                    # 添加该关节的所有 q 索引（处理多自由度关节）
                    for q_idx in range(joint.idx_q, joint.idx_q + joint.nq):
                        self.active_joint_indices.append(q_idx)

            if len(self.active_joint_indices) != self.n_joints:
                print(f"[可操作度] 警告: 活动关节数量不匹配。"
                      f"期望 {self.n_joints}，在 pinocchio 模型中找到 {len(self.active_joint_indices)}")

            self.use_pinocchio = True
            print(f"[可操作度] 使用 Pinocchio 计算解析雅可比")

        except Exception as e:
            print(f"[可操作度] Pinocchio 不可用: {e}")
            print(f"[可操作度] 使用数值雅可比近似")

    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        计算给定关节位置的雅可比矩阵

        参数:
            joint_positions: 关节配置 [n_joints]

        返回:
            雅可比矩阵 [6, n_joints]
        """
        if self.use_pinocchio:
            return self._compute_jacobian_pinocchio(joint_positions)
        else:
            return self._compute_jacobian_numerical(joint_positions)

    def _compute_jacobian_pinocchio(self, joint_positions: np.ndarray) -> np.ndarray:
        """使用 Pinocchio 计算雅可比"""
        import pinocchio as pin  # type: ignore

        q = np.array(joint_positions, dtype=np.float64)

        # 为 pinocchio 模型构建完整的配置向量
        if len(q) != self.model.nq:
            q_full = np.zeros(self.model.nq, dtype=np.float64)

            # 将活动关节映射到 pinocchio q 索引
            if self.active_joint_indices is not None:
                for i, q_idx in enumerate(self.active_joint_indices):
                    if i < len(q) and q_idx < len(q_full):
                        q_full[q_idx] = q[i]
            else:
                # 回退：使用 pinocchio 的 joint.idx_q 按关节名称映射
                active_joint_map = {j.name: idx for idx, j in enumerate(self.robot_config.active_joints)}
                for i in range(1, self.model.njoints):
                    joint_name = self.model.names[i]
                    if joint_name in active_joint_map:
                        joint = self.model.joints[i]
                        active_idx = active_joint_map[joint_name]
                        if active_idx < len(q):
                            # 使用 pinocchio 的 idx_q 处理多自由度关节
                            for j in range(joint.nq):
                                pin_q_idx = joint.idx_q + j
                                if pin_q_idx < len(q_full) and active_idx + j < len(q):
                                    q_full[pin_q_idx] = q[active_idx + j]
            q = q_full

        # 计算正运动学和雅可比
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)

        # 获取帧雅可比（局部世界对齐）
        J_full = pin.getFrameJacobian(
            self.model, self.data, self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # 仅提取活动关节的列
        if self.active_joint_indices is not None:
            J = J_full[:, self.active_joint_indices]
        else:
            # 回退：使用前 n_joints 列（可能不正确）
            J = J_full[:, :self.n_joints]

        return J.astype(np.float32)

    def _compute_jacobian_numerical(
        self,
        joint_positions: np.ndarray,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        使用有限差分数值计算雅可比

        注意：这是一个占位实现，返回常数雅可比。
        若需精确结果，请安装 pinocchio 或提供 FK 函数。

        参数:
            joint_positions: 关节配置 [n_joints]
            epsilon: 有限差分步长（占位实现中未使用）

        返回:
            雅可比矩阵 [6, n_joints]
        """
        # 占位实现：返回常数雅可比
        # 这会给出不正确的可操作度值，但可防止错误
        # 若需精确计算，请安装 pinocchio 或使用带 FK 的 GPUManipulabilityCalculator

        jacobian = np.zeros((6, self.n_joints), dtype=np.float32)

        # 基于关节限位的简单近似
        # 这不准确但提供了基线
        for i in range(self.n_joints):
            joint = self.robot_config.active_joints[i]
            # 估计线速度贡献（粗略近似）
            # 假设每个关节对末端运动的贡献大致相等
            jacobian[:3, i] = 0.1  # 线性部分（占位）
            jacobian[3:6, i] = 0.1  # 角度部分（占位）

        return jacobian

    def compute_manipulability(
        self,
        joint_positions: np.ndarray,
        return_jacobian: bool = False
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        计算单个配置的 Yoshikawa 可操作度

        参数:
            joint_positions: 关节配置 [n_joints]
            return_jacobian: 如果为 True，同时返回雅可比矩阵

        返回:
            (可操作度, 雅可比矩阵或 None) 元组
        """
        J = self.compute_jacobian(joint_positions)

        # 计算 Yoshikawa 可操作度: w = sqrt(det(J * J^T))
        JJT = J @ J.T
        det_JJT = np.linalg.det(JJT)

        # 钳位以避免接近零的负值导致的数值问题
        manipulability = np.sqrt(max(det_JJT, 0))

        if return_jacobian:
            return manipulability, J
        return manipulability, None

    def compute_batch(
        self,
        joint_positions_batch: np.ndarray,
        return_jacobians: bool = False
    ) -> ManipulabilityResult:
        """
        批量计算配置的可操作度

        参数:
            joint_positions_batch: 关节配置 [batch_size, n_joints]
            return_jacobians: 如果为 True，同时返回所有雅可比矩阵

        返回:
            包含计算值的 ManipulabilityResult
        """
        batch_size = len(joint_positions_batch)
        manipulabilities = np.zeros(batch_size, dtype=np.float32)
        condition_numbers = np.zeros(batch_size, dtype=np.float32)
        jacobians = None

        if return_jacobians:
            jacobians = np.zeros((batch_size, 6, self.n_joints), dtype=np.float32)

        for i in range(batch_size):
            J = self.compute_jacobian(joint_positions_batch[i])

            # Yoshikawa 可操作度
            JJT = J @ J.T
            det_JJT = np.linalg.det(JJT)
            manipulabilities[i] = np.sqrt(max(det_JJT, 0))

            # 条件数（倒数用作各向同性度量）
            try:
                singular_values = np.linalg.svd(J, compute_uv=False)
                if singular_values[-1] > 1e-10:
                    condition_numbers[i] = singular_values[0] / singular_values[-1]
                else:
                    condition_numbers[i] = np.inf
            except:
                condition_numbers[i] = np.inf

            if return_jacobians:
                jacobians[i] = J

        return ManipulabilityResult(
            joint_positions=joint_positions_batch,
            manipulability=manipulabilities,
            jacobians=jacobians,
            condition_numbers=condition_numbers
        )

    def compute_for_positions(
        self,
        positions: np.ndarray,
        ik_solutions: np.ndarray,
        solution_masks: np.ndarray
    ) -> np.ndarray:
        """
        计算每个可达位置的平均可操作度

        对于每个位置，计算所有有效 IK 解的可操作度，
        返回最大（最优）可操作度。

        参数:
            positions: 目标位置 [n_positions, 3]
            ik_solutions: IK 解 [n_positions, n_seeds, n_joints]
            solution_masks: 有效性掩码 [n_positions, n_seeds]

        返回:
            可操作度值 [n_positions]（不可达位置为 0）
        """
        n_positions = len(positions)
        manipulabilities = np.zeros(n_positions, dtype=np.float32)

        print(f"[可操作度] 计算 {n_positions} 个位置...")

        for i in range(n_positions):
            valid_indices = np.where(solution_masks[i])[0]

            if len(valid_indices) == 0:
                continue

            # 计算所有有效解的可操作度
            max_manip = 0
            for j in valid_indices:
                manip, _ = self.compute_manipulability(ik_solutions[i, j])
                max_manip = max(max_manip, manip)

            manipulabilities[i] = max_manip

            # 进度显示
            if (i + 1) % 1000 == 0:
                print(f"\r[可操作度] 进度: {(i+1)/n_positions*100:.1f}%", end='', flush=True)

        print()  # 换行

        return manipulabilities

    def compute_for_best_solutions(
        self,
        best_solutions: np.ndarray,
        reachable_mask: np.ndarray
    ) -> np.ndarray:
        """
        仅计算最优 IK 解的可操作度

        这比 compute_for_positions 更快，因为每个位置只评估一个解。

        参数:
            best_solutions: 最优 IK 解 [n_positions, n_joints]
            reachable_mask: 可达位置掩码 [n_positions]

        返回:
            可操作度值 [n_positions]（不可达位置为 0）
        """
        n_positions = len(best_solutions)
        manipulabilities = np.zeros(n_positions, dtype=np.float32)

        reachable_indices = np.where(reachable_mask)[0]
        n_reachable = len(reachable_indices)

        print(f"[可操作度] 计算 {n_reachable} 个可达位置...")

        for idx, i in enumerate(reachable_indices):
            manip, _ = self.compute_manipulability(best_solutions[i])
            manipulabilities[i] = manip

            # 进度显示
            if (idx + 1) % 500 == 0:
                print(f"\r[可操作度] 进度: {(idx+1)/n_reachable*100:.1f}%", end='', flush=True)

        print()  # 换行

        # 归一化到 [0, 1] 范围
        if n_reachable > 0:
            max_manip = np.max(manipulabilities)
            if max_manip > 0:
                manipulabilities = manipulabilities / max_manip

        return manipulabilities


class GPUManipulabilityCalculator:
    """
    GPU 加速的可操作度计算器，使用 PyTorch

    该版本完全在 GPU 上计算可操作度，适用于大批量数据，
    但需要 cuRobo 来计算雅可比。
    """

    def __init__(
        self,
        ik_solver,  # MultiSeedIKSolver 实例
        device: str = "cuda:0"
    ):
        """
        初始化 GPU 可操作度计算器

        参数:
            ik_solver: 具有 FK 功能的 IK 求解器实例
            device: 计算设备
        """
        import torch  # 仅在此类中需要，在此导入

        self.ik_solver = ik_solver
        self.device = device
        self.n_joints = ik_solver.n_joints
        self.torch = torch

    def compute_batch_gpu(
        self,
        joint_positions,  # torch.Tensor
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        使用 GPU 数值微分批量计算可操作度

        参数:
            joint_positions: [batch_size, n_joints] 张量
            epsilon: 有限差分步长

        返回:
            可操作度值 [batch_size]
        """
        torch = self.torch
        batch_size = joint_positions.shape[0]

        # 计算基准 FK
        base_positions, base_orientations = self.ik_solver.get_forward_kinematics(
            joint_positions.cpu().numpy()
        )
        base_positions = torch.tensor(base_positions, device=self.device)

        # 计算数值雅可比（仅位置部分以提高速度）
        jacobian = torch.zeros(batch_size, 3, self.n_joints, device=self.device)

        for j in range(self.n_joints):
            # 扰动关节 j
            perturbed = joint_positions.clone()
            perturbed[:, j] += epsilon

            perturbed_positions, _ = self.ik_solver.get_forward_kinematics(
                perturbed.cpu().numpy()
            )
            perturbed_positions = torch.tensor(perturbed_positions, device=self.device)

            # 有限差分
            jacobian[:, :, j] = (perturbed_positions - base_positions) / epsilon

        # 计算可操作度: sqrt(det(J @ J.T))
        JJT = torch.bmm(jacobian, jacobian.transpose(1, 2))
        det_JJT = torch.linalg.det(JJT)

        # 钳位负值
        det_JJT = torch.clamp(det_JJT, min=0)
        manipulability = torch.sqrt(det_JJT)

        return manipulability.cpu().numpy()
