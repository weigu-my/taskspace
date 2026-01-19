#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
通用机械臂可达性分析框架 (Universal Arm Reachability Analysis)
===========================================================

改进点：
1. 使用 cuRobo IKSolver 而非 MotionGen，专注于 IK 求解
2. 支持 batch IK，充分利用 GPU 并行能力
3. 修正四元数旋转误差计算（使用角度误差而非欧氏距离）
4. 通用化设计：支持单臂、双臂、任意机器人配置
5. 支持自动估计工作空间边界
6. 支持多种末端姿态采样策略
7. 配置与代码分离，使用 dataclass 管理配置

作者：自动生成
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

# ===================================================================
# cuRobo 导入（根据版本可能需要调整）
# ===================================================================
try:
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.types.base import TensorDeviceType
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.sdf.world import WorldCollision
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.util_file import load_yaml, get_robot_configs_path, join_path, get_robot_path
    CUROBO_AVAILABLE = True
except ImportError:
    CUROBO_AVAILABLE = False
    print("[WARN] cuRobo 未安装，将使用模拟模式进行演示")


# ===================================================================
# 配置数据类
# ===================================================================

class OrientationMode(Enum):
    """末端姿态采样模式"""
    FIXED = "fixed"                    # 固定单一姿态
    MULTI_FIXED = "multi_fixed"        # 多个预定义的固定姿态
    RANDOM = "random"                  # 随机采样姿态
    SPHERICAL = "spherical"            # 球面均匀采样


@dataclass
class ArmConfig:
    """单个机械臂的配置"""
    name: str                          # 臂的名称（如 "left_arm", "right_arm"）
    ee_link: str = ""                  # 末端执行器链接名（可自动检测）

    # 机器人模型配置（二选一）
    # 方式1: 直接使用 URDF 文件（推荐，会自动转换为 cuRobo 配置）
    urdf_path: Optional[str] = None
    # 方式2: 使用已有的 cuRobo 配置文件
    robot_config_file: Optional[str] = None
    world_config_file: Optional[str] = None

    # 基座链接（可选，自动检测）
    base_link: str = ""

    # 工作空间边界（可选，若为 None 则自动估计）
    bbox: Optional[dict] = None        # {"x": (min, max), "y": (...), "z": (...)}

    # IK 求解参数
    position_threshold: float = 0.005  # 位置阈值 (m)
    rotation_threshold: float = 0.05   # 旋转阈值 (rad)
    num_seeds: int = 32                # IK 初始种子数量

    # 自碰撞检测
    self_collision_check: bool = True  # 是否检测自碰撞
    self_collision_opt: bool = True    # 是否优化避免自碰撞


@dataclass
class ReachabilityConfig:
    """可达性分析的全局配置"""
    # 机械臂配置列表
    arms: list[ArmConfig] = field(default_factory=list)

    # 体素网格参数
    voxel_size: float = 0.05           # 体素大小 (m)

    # 末端姿态参数
    orientation_mode: OrientationMode = OrientationMode.FIXED
    fixed_orientations: list[np.ndarray] = field(default_factory=lambda: [
        np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] 单位四元数
    ])
    num_random_orientations: int = 1   # 随机模式下每个位置采样的姿态数

    # 灵巧性分析参数
    compute_dexterity: bool = True     # 是否计算灵巧性
    num_dexterity_orientations: int = 12  # 测试灵巧性的姿态数量

    # 批处理参数
    batch_size: int = 1024             # GPU batch 大小

    # 输出配置
    output_dir: str = "./reachability_output"
    save_failed_points: bool = False   # 是否保存失败点


# ===================================================================
# 工具函数
# ===================================================================

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """四元数归一化"""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    计算两个四元数之间的角度距离 (rad)

    使用公式: angle = 2 * arccos(|q1 · q2|)
    注意: q 和 -q 表示同一旋转，所以取绝对值
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)  # 数值稳定性
    return 2.0 * np.arccos(dot)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵 [w, x, y, z] -> R (3x3)"""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数 R (3x3) -> [w, x, y, z]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def sample_uniform_quaternions(n: int, seed: Optional[int] = None) -> np.ndarray:
    """在 SO(3) 上均匀采样 n 个四元数 (Shoemake's method)"""
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0, 1, (n, 3))
    sqrt1_u0 = np.sqrt(1 - u[:, 0])
    sqrt_u0 = np.sqrt(u[:, 0])

    quats = np.zeros((n, 4))
    quats[:, 0] = sqrt1_u0 * np.sin(2 * np.pi * u[:, 1])  # w
    quats[:, 1] = sqrt1_u0 * np.cos(2 * np.pi * u[:, 1])  # x
    quats[:, 2] = sqrt_u0 * np.sin(2 * np.pi * u[:, 2])   # y
    quats[:, 3] = sqrt_u0 * np.cos(2 * np.pi * u[:, 2])   # z

    return quats


def build_voxel_grid(bbox: dict, voxel_size: float) -> np.ndarray:
    """构建体素网格点集"""
    xs = np.arange(bbox["x"][0], bbox["x"][1] + 1e-9, voxel_size)
    ys = np.arange(bbox["y"][0], bbox["y"][1] + 1e-9, voxel_size)
    zs = np.arange(bbox["z"][0], bbox["z"][1] + 1e-9, voxel_size)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    return grid.reshape(-1, 3)


# ===================================================================
# IK 求解器抽象基类
# ===================================================================

class IKSolverBase(ABC):
    """IK 求解器的抽象基类"""

    @abstractmethod
    def solve_batch(
        self,
        positions: np.ndarray,       # (N, 3)
        orientations: np.ndarray,    # (N, 4) [w, x, y, z]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量求解 IK

        Returns:
            success: (N,) bool 数组
            position_errors: (N,) 位置误差
            rotation_errors: (N,) 旋转误差 (rad)
        """
        pass

    @abstractmethod
    def estimate_workspace_bbox(self, num_samples: int = 10000) -> dict:
        """通过 FK 随机采样估计工作空间边界"""
        pass


# ===================================================================
# cuRobo IK 求解器实现
# ===================================================================

class CuroboIKSolver(IKSolverBase):
    """基于 cuRobo 的高效 batch IK 求解器"""

    def __init__(self, arm_config: ArmConfig, device: str = "cuda:0"):
        if not CUROBO_AVAILABLE:
            raise RuntimeError("cuRobo 未安装")

        self.config = arm_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"[INFO] 初始化 cuRobo IKSolver: {arm_config.name}")
        print(f"[INFO] 使用设备: {self.device}")

        # 创建 TensorDeviceType 对象
        self.tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)

        # 加载机器人配置
        from pathlib import Path
        import yaml

        robot_cfg_path = Path(arm_config.robot_config_file).absolute()
        print(f"[DEBUG] Loading robot config from: {robot_cfg_path}")

        # 加载 YAML 配置
        with open(robot_cfg_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # 提取机器人配置的内层结构
        robot_cfg_content = full_config.get('robot_cfg', full_config)

        # 调试输出
        print(f"[DEBUG] Robot config keys: {robot_cfg_content.keys()}")
        if 'kinematics' in robot_cfg_content:
            kin_cfg = robot_cfg_content['kinematics']
            print(f"[DEBUG] Kinematics keys: {kin_cfg.keys()}")
            if 'cspace' in kin_cfg:
                cspace = kin_cfg['cspace']
                print(f"[DEBUG] CSpace keys: {cspace.keys()}")
                n_joints = len(cspace.get('joint_names', []))
                print(f"[DEBUG] joint_names count: {n_joints}")
                print(f"[DEBUG] cspace_distance_weight count: {len(cspace.get('cspace_distance_weight', []))}")

        # 尝试使用 RobotConfig.from_dict 或类似方法
        # 如果不可用，则使用 load_from_robot_config
        try:
            # 方法1: 尝试直接创建 RobotConfig
            robot_config = RobotConfig.from_dict(robot_cfg_content, self.tensor_args)
            print("[DEBUG] Successfully created RobotConfig from dict")

            # 使用 RobotConfig 对象创建 IKSolverConfig
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_config,
                None,  # 无世界碰撞
                rotation_threshold=arm_config.rotation_threshold,
                position_threshold=arm_config.position_threshold,
                num_seeds=arm_config.num_seeds,
                self_collision_check=arm_config.self_collision_check,
                self_collision_opt=arm_config.self_collision_opt,
                tensor_args=self.tensor_args,
                use_cuda_graph=False,
            )
        except (AttributeError, TypeError) as e:
            print(f"[DEBUG] RobotConfig.from_dict not available: {e}")
            print("[DEBUG] Trying alternative method...")

            # 方法2: 使用 URDF 路径直接创建配置
            # 从配置中获取 URDF 路径
            urdf_path = robot_cfg_content['kinematics']['urdf_path']
            base_link = robot_cfg_content['kinematics']['base_link']
            ee_link = robot_cfg_content['kinematics']['ee_link']
            cspace = robot_cfg_content['kinematics']['cspace']

            # 构建 cuRobo 期望的格式
            # 注意: load_from_robot_config 期望特定的字典结构
            robot_dict = {
                'kinematics': {
                    'urdf_path': urdf_path,
                    'base_link': base_link,
                    'ee_link': ee_link,
                    'cspace': cspace,
                }
            }

            print(f"[DEBUG] Using URDF path: {urdf_path}")

            # 世界配置：传递 None 禁用环境碰撞检测
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_dict,
                None,
                rotation_threshold=arm_config.rotation_threshold,
                position_threshold=arm_config.position_threshold,
                num_seeds=arm_config.num_seeds,
                self_collision_check=arm_config.self_collision_check,
                self_collision_opt=arm_config.self_collision_opt,
                tensor_args=self.tensor_args,
                use_cuda_graph=False,
            )

        self.ik_solver = IKSolver(ik_config)

        # 获取机器人模型用于 FK
        self.robot_model = self.ik_solver.robot_config.kinematics
        self.joint_limits = self._get_joint_limits()

        # 预热
        print("[INFO] 预热 IK 求解器...")
        self._warmup()
        print("[INFO] 初始化完成")

    def _get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """获取关节限位"""
        # cuRobo 使用 get_joint_limits() 方法
        joint_limits = self.robot_model.get_joint_limits()
        lower = joint_limits.position[0].cpu().numpy()
        upper = joint_limits.position[1].cpu().numpy()
        return lower, upper

    def _warmup(self):
        """预热 GPU 内核"""
        dummy_pos = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        dummy_quat = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32)
        goal = Pose(position=dummy_pos, quaternion=dummy_quat)
        _ = self.ik_solver.solve_batch(goal)

    def solve_batch(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量求解 IK

        Args:
            positions: (N, 3) 目标位置
            orientations: (N, 4) 目标姿态 [w, x, y, z]

        Returns:
            success: (N,) 是否成功
            position_errors: (N,) 位置误差
            rotation_errors: (N,) 旋转误差
        """
        n = len(positions)

        # 转换为 torch tensor
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.float32)

        # cuRobo 使用 [w, x, y, z] 格式，确保归一化
        orientations = np.array([quat_normalize(q) for q in orientations])
        quat_tensor = torch.tensor(orientations, device=self.device, dtype=torch.float32)

        # 构建 Pose 对象
        goal_pose = Pose(position=pos_tensor, quaternion=quat_tensor)

        # 求解 IK
        result = self.ik_solver.solve_batch(goal_pose)

        # 提取结果 (注意: cuRobo 可能返回 (N,1) 形状，需要 flatten)
        success = result.success.cpu().numpy().astype(bool).flatten()

        # 获取达到的位姿并计算误差
        # 初始化默认误差值
        position_errors = np.full(n, np.inf)
        rotation_errors = np.full(n, np.inf)

        if hasattr(result, 'goal_pose') and result.goal_pose is not None:
            achieved_pos = result.goal_pose.position.cpu().numpy().reshape(-1, 3)
            achieved_quat = result.goal_pose.quaternion.cpu().numpy().reshape(-1, 4)

            position_errors = np.linalg.norm(achieved_pos - positions, axis=1).flatten()
            rotation_errors = np.array([
                quat_angular_distance(q1, q2)
                for q1, q2 in zip(achieved_quat, orientations)
            ]).flatten()

        elif hasattr(result, 'solution') and result.solution is not None:
            # 如果没有返回达到的位姿，则使用 FK 计算
            q_sol = result.solution.position
            # 执行 FK 获取达到的位姿
            state = JointState.from_position(q_sol, self.robot_model.joint_names)
            fk_result = self.robot_model.get_state(state)
            achieved_pos = fk_result.ee_position.cpu().numpy().reshape(-1, 3)
            achieved_quat = fk_result.ee_quaternion.cpu().numpy().reshape(-1, 4)

            position_errors = np.linalg.norm(achieved_pos - positions, axis=1).flatten()
            rotation_errors = np.array([
                quat_angular_distance(q1, q2)
                for q1, q2 in zip(achieved_quat, orientations)
            ]).flatten()

        return success, position_errors, rotation_errors

    def estimate_workspace_bbox(
        self,
        num_samples: int = 10000,
        margin: float = 0.05
    ) -> dict:
        """通过随机 FK 采样估计工作空间边界"""
        print(f"[INFO] 通过 FK 采样估计工作空间边界 (n={num_samples})...")

        lower, upper = self.joint_limits
        n_joints = len(lower)

        # 随机采样关节配置
        q_samples = np.random.uniform(
            lower, upper,
            size=(num_samples, n_joints)
        )

        # 批量 FK
        q_tensor = torch.tensor(q_samples, device=self.device, dtype=torch.float32)

        # 获取末端位置
        ee_positions = []
        batch_size = 1024

        for i in range(0, num_samples, batch_size):
            batch_q = q_tensor[i:i+batch_size]
            state = JointState.from_position(batch_q, self.robot_model.joint_names)
            fk_result = self.robot_model.get_state(state)
            ee_positions.append(fk_result.ee_position.cpu().numpy())

        ee_positions = np.vstack(ee_positions)

        # 计算边界（添加一点余量）
        bbox = {
            "x": (float(ee_positions[:, 0].min() - margin),
                  float(ee_positions[:, 0].max() + margin)),
            "y": (float(ee_positions[:, 1].min() - margin),
                  float(ee_positions[:, 1].max() + margin)),
            "z": (float(ee_positions[:, 2].min() - margin),
                  float(ee_positions[:, 2].max() + margin)),
        }

        print(f"[INFO] 估计的工作空间: {bbox}")
        return bbox


# ===================================================================
# 模拟 IK 求解器（用于演示/测试）
# ===================================================================

class DummyIKSolver(IKSolverBase):
    """模拟 IK 求解器，用于无 cuRobo 环境下的演示"""

    def __init__(
        self,
        arm_config: ArmConfig,
        base_position: np.ndarray = np.array([0, 0, 0]),
        reach_radius: float = 0.8,
    ):
        self.config = arm_config
        self.base_position = base_position
        self.reach_radius = reach_radius
        self.min_reach = 0.2
        print(f"[INFO] 使用模拟 IK 求解器: {arm_config.name}")

    def solve_batch(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """模拟 IK 求解：简单的球形可达范围"""
        distances = np.linalg.norm(positions - self.base_position, axis=1)

        # 在球壳范围内视为可达
        success = (distances < self.reach_radius) & (distances > self.min_reach)

        # 模拟误差
        position_errors = np.where(success, np.random.uniform(0, 0.002, len(positions)), 0.1)
        rotation_errors = np.where(success, np.random.uniform(0, 0.01, len(positions)), 0.5)

        return success, position_errors, rotation_errors

    def estimate_workspace_bbox(self, num_samples: int = 10000) -> dict:
        """返回基于球形可达范围的估计"""
        r = self.reach_radius
        b = self.base_position
        return {
            "x": (b[0] - r, b[0] + r),
            "y": (b[1] - r, b[1] + r),
            "z": (b[2] - r, b[2] + r),
        }


# ===================================================================
# 可达性分析器
# ===================================================================

class ReachabilityAnalyzer:
    """通用机械臂可达性分析器"""

    def __init__(self, config: ReachabilityConfig):
        self.config = config
        self.solvers: dict[str, IKSolverBase] = {}
        self.results: dict[str, dict] = {}

        os.makedirs(config.output_dir, exist_ok=True)

        # 为每个臂初始化求解器
        for arm_cfg in config.arms:
            # 如果提供了 URDF 路径，自动转换为 cuRobo 配置
            if arm_cfg.urdf_path and not arm_cfg.robot_config_file:
                arm_cfg = self._convert_urdf_to_curobo(arm_cfg)

            if CUROBO_AVAILABLE and arm_cfg.robot_config_file:
                self.solvers[arm_cfg.name] = CuroboIKSolver(arm_cfg)
            else:
                # 使用模拟求解器
                base_pos = np.array([0, 0.3, 0]) if "left" in arm_cfg.name.lower() else np.array([0, -0.3, 0])
                self.solvers[arm_cfg.name] = DummyIKSolver(arm_cfg, base_position=base_pos)

    def _convert_urdf_to_curobo(self, arm_cfg: ArmConfig) -> ArmConfig:
        """将 URDF 转换为 cuRobo 配置"""
        try:
            from urdf_to_curobo import urdf_to_curobo_config, URDFParser

            print(f"[INFO] 转换 URDF: {arm_cfg.urdf_path}")

            # 解析 URDF 获取信息
            parser = URDFParser(arm_cfg.urdf_path)

            # 生成 cuRobo 配置
            config_dir = os.path.join(self.config.output_dir, "curobo_configs")
            robot_cfg_path, world_cfg_path = urdf_to_curobo_config(
                arm_cfg.urdf_path,
                ee_link=arm_cfg.ee_link or None,
                base_link=arm_cfg.base_link or None,
                output_dir=config_dir,
            )

            # 更新配置
            arm_cfg.robot_config_file = robot_cfg_path
            arm_cfg.world_config_file = world_cfg_path
            if not arm_cfg.ee_link:
                arm_cfg.ee_link = parser.config.ee_link
            if not arm_cfg.base_link:
                arm_cfg.base_link = parser.config.base_link

            print(f"[INFO] cuRobo 配置已生成: {robot_cfg_path}")

        except ImportError:
            print("[ERROR] urdf_to_curobo 模块未找到，请确保 urdf_to_curobo.py 在同一目录")
        except Exception as e:
            print(f"[ERROR] URDF 转换失败: {e}")

        return arm_cfg

    def _get_orientations(self, num_points: int) -> np.ndarray:
        """根据配置获取末端姿态"""
        mode = self.config.orientation_mode

        if mode == OrientationMode.FIXED:
            # 单一固定姿态，扩展到所有点
            quat = quat_normalize(self.config.fixed_orientations[0])
            return np.tile(quat, (num_points, 1))

        elif mode == OrientationMode.MULTI_FIXED:
            # 多个固定姿态循环使用
            quats = [quat_normalize(q) for q in self.config.fixed_orientations]
            n_quats = len(quats)
            indices = np.arange(num_points) % n_quats
            return np.array([quats[i] for i in indices])

        elif mode == OrientationMode.RANDOM:
            return sample_uniform_quaternions(num_points)

        elif mode == OrientationMode.SPHERICAL:
            return sample_uniform_quaternions(num_points)

        else:
            raise ValueError(f"未知的姿态模式: {mode}")

    def analyze_arm(self, arm_name: str) -> dict:
        """分析单个机械臂的可达性和灵巧性"""
        if arm_name not in self.solvers:
            raise ValueError(f"未找到机械臂: {arm_name}")

        solver = self.solvers[arm_name]
        arm_cfg = next(a for a in self.config.arms if a.name == arm_name)

        # 确定工作空间边界
        if arm_cfg.bbox is not None:
            bbox = arm_cfg.bbox
        else:
            bbox = solver.estimate_workspace_bbox()

        # 构建网格
        grid_points = build_voxel_grid(bbox, self.config.voxel_size)
        n_points = len(grid_points)
        print(f"[INFO] {arm_name}: 网格点数 = {n_points}")

        # 生成用于灵巧性测试的多个姿态
        n_orientations = self.config.num_dexterity_orientations
        test_orientations = sample_uniform_quaternions(n_orientations, seed=42)
        print(f"[INFO] 测试姿态数: {n_orientations}")

        # 灵巧性分数：每个点能用多少种姿态到达
        dexterity_scores = np.zeros(n_points, dtype=np.int32)

        batch_size = self.config.batch_size
        t0 = time.time()

        # 对每种姿态进行测试
        for ori_idx, orientation in enumerate(test_orientations):
            print(f"[RUN] 测试姿态 {ori_idx+1}/{n_orientations}...")

            # 为所有点使用同一姿态
            orientations = np.tile(orientation, (n_points, 1))

            n_batches = (n_points + batch_size - 1) // batch_size

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_points)

                batch_pos = grid_points[start_idx:end_idx]
                batch_quat = orientations[start_idx:end_idx]

                success, _, _ = solver.solve_batch(batch_pos, batch_quat)

                # 累加成功次数
                dexterity_scores[start_idx:end_idx] += success.astype(np.int32)

            # 显示进度
            elapsed = time.time() - t0
            n_reachable = np.sum(dexterity_scores > 0)
            avg_dexterity = np.mean(dexterity_scores[dexterity_scores > 0]) if n_reachable > 0 else 0
            print(f"  姿态 {ori_idx+1} 完成 | 可达点: {n_reachable} | "
                  f"平均灵巧性: {avg_dexterity:.2f}/{ori_idx+1} | 耗时: {elapsed:.1f}s")

        total_time = time.time() - t0

        # 收集结果
        reachable_mask = dexterity_scores > 0
        reachable_points = grid_points[reachable_mask]
        reachable_dexterity = dexterity_scores[reachable_mask]
        unreachable_points = grid_points[~reachable_mask]

        # 归一化灵巧性分数 (0-1)
        normalized_dexterity = reachable_dexterity / n_orientations

        result = {
            "arm_name": arm_name,
            "bbox": bbox,
            "voxel_size": self.config.voxel_size,
            "total_points": n_points,
            "reachable_count": int(np.sum(reachable_mask)),
            "reachable_ratio": float(np.sum(reachable_mask) / n_points),
            "reachable_points": reachable_points,
            "unreachable_points": unreachable_points,
            "dexterity_scores": reachable_dexterity,           # 绝对灵巧性分数
            "normalized_dexterity": normalized_dexterity,       # 归一化灵巧性 (0-1)
            "num_orientations_tested": n_orientations,
            "computation_time": total_time,
            "test_orientations": test_orientations,
        }

        self.results[arm_name] = result
        return result

    def analyze_all(self) -> dict[str, dict]:
        """分析所有配置的机械臂"""
        for arm_cfg in self.config.arms:
            print(f"\n{'='*60}")
            print(f"分析机械臂: {arm_cfg.name}")
            print('='*60)
            self.analyze_arm(arm_cfg.name)

        return self.results

    def save_results(self):
        """保存所有结果"""
        for arm_name, result in self.results.items():
            prefix = os.path.join(self.config.output_dir, arm_name)

            # 保存可达点
            reachable = result["reachable_points"]
            if len(reachable) > 0:
                np.save(f"{prefix}_reachable.npy", reachable)
                pd.DataFrame(reachable, columns=["x", "y", "z"]).to_csv(
                    f"{prefix}_reachable.csv", index=False
                )

            # 保存灵巧性数据
            if "dexterity_scores" in result and len(result["dexterity_scores"]) > 0:
                dexterity = result["dexterity_scores"]
                normalized = result["normalized_dexterity"]

                # 保存灵巧性分数
                np.save(f"{prefix}_dexterity.npy", dexterity)

                # 保存带灵巧性的完整数据 (x, y, z, dexterity, normalized_dexterity)
                full_data = np.column_stack([reachable, dexterity, normalized])
                np.save(f"{prefix}_reachable_with_dexterity.npy", full_data)
                pd.DataFrame(
                    full_data,
                    columns=["x", "y", "z", "dexterity", "normalized_dexterity"]
                ).to_csv(f"{prefix}_reachable_with_dexterity.csv", index=False)

            # 可选：保存不可达点
            if self.config.save_failed_points:
                unreachable = result["unreachable_points"]
                if len(unreachable) > 0:
                    np.save(f"{prefix}_unreachable.npy", unreachable)

            # 保存统计信息
            stats = {
                "arm_name": arm_name,
                "bbox": result["bbox"],
                "voxel_size": result["voxel_size"],
                "total_points": result["total_points"],
                "reachable_count": result["reachable_count"],
                "reachable_ratio": result["reachable_ratio"],
                "computation_time_seconds": result["computation_time"],
                "num_orientations_tested": result.get("num_orientations_tested", 1),
                "avg_dexterity": float(np.mean(result.get("dexterity_scores", [0]))),
                "max_dexterity": int(np.max(result.get("dexterity_scores", [0]))),
            }

            import json
            with open(f"{prefix}_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            print(f"[SAVE] {arm_name} 结果已保存到: {prefix}_*")

    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("可达性与灵巧性分析摘要")
        print("="*60)

        for arm_name, result in self.results.items():
            print(f"\n机械臂: {arm_name}")
            print(f"  总点数: {result['total_points']}")
            print(f"  可达点数: {result['reachable_count']}")
            print(f"  可达率: {100*result['reachable_ratio']:.2f}%")
            print(f"  计算时间: {result['computation_time']:.2f}s")

            # 灵巧性统计
            if "dexterity_scores" in result and len(result["dexterity_scores"]) > 0:
                dex = result["dexterity_scores"]
                n_ori = result["num_orientations_tested"]
                print(f"  测试姿态数: {n_ori}")
                print(f"  灵巧性统计:")
                print(f"    - 平均: {np.mean(dex):.2f}/{n_ori} ({100*np.mean(dex)/n_ori:.1f}%)")
                print(f"    - 最大: {np.max(dex)}/{n_ori}")
                print(f"    - 最小: {np.min(dex)}/{n_ori}")
                print(f"    - 高灵巧性点 (>50%): {np.sum(dex > n_ori/2)} ({100*np.sum(dex > n_ori/2)/len(dex):.1f}%)")


# ===================================================================
# 示例用法
# ===================================================================

def create_config_from_urdf(
    urdf_path: str,
    arm_name: str = "robot_arm",
    ee_link: str = "",
    base_link: str = "",
    bbox: dict = None,
    voxel_size: float = 0.05,
    num_dexterity_orientations: int = 12,
) -> ReachabilityConfig:
    """
    从 URDF 文件创建可达性分析配置（推荐使用）

    Args:
        urdf_path: URDF 文件路径
        arm_name: 机械臂名称
        ee_link: 末端执行器链接名（可选，自动检测）
        base_link: 基座链接名（可选，自动检测）
        bbox: 工作空间范围，如 {"x": (-1, 1), "y": (-1, 1), "z": (0, 1.5)}
        voxel_size: 体素大小
        num_dexterity_orientations: 灵巧性测试姿态数

    Returns:
        ReachabilityConfig 配置对象

    示例:
        config = create_config_from_urdf(
            urdf_path="/path/to/robot.urdf",
            ee_link="tool0",
            bbox={"x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (0, 1.2)},
        )
    """
    if bbox is None:
        bbox = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.5)}

    arm = ArmConfig(
        name=arm_name,
        urdf_path=urdf_path,        # 直接使用 URDF，自动转换为 cuRobo 配置
        ee_link=ee_link,
        base_link=base_link,
        bbox=bbox,
        self_collision_check=True,   # 启用自碰撞检测
        self_collision_opt=True,
    )

    config = ReachabilityConfig(
        arms=[arm],
        voxel_size=voxel_size,
        compute_dexterity=True,
        num_dexterity_orientations=num_dexterity_orientations,
        batch_size=1024,
    )

    return config


def create_example_config() -> ReachabilityConfig:
    """创建一个示例配置（使用 cuRobo 内置的 Franka）"""

    # 完整工作空间范围
    full_bbox = {"x": (-0.9, 0.9), "y": (-0.9, 0.9), "z": (0.0, 1.3)}

    # 单臂配置（使用 cuRobo 内置的 Franka Panda 配置）
    single_arm = ArmConfig(
        name="franka_arm",
        ee_link="panda_hand",
        robot_config_file="franka.yml",  # cuRobo 内置配置
        bbox=full_bbox,
        self_collision_check=True,
        self_collision_opt=True,
    )

    # 全局配置
    config = ReachabilityConfig(
        arms=[single_arm],
        voxel_size=0.05,
        orientation_mode=OrientationMode.FIXED,
        fixed_orientations=[
            np.array([1.0, 0.0, 0.0, 0.0]),
        ],
        batch_size=1024,
        output_dir="./reachability_output",
    )

    return config


def create_dual_arm_config() -> ReachabilityConfig:
    """创建双臂配置（每个臂各自计算完整可达空间）"""

    # 完整工作空间范围
    full_bbox = {"x": (-0.9, 0.9), "y": (-0.9, 0.9), "z": (0.0, 1.3)}

    # 左臂（完整工作空间）
    left_arm = ArmConfig(
        name="left_arm",
        ee_link="panda_hand",
        robot_config_file="franka.yml",
        bbox=full_bbox,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_seeds=32,
    )

    # 右臂（完整工作空间）
    right_arm = ArmConfig(
        name="right_arm",
        ee_link="panda_hand",
        robot_config_file="franka.yml",
        bbox=full_bbox,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_seeds=32,
    )

    config = ReachabilityConfig(
        arms=[left_arm, right_arm],
        voxel_size=0.05,
        orientation_mode=OrientationMode.FIXED,
        fixed_orientations=[
            np.array([1.0, 0.0, 0.0, 0.0]),
        ],
        batch_size=1024,
        output_dir="./reachability_output",
    )

    return config


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="通用机械臂可达性分析框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认 URDF 文件
  python reachability_analysis.py

  # 指定 URDF 文件路径
  python reachability_analysis.py --urdf /path/to/robot.urdf

  # 指定末端执行器和基座链接
  python reachability_analysis.py --urdf robot.urdf --ee-link tool0 --base-link base_link

  # 自定义工作空间范围和体素大小
  python reachability_analysis.py --urdf robot.urdf --voxel-size 0.03 --bbox -1,1,-1,1,0,1.5

  # 设置灵巧性测试姿态数量
  python reachability_analysis.py --urdf robot.urdf --num-orientations 24
        """
    )

    parser.add_argument(
        "--urdf", "-u",
        type=str,
        default="./wheel_robot.urdf",
        help="URDF 文件路径 (默认: ./wheel_robot.urdf)"
    )
    parser.add_argument(
        "--ee-link", "-e",
        type=str,
        default="",
        help="末端执行器链接名 (默认: 自动检测)"
    )
    parser.add_argument(
        "--base-link", "-b",
        type=str,
        default="",
        help="基座链接名 (默认: 自动检测)"
    )
    parser.add_argument(
        "--voxel-size", "-v",
        type=float,
        default=0.05,
        help="体素大小 (米) (默认: 0.05)"
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="工作空间边界: xmin,xmax,ymin,ymax,zmin,zmax (默认: 自动估计)"
    )
    parser.add_argument(
        "--num-orientations", "-n",
        type=int,
        default=12,
        help="灵巧性测试姿态数量 (默认: 12)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./reachability_output",
        help="输出目录 (默认: ./reachability_output)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="robot_arm",
        help="机械臂名称 (默认: robot_arm)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="GPU batch 大小 (默认: 1024)"
    )
    parser.add_argument(
        "--no-self-collision",
        action="store_true",
        help="禁用自碰撞检测"
    )

    args = parser.parse_args()

    print("="*60)
    print("通用机械臂可达性分析框架")
    print("="*60)

    # 解析工作空间边界
    bbox = None
    if args.bbox:
        try:
            values = [float(x) for x in args.bbox.split(",")]
            if len(values) == 6:
                bbox = {
                    "x": (values[0], values[1]),
                    "y": (values[2], values[3]),
                    "z": (values[4], values[5])
                }
            else:
                print("[WARN] bbox 格式错误，应为 xmin,xmax,ymin,ymax,zmin,zmax，将自动估计")
        except ValueError:
            print("[WARN] bbox 解析失败，将自动估计")

    # 检查 URDF 文件是否存在
    from pathlib import Path
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"[ERROR] URDF 文件不存在: {args.urdf}")
        print("[INFO] 请指定正确的 URDF 文件路径，例如:")
        print("       python reachability_analysis.py --urdf /path/to/your/robot.urdf")
        return

    print(f"[INFO] URDF 文件: {urdf_path.absolute()}")
    print(f"[INFO] 末端链接: {args.ee_link or '自动检测'}")
    print(f"[INFO] 基座链接: {args.base_link or '自动检测'}")
    print(f"[INFO] 体素大小: {args.voxel_size}m")
    print(f"[INFO] 灵巧性测试姿态数: {args.num_orientations}")
    print(f"[INFO] 输出目录: {args.output_dir}")

    # 创建配置
    config = create_config_from_urdf(
        urdf_path=str(urdf_path.absolute()),
        arm_name=args.name,
        ee_link=args.ee_link,
        base_link=args.base_link,
        bbox=bbox,
        voxel_size=args.voxel_size,
        num_dexterity_orientations=args.num_orientations,
    )

    # 更新配置
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size

    # 更新自碰撞设置
    if args.no_self_collision:
        for arm in config.arms:
            arm.self_collision_check = False
            arm.self_collision_opt = False

    # 创建分析器
    analyzer = ReachabilityAnalyzer(config)

    # 运行分析
    analyzer.analyze_all()

    # 保存结果
    analyzer.save_results()

    # 打印摘要
    analyzer.print_summary()


if __name__ == "__main__":
    main()
