"""
动态可达性分析模块

支持在指定姿态下计算可达空间，考虑机器人其他部分（躯体、另一臂等）的碰撞。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import math

from .config import (
    ReachabilityConfig, DynamicReachabilityConfig,
    IKConfig, VoxelConfig, OrientationConfig
)
from .urdf_parser import URDFParser, RobotConfig, ArmChain


@dataclass
class CollisionSphere:
    """碰撞检测用球体"""
    position: np.ndarray  # [3] 世界坐标系位置
    radius: float
    link_name: str


@dataclass
class CollisionMesh:
    """碰撞检测用网格"""
    vertices: np.ndarray  # [N, 3] 世界坐标系顶点
    faces: np.ndarray     # [M, 3] 面索引
    link_name: str


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY角转旋转矩阵"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """轴角转旋转矩阵"""
    axis = axis.astype(np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


class WorldConfigBuilder:
    """
    构建cuRobo WorldConfig的工具类

    将URDF中的碰撞几何转换为cuRobo可用的障碍物配置。
    """

    def __init__(self, robot_config: RobotConfig):
        """
        参数:
            robot_config: 从URDFParser解析的机器人配置
        """
        self.robot_config = robot_config
        self._joint_by_child = {j.child_link: j for j in robot_config.joints}
        self._joint_by_parent = {}
        for j in robot_config.joints:
            self._joint_by_parent.setdefault(j.parent_link, []).append(j)

    def get_kinematic_chain_links(self, arm_chain: ArmChain) -> Set[str]:
        """
        获取运动链上的所有link名称

        参数:
            arm_chain: 机械臂链配置

        返回:
            运动链上所有link的名称集合
        """
        chain_links = set(arm_chain.links)
        return chain_links

    def compute_link_transforms(
        self,
        joint_positions: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        计算所有link相对于世界坐标系的4x4变换矩阵

        参数:
            joint_positions: 关节名到位置的映射

        返回:
            link名到4x4变换矩阵的映射
        """
        # 找到根link（不是任何关节的child）
        link_names = {l.name for l in self.robot_config.links}
        child_links = {j.child_link for j in self.robot_config.joints}
        roots = list(link_names - child_links)

        # 优先处理world-like链接
        world_like = {"world", "map", "odom"}
        roots.sort(key=lambda x: (0 if x.lower() in world_like else 1, x))

        # BFS计算所有变换
        T = {r: np.eye(4, dtype=np.float64) for r in roots}
        queue = list(roots)
        visited = set(roots)

        while queue:
            parent = queue.pop(0)
            for j in self._joint_by_parent.get(parent, []):
                child = j.child_link
                if child in visited:
                    continue

                # 关节origin变换
                T_origin = np.eye(4, dtype=np.float64)
                T_origin[:3, 3] = np.array(j.origin_xyz, dtype=np.float64)
                T_origin[:3, :3] = _rpy_to_matrix(
                    j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2]
                )

                # 关节运动变换
                q = float(joint_positions.get(j.name, 0.0))
                T_motion = np.eye(4, dtype=np.float64)
                if j.type in ("revolute", "continuous"):
                    R = _axis_angle_to_matrix(np.array(j.axis, dtype=np.float64), q)
                    T_motion[:3, :3] = R
                elif j.type == "prismatic":
                    axis = np.array(j.axis, dtype=np.float64)
                    n = np.linalg.norm(axis)
                    if n > 1e-12:
                        axis = axis / n
                    T_motion[:3, 3] = axis * q

                T_child = T[parent] @ T_origin @ T_motion
                T[child] = T_child
                visited.add(child)
                queue.append(child)

        return T

    def create_collision_spheres(
        self,
        exclude_links: Set[str],
        joint_positions: Dict[str, float],
        sphere_radius: float = 0.05
    ) -> List[CollisionSphere]:
        """
        为非排除link创建碰撞球体

        参数:
            exclude_links: 要排除的link集合（通常是目标臂的运动链）
            joint_positions: 关节位置
            sphere_radius: 球体半径

        返回:
            碰撞球体列表
        """
        transforms = self.compute_link_transforms(joint_positions)
        spheres = []

        for link in self.robot_config.links:
            if link.name in exclude_links:
                continue
            if link.name not in transforms:
                continue

            T = transforms[link.name]
            position = T[:3, 3].copy()

            spheres.append(CollisionSphere(
                position=position,
                radius=sphere_radius,
                link_name=link.name
            ))

        return spheres

    def build_world_config(
        self,
        exclude_links: Set[str],
        joint_positions: Dict[str, float],
        collision_buffer: float = 0.02,
        use_simplified: bool = True,
        sphere_radius: float = 0.05
    ) -> Optional[Any]:
        """
        构建cuRobo WorldConfig

        参数:
            exclude_links: 要排除的link集合
            joint_positions: 关节位置
            collision_buffer: 碰撞安全距离
            use_simplified: 使用简化球体模型
            sphere_radius: 球体半径

        返回:
            cuRobo WorldConfig对象，如果cuRobo不可用则返回None
        """
        try:
            from curobo.geom.types import WorldConfig, Sphere
        except ImportError:
            print("[警告] cuRobo不可用，无法创建WorldConfig")
            return None

        spheres = self.create_collision_spheres(
            exclude_links, joint_positions, sphere_radius
        )

        if not spheres:
            print("[信息] 没有需要添加的碰撞障碍物")
            return None

        # 构建cuRobo球体列表
        curobo_spheres = []
        for s in spheres:
            curobo_spheres.append(Sphere(
                name=f"obstacle_{s.link_name}",
                pose=[s.position[0], s.position[1], s.position[2], 1, 0, 0, 0],
                radius=s.radius + collision_buffer
            ))

        world_config = WorldConfig(sphere=curobo_spheres)
        print(f"[WorldConfig] 已创建，包含 {len(curobo_spheres)} 个碰撞球体")
        return world_config


class DynamicReachabilityController:
    """
    动态可达性分析控制器

    支持在指定姿态下计算可达空间，考虑机器人其他部分的碰撞。
    """

    def __init__(
        self,
        urdf_path: str,
        reachability_config: ReachabilityConfig,
        dynamic_config: Optional[DynamicReachabilityConfig] = None
    ):
        """
        参数:
            urdf_path: URDF文件路径
            reachability_config: 可达性分析基础配置
            dynamic_config: 动态可达性配置
        """
        self.urdf_path = urdf_path
        self.config = reachability_config
        self.dynamic_config = dynamic_config or DynamicReachabilityConfig()

        # 解析URDF
        self._parser = URDFParser(urdf_path)
        self._robot_config = self._parser.parse()

        # 检测所有臂链
        self._arm_chains = self._parser.detect_all_arm_chains()
        print(f"[动态可达性] 检测到 {len(self._arm_chains)} 个机械臂链:")
        for name, chain in self._arm_chains.items():
            print(f"  - {name}: {chain.base_link} -> {chain.ee_link}")

        # WorldConfig构建器
        self._world_builder = WorldConfigBuilder(self._robot_config)

        # 当前关节状态
        self._current_joint_positions: Dict[str, float] = {}

        # 缓存的IK求解器（按目标臂名）
        self._ik_solvers: Dict[str, Any] = {}

    def set_joint_positions(self, joint_positions: Dict[str, float]) -> None:
        """
        设置当前关节位置

        参数:
            joint_positions: 关节名到位置的映射
        """
        self._current_joint_positions.update(joint_positions)

    def get_arm_names(self) -> List[str]:
        """获取所有检测到的臂名称"""
        return list(self._arm_chains.keys())

    def compute_reachability(
        self,
        target_arm: str,
        joint_positions: Optional[Dict[str, float]] = None
    ):
        """
        计算指定臂的可达空间（考虑环境碰撞）

        参数:
            target_arm: 目标臂名称
            joint_positions: 关节位置（可选，默认使用当前存储的位置）

        返回:
            ReachabilityResult对象
        """
        if target_arm not in self._arm_chains:
            raise ValueError(f"未知的臂名称: {target_arm}，可用: {list(self._arm_chains.keys())}")

        # 合并关节位置
        all_positions = self._current_joint_positions.copy()
        if joint_positions:
            all_positions.update(joint_positions)

        target_chain = self._arm_chains[target_arm]

        # 获取目标臂运动链上的link
        chain_links = self._world_builder.get_kinematic_chain_links(target_chain)

        # 添加排除的link
        exclude_links = chain_links.copy()
        for link_name in self.dynamic_config.exclude_links:
            exclude_links.add(link_name)

        print(f"[动态可达性] 目标臂: {target_arm}")
        print(f"[动态可达性] 运动链link数: {len(chain_links)}")
        print(f"[动态可达性] 排除link数: {len(exclude_links)}")

        # 构建WorldConfig
        world_config = None
        if self.dynamic_config.environment_collision_check:
            world_config = self._world_builder.build_world_config(
                exclude_links=exclude_links,
                joint_positions=all_positions,
                collision_buffer=self.dynamic_config.collision_buffer,
                use_simplified=self.dynamic_config.use_simplified_collision,
                sphere_radius=self.dynamic_config.simplified_sphere_radius
            )

        # 创建针对目标臂的配置
        arm_config = self.config.copy()
        arm_config.base_link = target_chain.base_link
        arm_config.ee_link = target_chain.ee_link

        # 执行可达性分析
        from .reachability import ReachabilityAnalyzer

        analyzer = ReachabilityAnalyzer(
            urdf_path=self.urdf_path,
            config=arm_config,
            world_config=world_config
        )

        result = analyzer.analyze()
        return result

    def compute_both_arms(
        self,
        fixed_arm: str,
        fixed_positions: List[float],
        compute_arm: str
    ):
        """
        固定一个臂，计算另一个臂的可达空间

        参数:
            fixed_arm: 固定臂名称
            fixed_positions: 固定臂的关节位置列表
            compute_arm: 要计算可达空间的臂名称

        返回:
            ReachabilityResult对象
        """
        if fixed_arm not in self._arm_chains:
            raise ValueError(f"未知的固定臂: {fixed_arm}")
        if compute_arm not in self._arm_chains:
            raise ValueError(f"未知的计算臂: {compute_arm}")

        # 设置固定臂关节位置
        fixed_chain = self._arm_chains[fixed_arm]
        joint_names = [j.name for j in self._robot_config.active_joints
                      if j.name in [jn for jn in fixed_chain.joints]]

        # 从运动链获取关节名
        fixed_joints = {}
        for i, jname in enumerate(fixed_chain.joints):
            if i < len(fixed_positions):
                fixed_joints[jname] = fixed_positions[i]

        self.set_joint_positions(fixed_joints)

        # 计算目标臂可达空间
        return self.compute_reachability(compute_arm)
