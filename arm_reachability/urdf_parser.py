"""
URDF 解析器和 cuRobo 配置生成器

本模块负责解析 URDF 文件并为任意机械臂生成 cuRobo 兼容的配置。
"""

import os
import yaml
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class JointInfo:
    """机器人关节信息"""
    name: str                      # 关节名称
    type: str                      # 关节类型
    parent_link: str               # 父链接名称
    child_link: str                # 子链接名称
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])  # 关节轴
    lower_limit: float = -np.pi    # 下限位
    upper_limit: float = np.pi     # 上限位
    velocity_limit: float = 2.0    # 速度限制
    effort_limit: float = 100.0    # 力矩限制
    origin_xyz: List[float] = field(default_factory=lambda: [0, 0, 0])  # 原点位置
    origin_rpy: List[float] = field(default_factory=lambda: [0, 0, 0])  # 原点姿态


@dataclass
class LinkInfo:
    """机器人链接信息"""
    name: str                                      # 链接名称
    visual_mesh: Optional[str] = None              # 可视化网格路径
    collision_mesh: Optional[str] = None           # 碰撞网格路径
    visual_origin: Optional[List[float]] = None    # 可视化原点
    collision_origin: Optional[List[float]] = None # 碰撞原点
    visual_scale: Optional[List[float]] = None     # 可视化缩放
    collision_scale: Optional[List[float]] = None  # 碰撞缩放


@dataclass
class RobotConfig:
    """从 URDF 提取的机器人配置"""
    name: str                      # 机器人名称
    joints: List[JointInfo]        # 关节列表
    links: List[LinkInfo]          # 链接列表
    base_link: str                 # 基座链接名称
    ee_link: str                   # 末端执行器链接名称
    urdf_path: str                 # URDF 文件路径
    urdf_dir: str                  # URDF 文件目录

    @property
    def active_joints(self) -> List[JointInfo]:
        """获取活动（非固定）关节列表"""
        return [j for j in self.joints if j.type in ['revolute', 'prismatic', 'continuous']]

    @property
    def num_dof(self) -> int:
        """获取自由度数量"""
        return len(self.active_joints)


class URDFParser:
    """URDF 文件解析器"""

    # 用于识别末端执行器链接的关键词
    EE_KEYWORDS = [
        'end_effector', 'ee', 'tool', 'gripper', 'hand',
        'tcp', 'flange', 'wrist', 'link_6', 'link_7', 'link6', 'link7'
    ]

    # 用于识别基座链接的关键词（优先级从高到低）
    BASE_KEYWORDS = ['base_link', 'base', 'root', 'link_0', 'link0', 'world']

    # 通常作为外部世界坐标系的链接
    WORLD_LIKE = {'world', 'map', 'odom'}

    def __init__(self, urdf_path: str):
        """
        初始化 URDF 解析器

        参数:
            urdf_path: URDF 文件路径
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)
        self.tree = None
        self.root = None

        self._parse_urdf()

    def _parse_urdf(self):
        """解析 URDF 文件"""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF 文件不存在: {self.urdf_path}")

        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

    def get_robot_name(self) -> str:
        """从 URDF 获取机器人名称"""
        return self.root.get('name', 'robot')

    def parse_joints(self) -> List[JointInfo]:
        """从 URDF 解析所有关节"""
        joints = []

        for joint_elem in self.root.findall('joint'):
            joint = self._parse_joint_element(joint_elem)
            joints.append(joint)

        return joints

    def _parse_joint_element(self, joint_elem: ET.Element) -> JointInfo:
        """解析单个关节元素"""
        name = joint_elem.get('name', '')
        joint_type = joint_elem.get('type', 'fixed')

        # 获取父子链接
        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')
        parent_link = parent_elem.get('link', '') if parent_elem is not None else ''
        child_link = child_elem.get('link', '') if child_elem is not None else ''

        # 获取关节轴
        axis = [0, 0, 1]
        axis_elem = joint_elem.find('axis')
        if axis_elem is not None:
            axis_str = axis_elem.get('xyz', '0 0 1')
            axis = [float(x) for x in axis_str.split()]

        # 获取限位
        lower = -np.pi
        upper = np.pi
        velocity = 2.0
        effort = 100.0

        limit_elem = joint_elem.find('limit')
        if limit_elem is not None:
            lower = float(limit_elem.get('lower', str(lower)))
            upper = float(limit_elem.get('upper', str(upper)))
            velocity = float(limit_elem.get('velocity', str(velocity)))
            effort = float(limit_elem.get('effort', str(effort)))

        # 对于连续关节，设置较大的限位
        if joint_type == 'continuous':
            lower = -2 * np.pi
            upper = 2 * np.pi

        # 获取原点
        origin_xyz = [0, 0, 0]
        origin_rpy = [0, 0, 0]
        origin_elem = joint_elem.find('origin')
        if origin_elem is not None:
            xyz_str = origin_elem.get('xyz', '0 0 0')
            origin_xyz = [float(x) for x in xyz_str.split()]
            rpy_str = origin_elem.get('rpy', '0 0 0')
            origin_rpy = [float(x) for x in rpy_str.split()]

        return JointInfo(
            name=name,
            type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            axis=axis,
            lower_limit=lower,
            upper_limit=upper,
            velocity_limit=velocity,
            effort_limit=effort,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy
        )

    def parse_links(self) -> List[LinkInfo]:
        """从 URDF 解析所有链接"""
        links = []

        for link_elem in self.root.findall('link'):
            link = self._parse_link_element(link_elem)
            links.append(link)

        return links

    def _parse_link_element(self, link_elem: ET.Element) -> LinkInfo:
        """解析单个链接元素"""
        name = link_elem.get('name', '')

        visual_mesh = None
        collision_mesh = None
        visual_origin = None
        collision_origin = None
        visual_scale = None
        collision_scale = None

        # 解析可视化部分
        visual_elem = link_elem.find('visual')
        if visual_elem is not None:
            geometry = visual_elem.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    visual_mesh = mesh.get('filename', '')
                    scale_str = mesh.get('scale')
                    if scale_str:
                        try:
                            visual_scale = [float(x) for x in scale_str.split()]
                        except Exception:
                            visual_scale = None

            origin = visual_elem.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                visual_origin = [float(x) for x in xyz.split()] + [float(x) for x in rpy.split()]

        # 解析碰撞部分
        collision_elem = link_elem.find('collision')
        if collision_elem is not None:
            geometry = collision_elem.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    collision_mesh = mesh.get('filename', '')
                    scale_str = mesh.get('scale')
                    if scale_str:
                        try:
                            collision_scale = [float(x) for x in scale_str.split()]
                        except Exception:
                            collision_scale = None

            origin = collision_elem.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                collision_origin = [float(x) for x in xyz.split()] + [float(x) for x in rpy.split()]

        return LinkInfo(
            name=name,
            visual_mesh=visual_mesh,
            collision_mesh=collision_mesh,
            visual_origin=visual_origin,
            collision_origin=collision_origin,
            visual_scale=visual_scale,
            collision_scale=collision_scale
        )

    def detect_base_link(self, joints: List[JointInfo], links: List[LinkInfo]) -> str:
        """自动检测基座链接"""
        link_names = {link.name for link in links}

        # 找出仅作为父链接的链接（不是任何关节的子链接）
        child_links = {j.child_link for j in joints}
        root_links = link_names - child_links

        # 特殊情况：许多 URDF 有一个虚拟的 'world' 根节点和一个固定关节连接到实际基座
        # 例如 (AR5): world --(fixed)--> <robot_base>
        # 此时我们希望 <robot_base> 作为 base_link，使分析以机器人基座为中心
        if root_links:
            world_roots = [l for l in root_links if l.lower() in self.WORLD_LIKE]
            if world_roots:
                # 优先选择从 world 类型根节点通过固定关节连接的子链接
                candidates = []
                for j in joints:
                    if j.parent_link.lower() in self.WORLD_LIKE and j.type == 'fixed':
                        if j.child_link in link_names:
                            candidates.append(j.child_link)

                if candidates:
                    # 优先选择看起来像 base/base_link 的名称
                    for cand in candidates:
                        cand_lower = cand.lower()
                        if 'base_link' in cand_lower or cand_lower.endswith('base') or 'base' in cand_lower:
                            return cand
                    # 否则确定性地选择第一个候选
                    return sorted(candidates)[0]

        # 检查根链接是否包含基座关键词
        for link in root_links:
            link_lower = link.lower()
            for keyword in self.BASE_KEYWORDS:
                if keyword in link_lower:
                    return link

        # 返回第一个根链接（如果有）
        if root_links:
            return list(root_links)[0]

        # 回退：第一个链接
        if links:
            return links[0].name

        return "base_link"

    def detect_ee_link(self, joints: List[JointInfo], links: List[LinkInfo]) -> str:
        """自动检测末端执行器链接"""
        link_names = {link.name for link in links}

        # 找出仅作为子链接的链接（不是任何关节的父链接）
        parent_links = {j.parent_link for j in joints}
        leaf_links = link_names - parent_links

        # 检查叶子链接是否包含末端执行器关键词
        for link in leaf_links:
            link_lower = link.lower()
            for keyword in self.EE_KEYWORDS:
                if keyword in link_lower:
                    return link

        # 检查所有链接是否包含末端执行器关键词
        for link in links:
            link_lower = link.name.lower()
            for keyword in self.EE_KEYWORDS:
                if keyword in link_lower:
                    return link.name

        # 查找最长运动链
        if joints:
            chain = self._find_longest_chain(joints, links)
            if chain:
                return chain[-1]

        # 回退：最后一个叶子链接
        if leaf_links:
            return list(leaf_links)[-1]

        return links[-1].name if links else "ee_link"

    def _find_longest_chain(self, joints: List[JointInfo], links: List[LinkInfo]) -> List[str]:
        """查找机器人中最长的运动链"""
        # 构建邻接图
        link_names = {link.name for link in links}
        child_to_parent = {}
        parent_to_children = {}

        for joint in joints:
            if joint.parent_link in link_names and joint.child_link in link_names:
                child_to_parent[joint.child_link] = joint.parent_link
                if joint.parent_link not in parent_to_children:
                    parent_to_children[joint.parent_link] = []
                parent_to_children[joint.parent_link].append(joint.child_link)

        # 查找叶子链接
        leaf_links = link_names - set(parent_to_children.keys())

        # 从任意叶子到根查找最长链
        longest_chain = []
        for leaf in leaf_links:
            chain = [leaf]
            current = leaf
            while current in child_to_parent:
                current = child_to_parent[current]
                chain.append(current)

            if len(chain) > len(longest_chain):
                longest_chain = chain

        return list(reversed(longest_chain))

    def get_kinematic_chain(self, base_link: str, ee_link: str,
                           joints: List[JointInfo]) -> List[JointInfo]:
        """获取从基座到末端执行器的运动链"""
        # 构建父子映射
        link_to_joint = {j.child_link: j for j in joints}

        chain = []
        current = ee_link

        while current != base_link:
            if current in link_to_joint:
                joint = link_to_joint[current]
                chain.append(joint)
                current = joint.parent_link
            else:
                break

        return list(reversed(chain))

    def parse(self, base_link: str = "", ee_link: str = "") -> RobotConfig:
        """
        解析 URDF 并返回机器人配置

        参数:
            base_link: 基座链接名（为空则自动检测）
            ee_link: 末端执行器链接名（为空则自动检测）

        返回:
            包含提取信息的 RobotConfig
        """
        joints = self.parse_joints()
        links = self.parse_links()

        if not base_link:
            base_link = self.detect_base_link(joints, links)

        if not ee_link:
            ee_link = self.detect_ee_link(joints, links)

        return RobotConfig(
            name=self.get_robot_name(),
            joints=joints,
            links=links,
            base_link=base_link,
            ee_link=ee_link,
            urdf_path=self.urdf_path,
            urdf_dir=self.urdf_dir
        )


class CuroboConfigGenerator:
    """从机器人配置生成 cuRobo 配置"""

    def __init__(self, robot_config: RobotConfig):
        """
        初始化配置生成器

        参数:
            robot_config: 解析后的机器人配置
        """
        self.robot = robot_config

    def generate(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成 cuRobo 配置

        参数:
            output_path: 可选的 YAML 配置保存路径

        返回:
            配置字典
        """
        # 获取运动链
        chain_joints = self.robot.active_joints

        # 生成关节名称
        joint_names = [j.name for j in chain_joints]

        # 生成默认配置（关节限位的中点）
        default_config = []
        for joint in chain_joints:
            mid = (joint.lower_limit + joint.upper_limit) / 2
            default_config.append(float(mid))

        # 生成收回配置（为安全起见与默认配置相同）
        retract_config = default_config.copy()

        # 生成构型空间配置
        cspace_config = {
            'joint_names': joint_names,
            'retract_config': retract_config,
            'null_space_weight': [1.0] * len(joint_names),
            'cspace_distance_weight': [1.0] * len(joint_names),
            'max_jerk': 500.0,
            'max_acceleration': 15.0,
        }

        # 生成运动学配置
        kinematics_config = {
            'usd_path': None,
            'usd_robot_root': None,
            'isaac_usd_path': None,
            'urdf_path': self.robot.urdf_path,
            'asset_root_path': self.robot.urdf_dir,
            'base_link': self.robot.base_link,
            'ee_link': self.robot.ee_link,
            'link_names': None,
            'lock_joints': {},
            'extra_links': {},
            'cspace': cspace_config,
        }

        # 生成自碰撞配置
        collision_spheres = self._generate_collision_spheres()
        collision_config = {
            'collision_spheres': collision_spheres,
            'buffer_distance': 0.005,
            'self_collision_buffer': {
                'default': 0.02,
            }
        }

        # 完整配置
        config = {
            'robot_cfg': {
                'kinematics': kinematics_config,
                'collision': collision_config,
            }
        }

        # 如果提供了路径则保存
        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config

    def _generate_collision_spheres(self) -> Dict[str, List[Dict]]:
        """为每个链接生成基本碰撞球"""
        spheres = {}

        for link in self.robot.links:
            # 为每个链接创建基本球体
            spheres[link.name] = [
                {'center': [0.0, 0.0, 0.0], 'radius': 0.05}
            ]

        return spheres

    def generate_ik_config(self) -> Dict[str, Any]:
        """为 cuRobo 生成 IK 求解器配置"""
        chain_joints = self.robot.active_joints
        joint_names = [j.name for j in chain_joints]

        # 构建位置和速度限位
        position_lower = []
        position_upper = []
        velocity = []

        for joint in chain_joints:
            position_lower.append(float(joint.lower_limit))
            position_upper.append(float(joint.upper_limit))
            velocity.append(float(joint.velocity_limit))

        config = {
            'ik_solver': {
                'position_threshold': 0.005,
                'rotation_threshold': 0.05,
                'num_seeds': 32,
                'num_graph_seeds': 12,
                'num_trajopt_seeds': 8,
            },
            'robot': {
                'urdf_path': self.robot.urdf_path,
                'base_link': self.robot.base_link,
                'ee_link': self.robot.ee_link,
                'joint_names': joint_names,
                'joint_limits': {
                    'position_lower': position_lower,
                    'position_upper': position_upper,
                    'velocity': velocity,
                }
            }
        }

        return config


def load_urdf(urdf_path: str, base_link: str = "", ee_link: str = "") -> Tuple[RobotConfig, Dict]:
    """
    加载 URDF 并生成 cuRobo 配置的便捷函数

    参数:
        urdf_path: URDF 文件路径
        base_link: 可选的基座链接名
        ee_link: 可选的末端执行器链接名

    返回:
        (RobotConfig, cuRobo 配置字典) 元组
    """
    parser = URDFParser(urdf_path)
    robot_config = parser.parse(base_link, ee_link)

    generator = CuroboConfigGenerator(robot_config)
    curobo_config = generator.generate()

    return robot_config, curobo_config
