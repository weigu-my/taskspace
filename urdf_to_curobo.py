#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
URDF 到 cuRobo 配置自动转换工具
===========================================================

功能：
1. 解析 URDF 文件，提取机器人运动学信息
2. 自动生成 cuRobo 格式的 YAML 配置文件
3. 配置自碰撞检测
4. 支持直接用于可达性分析

依赖：
- yourdfpy: pip install yourdfpy
- pyyaml: pip install pyyaml
"""

from __future__ import annotations

import os
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# 尝试导入 URDF 解析库
try:
    from yourdfpy import URDF
    YOURDFPY_AVAILABLE = True
except ImportError:
    YOURDFPY_AVAILABLE = False
    print("[WARN] yourdfpy 未安装，请运行: pip install yourdfpy")

try:
    import xml.etree.ElementTree as ET
except ImportError:
    pass


@dataclass
class URDFRobotConfig:
    """从 URDF 提取的机器人配置"""
    urdf_path: str
    robot_name: str = ""
    base_link: str = ""
    ee_link: str = ""

    # 关节信息
    joint_names: List[str] = field(default_factory=list)
    joint_types: List[str] = field(default_factory=list)
    joint_limits_lower: List[float] = field(default_factory=list)
    joint_limits_upper: List[float] = field(default_factory=list)
    joint_velocity_limits: List[float] = field(default_factory=list)
    joint_effort_limits: List[float] = field(default_factory=list)

    # 链接信息
    link_names: List[str] = field(default_factory=list)

    # 自碰撞配置
    self_collision_ignore: List[tuple] = field(default_factory=list)

    # 默认配置
    default_joint_positions: List[float] = field(default_factory=list)
    retract_config: List[float] = field(default_factory=list)


class URDFParser:
    """URDF 解析器"""

    def __init__(self, urdf_path: str, ee_link: str = None, base_link: str = None, arm_select: str = "auto"):
        """
        初始化 URDF 解析器

        Args:
            urdf_path: URDF 文件路径
            ee_link: 指定的末端执行器链接（可选）
            base_link: 指定的基座链接（可选）
            arm_select: 对于双臂机器人，选择 "left", "right", 或 "auto"
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")

        self.config = URDFRobotConfig(urdf_path=str(urdf_path))
        self.specified_ee_link = ee_link
        self.specified_base_link = base_link
        self.arm_select = arm_select.lower() if arm_select else "auto"
        self._parse()

    def _parse(self):
        """解析 URDF 文件"""
        if YOURDFPY_AVAILABLE:
            self._parse_with_yourdfpy()
        else:
            self._parse_with_xml()

    def _parse_with_yourdfpy(self):
        """使用 yourdfpy 解析"""
        robot = URDF.load(self.urdf_path)

        # 获取机器人名称 - yourdfpy 可能通过不同方式存储
        try:
            if hasattr(robot, 'name') and robot.name:
                self.config.robot_name = robot.name
            elif hasattr(robot, 'robot') and hasattr(robot.robot, 'name'):
                self.config.robot_name = robot.robot.name
            elif hasattr(robot, '_robot') and hasattr(robot._robot, 'attrib'):
                self.config.robot_name = robot._robot.attrib.get('name', self.urdf_path.stem)
            else:
                self.config.robot_name = self.urdf_path.stem
        except Exception:
            self.config.robot_name = self.urdf_path.stem

        # 获取所有链接
        self.config.link_names = list(robot.link_map.keys())

        # 找到根链接（没有父关节的链接）
        child_links = set()
        for joint in robot.joint_map.values():
            child_links.add(joint.child)

        root_link = None
        for link_name in self.config.link_names:
            if link_name not in child_links:
                root_link = link_name
                break

        # 设置末端链接（优先使用指定的，否则自动检测）
        if self.specified_ee_link:
            if self.specified_ee_link in self.config.link_names:
                self.config.ee_link = self.specified_ee_link
                print(f"[INFO] 使用指定的末端链接: {self.specified_ee_link}")
            else:
                print(f"[WARN] 指定的末端链接 '{self.specified_ee_link}' 不存在，将自动检测")
                self.config.ee_link = self._find_ee_link(robot)
        else:
            self.config.ee_link = self._find_ee_link(robot)

        # 找到从末端到根的运动链，并确定合适的基座链接
        chain_joints, chain_base_link = self._get_kinematic_chain_with_base(robot, root_link)

        # 设置基座链接（优先使用指定的，否则使用运动链的起点）
        if self.specified_base_link:
            if self.specified_base_link in self.config.link_names:
                self.config.base_link = self.specified_base_link
                print(f"[INFO] 使用指定的基座链接: {self.specified_base_link}")
            else:
                print(f"[WARN] 指定的基座链接 '{self.specified_base_link}' 不存在，将自动检测")
                self.config.base_link = chain_base_link if chain_base_link else root_link
        else:
            self.config.base_link = chain_base_link if chain_base_link else root_link

        if len(chain_joints) == 0:
            # 如果无法找到链，则使用所有活动关节
            print("[WARN] 无法确定运动链，使用所有活动关节")
            if not self.specified_base_link:
                self.config.base_link = root_link
            # 收集所有活动关节
            for joint_name, joint in robot.joint_map.items():
                if joint.type in ['revolute', 'prismatic', 'continuous']:
                    chain_joints.append(joint_name)

        # 按运动链顺序添加关节
        for joint_name in chain_joints:
            joint = robot.joint_map[joint_name]

            self.config.joint_names.append(joint_name)
            self.config.joint_types.append(joint.type)

            # 关节限位
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -3.14159
                upper = joint.limit.upper if joint.limit.upper is not None else 3.14159
                velocity = joint.limit.velocity if joint.limit.velocity is not None else 2.0
                effort = joint.limit.effort if joint.limit.effort is not None else 100.0
            else:
                lower, upper = -3.14159, 3.14159
                velocity, effort = 2.0, 100.0

            # continuous 关节没有限位
            if joint.type == 'continuous':
                lower, upper = -6.28318, 6.28318

            self.config.joint_limits_lower.append(lower)
            self.config.joint_limits_upper.append(upper)
            self.config.joint_velocity_limits.append(velocity)
            self.config.joint_effort_limits.append(effort)

        # 默认关节位置（中间位置）
        self.config.default_joint_positions = [
            (l + u) / 2 for l, u in zip(
                self.config.joint_limits_lower,
                self.config.joint_limits_upper
            )
        ]
        self.config.retract_config = self.config.default_joint_positions.copy()

        # 生成自碰撞忽略对（相邻链接）
        self._generate_collision_ignore_pairs(robot)

        print(f"[INFO] 解析完成: {self.config.robot_name}")
        print(f"[INFO] 基座链接: {self.config.base_link}")
        print(f"[INFO] 末端链接: {self.config.ee_link}")
        print(f"[INFO] 活动关节数: {len(self.config.joint_names)}")

    def _get_kinematic_chain_with_base(self, robot, root_link: str) -> tuple:
        """
        获取从末端到根的运动链，并找到合适的基座链接

        Args:
            robot: URDF 机器人对象
            root_link: 根链接名称

        Returns:
            (chain_joints, base_link): 运动链关节列表和基座链接名
        """
        chain_joints = []
        chain_links = []  # 记录链路中的所有链接

        # 构建从子链接到父链接和关节的映射
        child_to_parent = {}  # child_link -> (parent_link, joint_name, joint_type)
        for joint_name, joint in robot.joint_map.items():
            child_to_parent[joint.child] = (joint.parent, joint_name, joint.type)

        # 从 ee_link 回溯到根链接
        current_link = self.config.ee_link
        visited = set()
        first_active_joint_parent = None

        while current_link in child_to_parent:
            if current_link in visited:
                print(f"[WARN] 检测到循环，无法确定运动链")
                break
            visited.add(current_link)

            parent_link, joint_name, joint_type = child_to_parent[current_link]
            chain_links.append(current_link)

            # 只包含活动关节
            if joint_type in ['revolute', 'prismatic', 'continuous']:
                chain_joints.append(joint_name)
                # 记录第一个活动关节的父链接作为基座
                first_active_joint_parent = parent_link

            current_link = parent_link

        # 反转列表，使其从基座到末端
        chain_joints.reverse()

        # 基座链接是第一个活动关节的父链接
        base_link = first_active_joint_parent if first_active_joint_parent else root_link

        print(f"[INFO] 运动链关节数: {len(chain_joints)}")
        if len(chain_joints) > 0:
            print(f"[INFO] 运动链关节: {chain_joints[:5]}{'...' if len(chain_joints) > 5 else ''}")
            print(f"[INFO] 运动链基座: {base_link}")

        return chain_joints, base_link

    def _get_kinematic_chain_joints(self, robot) -> list:
        """
        获取从 base_link 到 ee_link 的运动链中的所有关节名称（按顺序）
        （保留此方法以保持兼容性）

        Returns:
            运动链中的关节名称列表（从基座到末端的顺序）
        """
        chain_joints, _ = self._get_kinematic_chain_with_base(robot, self.config.base_link)
        return chain_joints

    def _find_ee_link(self, robot) -> str:
        """找到末端执行器链接"""
        # 构建链接的子链接映射
        children = {link: [] for link in self.config.link_names}
        for joint in robot.joint_map.values():
            if joint.parent in children:
                children[joint.parent].append(joint.child)

        # 找到没有子链接的链接（叶子节点）
        leaf_links = [link for link, childs in children.items() if len(childs) == 0]

        # 计算每个叶子链接的运动链深度（活动关节数量）
        def get_chain_depth(link_name):
            """计算从基座到指定链接的活动关节数量"""
            depth = 0
            current = link_name
            child_to_parent = {}
            for joint_name, joint in robot.joint_map.items():
                child_to_parent[joint.child] = (joint.parent, joint.type)

            visited = set()
            while current in child_to_parent and current not in visited:
                visited.add(current)
                parent, joint_type = child_to_parent[current]
                if joint_type in ['revolute', 'prismatic', 'continuous']:
                    depth += 1
                current = parent
            return depth

        # 优先选择机械臂相关的末端链接
        # 关键词优先级：gripper/hand > flange/tool > ee/end/tcp
        arm_keywords_high = ['gripper', 'hand', 'finger', 'pad']
        arm_keywords_mid = ['flange', 'tool', 'end_effector']
        arm_keywords_low = ['ee', 'end', 'tcp', 'link7', 'link6']

        # 排除轮子和底盘相关的链接
        exclude_keywords = ['wheel', 'chassis', 'caster', 'base_footprint']

        # 根据手臂选择过滤
        arm_filter_keywords = []
        opposite_arm_keywords = []
        if self.arm_select in ['left', 'l']:
            arm_filter_keywords = ['left', '_l_', '_l/', '/l_', 'l_arm', 'larm', '_l.']
            opposite_arm_keywords = ['right', '_r_', '_r/', '/r_', 'r_arm', 'rarm', '_r.']
            print(f"[INFO] 选择左臂进行分析")
        elif self.arm_select in ['right', 'r']:
            arm_filter_keywords = ['right', '_r_', '_r/', '/r_', 'r_arm', 'rarm', '_r.']
            opposite_arm_keywords = ['left', '_l_', '_l/', '/l_', 'l_arm', 'larm', '_l.']
            print(f"[INFO] 选择右臂进行分析")

        # 筛选候选链接（排除底盘相关，根据手臂选择过滤）
        candidate_links = []
        for link in leaf_links:
            link_lower = link.lower()
            is_excluded = any(kw in link_lower for kw in exclude_keywords)

            # 如果指定了手臂，排除另一只手臂的链接
            if opposite_arm_keywords:
                is_opposite_arm = any(kw in link_lower for kw in opposite_arm_keywords)
                if is_opposite_arm:
                    is_excluded = True

            if not is_excluded:
                depth = get_chain_depth(link)
                # 如果指定了手臂，优先选择匹配的链接
                is_matching_arm = any(kw in link_lower for kw in arm_filter_keywords) if arm_filter_keywords else False
                candidate_links.append((link, depth, is_matching_arm))

        # 按深度和是否匹配手臂排序（优先匹配手臂，其次深度）
        candidate_links.sort(key=lambda x: (x[2], x[1]), reverse=True)

        # 在深度足够的候选中，按关键词优先级选择
        for link, depth, is_matching in candidate_links:
            if depth >= 5:  # 至少 5 个活动关节才可能是机械臂
                link_lower = link.lower()
                for keyword in arm_keywords_high:
                    if keyword in link_lower:
                        print(f"[INFO] 自动检测到末端链接: {link} (深度={depth}, 关键词={keyword})")
                        return link

        for link, depth, is_matching in candidate_links:
            if depth >= 5:
                link_lower = link.lower()
                for keyword in arm_keywords_mid:
                    if keyword in link_lower:
                        print(f"[INFO] 自动检测到末端链接: {link} (深度={depth}, 关键词={keyword})")
                        return link

        for link, depth, is_matching in candidate_links:
            if depth >= 5:
                link_lower = link.lower()
                for keyword in arm_keywords_low:
                    if keyword in link_lower:
                        print(f"[INFO] 自动检测到末端链接: {link} (深度={depth}, 关键词={keyword})")
                        return link

        # 如果没有匹配关键词，选择深度最大的候选
        if candidate_links:
            best_link, best_depth, _ = candidate_links[0]
            if best_depth >= 3:
                print(f"[INFO] 自动检测到末端链接: {best_link} (深度={best_depth})")
                return best_link

        # 最后备选：原始叶子节点中深度最大的
        all_with_depth = [(link, get_chain_depth(link)) for link in leaf_links]
        all_with_depth.sort(key=lambda x: x[1], reverse=True)
        if all_with_depth:
            best_link, best_depth = all_with_depth[0]
            print(f"[WARN] 使用备选末端链接: {best_link} (深度={best_depth})")
            return best_link

        return self.config.link_names[-1]

    def _generate_collision_ignore_pairs(self, robot):
        """生成自碰撞忽略对（相邻链接不检测碰撞）"""
        ignore_pairs = set()

        # 相邻链接（通过关节连接的）不检测碰撞
        for joint in robot.joint_map.values():
            pair = tuple(sorted([joint.parent, joint.child]))
            ignore_pairs.add(pair)

        # 也可以忽略基座与第一个链接的碰撞
        self.config.self_collision_ignore = list(ignore_pairs)

    def _parse_with_xml(self):
        """使用 xml 解析（备用方案）"""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        self.config.robot_name = root.get('name', self.urdf_path.stem)

        # 解析链接
        for link in root.findall('link'):
            self.config.link_names.append(link.get('name'))

        # 解析关节
        child_links = set()
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')

            child = joint.find('child')
            parent = joint.find('parent')
            if child is not None:
                child_links.add(child.get('link'))

            if joint_type in ['revolute', 'prismatic', 'continuous']:
                self.config.joint_names.append(joint_name)
                self.config.joint_types.append(joint_type)

                limit = joint.find('limit')
                if limit is not None:
                    lower = float(limit.get('lower', -3.14159))
                    upper = float(limit.get('upper', 3.14159))
                    velocity = float(limit.get('velocity', 2.0))
                    effort = float(limit.get('effort', 100.0))
                else:
                    lower, upper = -3.14159, 3.14159
                    velocity, effort = 2.0, 100.0

                if joint_type == 'continuous':
                    lower, upper = -6.28318, 6.28318

                self.config.joint_limits_lower.append(lower)
                self.config.joint_limits_upper.append(upper)
                self.config.joint_velocity_limits.append(velocity)
                self.config.joint_effort_limits.append(effort)

        # 找基座链接
        for link_name in self.config.link_names:
            if link_name not in child_links:
                self.config.base_link = link_name
                break

        # 默认末端链接
        self.config.ee_link = self.config.link_names[-1]

        # 默认关节位置
        self.config.default_joint_positions = [
            (l + u) / 2 for l, u in zip(
                self.config.joint_limits_lower,
                self.config.joint_limits_upper
            )
        ]
        self.config.retract_config = self.config.default_joint_positions.copy()


class CuroboConfigGenerator:
    """cuRobo 配置文件生成器"""

    def __init__(self, urdf_config: URDFRobotConfig):
        self.config = urdf_config

    def generate_robot_config(self, output_path: str = None) -> dict:
        """生成 cuRobo 机器人配置 YAML"""

        # 获取 URDF 的绝对路径
        urdf_abs_path = str(Path(self.config.urdf_path).absolute())
        asset_root = str(Path(self.config.urdf_path).parent.absolute())

        n_joints = len(self.config.joint_names)
        if n_joints == 0:
            raise ValueError("没有找到活动关节，请检查 URDF 文件")

        # cuRobo 期望的配置格式 - 注意结构
        # 这个格式直接传给 IKSolverConfig.load_from_robot_config()
        robot_cfg = {
            'robot_cfg': {
                'kinematics': {
                    'urdf_path': urdf_abs_path,
                    'asset_root_path': asset_root,
                    'base_link': self.config.base_link,
                    'ee_link': self.config.ee_link,
                    'cspace': {
                        'joint_names': self.config.joint_names,
                        'retract_config': self.config.retract_config,
                        'null_space_weight': [1.0] * n_joints,
                        'cspace_distance_weight': [1.0] * n_joints,
                        'max_jerk': 500.0,
                        'max_acceleration': 15.0,
                    },
                },
            }
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(robot_cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"[SAVE] 机器人配置已保存到: {output_path}")

        return robot_cfg

    def _generate_collision_spheres(self) -> dict:
        """为每个链接生成碰撞球体"""
        spheres = {}
        for link_name in self.config.link_names:
            # 默认在每个链接中心放置一个球体
            spheres[link_name] = [
                {'center': [0.0, 0.0, 0.0], 'radius': 0.05}
            ]
        return spheres

    def generate_world_config(self, output_path: str = None) -> dict:
        """生成 cuRobo 世界配置（无障碍物）"""

        # cuRobo 期望的世界配置格式 - 空世界
        # 当传入字典时，可以直接传 None 或空的障碍物配置
        world_cfg = {
            'world_model': {
                'cuboid': {},
                'mesh': {},
            }
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(world_cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"[SAVE] 世界配置已保存到: {output_path}")

        return world_cfg

    def generate_all_configs(self, output_dir: str) -> tuple[str, str]:
        """生成所有配置文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        robot_cfg_path = output_dir / f"{self.config.robot_name}_robot.yaml"
        world_cfg_path = output_dir / f"{self.config.robot_name}_world.yaml"

        self.generate_robot_config(robot_cfg_path)
        self.generate_world_config(world_cfg_path)

        return str(robot_cfg_path), str(world_cfg_path)


def urdf_to_curobo_config(
    urdf_path: str,
    ee_link: str = None,
    base_link: str = None,
    output_dir: str = "./curobo_configs",
    arm_select: str = "auto",
) -> tuple[str, str]:
    """
    将 URDF 文件转换为 cuRobo 配置

    Args:
        urdf_path: URDF 文件路径
        ee_link: 末端执行器链接名（可选，自动检测）
        base_link: 基座链接名（可选，自动检测）
        output_dir: 输出目录
        arm_select: 对于双臂机器人，选择 "left", "right", 或 "auto"

    Returns:
        (robot_config_path, world_config_path)
    """
    # 解析 URDF（传入用户指定的参数）
    parser = URDFParser(urdf_path, ee_link=ee_link, base_link=base_link, arm_select=arm_select)

    # 生成配置
    generator = CuroboConfigGenerator(parser.config)
    robot_cfg_path, world_cfg_path = generator.generate_all_configs(output_dir)

    return robot_cfg_path, world_cfg_path


def create_reachability_config_from_urdf(
    urdf_path: str,
    ee_link: str = None,
    base_link: str = None,
    arm_name: str = "robot_arm",
    bbox: dict = None,
    voxel_size: float = 0.05,
    num_dexterity_orientations: int = 12,
    output_dir: str = "./reachability_output",
):
    """
    从 URDF 创建完整的可达性分析配置

    Args:
        urdf_path: URDF 文件路径
        ee_link: 末端执行器链接名
        base_link: 基座链接名
        arm_name: 机械臂名称
        bbox: 工作空间范围，如 {"x": (-1, 1), "y": (-1, 1), "z": (0, 1.5)}
        voxel_size: 体素大小
        num_dexterity_orientations: 测试姿态数
        output_dir: 输出目录

    Returns:
        ReachabilityConfig 对象
    """
    # 导入可达性分析模块
    from reachability_analysis import ReachabilityConfig, ArmConfig, OrientationMode

    # 转换 URDF 到 cuRobo 配置
    config_dir = Path(output_dir) / "curobo_configs"
    robot_cfg_path, world_cfg_path = urdf_to_curobo_config(
        urdf_path,
        ee_link=ee_link,
        base_link=base_link,
        output_dir=str(config_dir),
    )

    # 解析 URDF 获取末端链接名
    parser = URDFParser(urdf_path)
    actual_ee_link = ee_link or parser.config.ee_link

    # 默认 bbox
    if bbox is None:
        bbox = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.5)}

    # 创建臂配置
    arm_config = ArmConfig(
        name=arm_name,
        ee_link=actual_ee_link,
        robot_config_file=robot_cfg_path,
        world_config_file=world_cfg_path,
        bbox=bbox,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_seeds=32,
    )

    # 创建全局配置
    config = ReachabilityConfig(
        arms=[arm_config],
        voxel_size=voxel_size,
        orientation_mode=OrientationMode.FIXED,
        compute_dexterity=True,
        num_dexterity_orientations=num_dexterity_orientations,
        batch_size=1024,
        output_dir=output_dir,
    )

    return config


# ===================================================================
# 命令行接口
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="URDF 到 cuRobo 配置转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换 URDF 为 cuRobo 配置
  python urdf_to_curobo.py robot.urdf -o ./configs

  # 指定末端链接
  python urdf_to_curobo.py robot.urdf --ee-link tool0 -o ./configs

  # 直接运行可达性分析
  python urdf_to_curobo.py robot.urdf --analyze --voxel-size 0.05
        """
    )

    parser.add_argument("urdf_path", type=str, help="URDF 文件路径")
    parser.add_argument("-o", "--output", type=str, default="./curobo_configs",
                        help="输出目录")
    parser.add_argument("--ee-link", type=str, default=None,
                        help="末端执行器链接名")
    parser.add_argument("--base-link", type=str, default=None,
                        help="基座链接名")
    parser.add_argument("--analyze", action="store_true",
                        help="转换后直接运行可达性分析")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="可达性分析的体素大小")
    parser.add_argument("--num-orientations", type=int, default=12,
                        help="灵巧性测试的姿态数量")

    args = parser.parse_args()

    # 转换 URDF
    robot_cfg, world_cfg = urdf_to_curobo_config(
        args.urdf_path,
        ee_link=args.ee_link,
        base_link=args.base_link,
        output_dir=args.output,
    )

    print(f"\n[完成] cuRobo 配置文件:")
    print(f"  - 机器人配置: {robot_cfg}")
    print(f"  - 世界配置: {world_cfg}")

    # 如果需要运行可达性分析
    if args.analyze:
        print("\n[INFO] 开始可达性分析...")

        config = create_reachability_config_from_urdf(
            args.urdf_path,
            ee_link=args.ee_link,
            base_link=args.base_link,
            voxel_size=args.voxel_size,
            num_dexterity_orientations=args.num_orientations,
        )

        from reachability_analysis import ReachabilityAnalyzer

        analyzer = ReachabilityAnalyzer(config)
        analyzer.analyze_all()
        analyzer.save_results()
        analyzer.print_summary()


if __name__ == "__main__":
    main()
