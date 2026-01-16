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

    def __init__(self, urdf_path: str):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")

        self.config = URDFRobotConfig(urdf_path=str(urdf_path))
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

        # 找到基座链接（没有父关节的链接）
        child_links = set()
        for joint in robot.joint_map.values():
            child_links.add(joint.child)

        for link_name in self.config.link_names:
            if link_name not in child_links:
                self.config.base_link = link_name
                break

        # 获取所有活动关节（非 fixed）
        for joint_name, joint in robot.joint_map.items():
            if joint.type in ['revolute', 'prismatic', 'continuous']:
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

        # 找到末端链接（运动链的最后一个链接）
        # 简单方法：找到最深的链接
        self.config.ee_link = self._find_ee_link(robot)

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

    def _find_ee_link(self, robot) -> str:
        """找到末端执行器链接"""
        # 构建链接的子链接映射
        children = {link: [] for link in self.config.link_names}
        for joint in robot.joint_map.values():
            if joint.parent in children:
                children[joint.parent].append(joint.child)

        # 找到没有子链接的链接（叶子节点）
        leaf_links = [link for link, childs in children.items() if len(childs) == 0]

        # 优先选择名称中包含 'ee', 'end', 'tool', 'gripper', 'hand' 的链接
        ee_keywords = ['ee', 'end', 'tool', 'gripper', 'hand', 'tcp', 'flange']
        for link in leaf_links:
            for keyword in ee_keywords:
                if keyword in link.lower():
                    return link

        # 否则返回第一个叶子链接
        return leaf_links[0] if leaf_links else self.config.link_names[-1]

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

        robot_cfg = {
            'robot_cfg': {
                'kinematics': {
                    'urdf_path': urdf_abs_path,
                    'asset_root_path': str(Path(self.config.urdf_path).parent.absolute()),
                    'base_link': self.config.base_link,
                    'ee_link': self.config.ee_link,
                    'cspace': {
                        'joint_names': self.config.joint_names,
                        'retract_config': self.config.retract_config,
                        'null_space_weight': [1.0] * len(self.config.joint_names),
                        'cspace_distance_weight': [1.0] * len(self.config.joint_names),
                        'max_jerk': 500.0,
                        'max_acceleration': 15.0,
                    },
                },
                'collision': {
                    'self_collision_check': True,
                    'self_collision_opt': True,
                    # 生成碰撞球体配置
                    'collision_spheres': self._generate_collision_spheres(),
                    # 自碰撞忽略对
                    'self_collision_ignore': {
                        pair[0]: [pair[1]] for pair in self.config.self_collision_ignore
                    } if self.config.self_collision_ignore else {},
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

        world_cfg = {
            'world_model': {
                'world_collision': {
                    'cube': {},
                    'mesh': {},
                },
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
) -> tuple[str, str]:
    """
    将 URDF 文件转换为 cuRobo 配置

    Args:
        urdf_path: URDF 文件路径
        ee_link: 末端执行器链接名（可选，自动检测）
        base_link: 基座链接名（可选，自动检测）
        output_dir: 输出目录

    Returns:
        (robot_config_path, world_config_path)
    """
    # 解析 URDF
    parser = URDFParser(urdf_path)

    # 覆盖用户指定的链接
    if ee_link:
        parser.config.ee_link = ee_link
    if base_link:
        parser.config.base_link = base_link

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
