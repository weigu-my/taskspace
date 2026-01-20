"""
URDF Parser and cuRobo Configuration Generator.

This module handles parsing URDF files and generating cuRobo-compatible
configuration for any robot arm.
"""

import os
import yaml
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class JointInfo:
    """Information about a robot joint."""
    name: str
    type: str
    parent_link: str
    child_link: str
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])
    lower_limit: float = -np.pi
    upper_limit: float = np.pi
    velocity_limit: float = 2.0
    effort_limit: float = 100.0
    origin_xyz: List[float] = field(default_factory=lambda: [0, 0, 0])
    origin_rpy: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class LinkInfo:
    """Information about a robot link."""
    name: str
    visual_mesh: Optional[str] = None
    collision_mesh: Optional[str] = None
    visual_origin: Optional[List[float]] = None
    collision_origin: Optional[List[float]] = None
    visual_scale: Optional[List[float]] = None
    collision_scale: Optional[List[float]] = None


@dataclass
class RobotConfig:
    """Extracted robot configuration from URDF."""
    name: str
    joints: List[JointInfo]
    links: List[LinkInfo]
    base_link: str
    ee_link: str
    urdf_path: str
    urdf_dir: str

    @property
    def active_joints(self) -> List[JointInfo]:
        """Get list of active (non-fixed) joints."""
        return [j for j in self.joints if j.type in ['revolute', 'prismatic', 'continuous']]

    @property
    def num_dof(self) -> int:
        """Get number of degrees of freedom."""
        return len(self.active_joints)


class URDFParser:
    """Parser for URDF files."""

    # Keywords to identify end-effector links
    EE_KEYWORDS = [
        'end_effector', 'ee', 'tool', 'gripper', 'hand',
        'tcp', 'flange', 'wrist', 'link_6', 'link_7', 'link6', 'link7'
    ]

    # Keywords to identify base links (higher priority first)
    BASE_KEYWORDS = ['base_link', 'base', 'root', 'link_0', 'link0', 'world']

    # Links that often serve as an external world frame
    WORLD_LIKE = {'world', 'map', 'odom'}

    def __init__(self, urdf_path: str):
        """
        Initialize URDF parser.

        Args:
            urdf_path: Path to URDF file
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)
        self.tree = None
        self.root = None

        self._parse_urdf()

    def _parse_urdf(self):
        """Parse the URDF file."""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

    def get_robot_name(self) -> str:
        """Get the robot name from URDF."""
        return self.root.get('name', 'robot')

    def parse_joints(self) -> List[JointInfo]:
        """Parse all joints from URDF."""
        joints = []

        for joint_elem in self.root.findall('joint'):
            joint = self._parse_joint_element(joint_elem)
            joints.append(joint)

        return joints

    def _parse_joint_element(self, joint_elem: ET.Element) -> JointInfo:
        """Parse a single joint element."""
        name = joint_elem.get('name', '')
        joint_type = joint_elem.get('type', 'fixed')

        # Get parent and child links
        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')
        parent_link = parent_elem.get('link', '') if parent_elem is not None else ''
        child_link = child_elem.get('link', '') if child_elem is not None else ''

        # Get axis
        axis = [0, 0, 1]
        axis_elem = joint_elem.find('axis')
        if axis_elem is not None:
            axis_str = axis_elem.get('xyz', '0 0 1')
            axis = [float(x) for x in axis_str.split()]

        # Get limits
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

        # For continuous joints, set wide limits
        if joint_type == 'continuous':
            lower = -2 * np.pi
            upper = 2 * np.pi

        # Get origin
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
        """Parse all links from URDF."""
        links = []

        for link_elem in self.root.findall('link'):
            link = self._parse_link_element(link_elem)
            links.append(link)

        return links

    def _parse_link_element(self, link_elem: ET.Element) -> LinkInfo:
        """Parse a single link element."""
        name = link_elem.get('name', '')

        visual_mesh = None
        collision_mesh = None
        visual_origin = None
        collision_origin = None
        visual_scale = None
        collision_scale = None

        # Parse visual
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

        # Parse collision
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
        """Auto-detect the base link."""
        link_names = {link.name for link in links}

        # Find links that are only parents (not children of any joint)
        child_links = {j.child_link for j in joints}
        root_links = link_names - child_links

        # Special-case: many URDFs have a dummy 'world' root and a fixed joint to the real base.
        # Example (AR5): world --(fixed)--> <robot_base>
        # In this case, we want <robot_base> as base_link so analysis is centered at the robot base.
        if root_links:
            world_roots = [l for l in root_links if l.lower() in self.WORLD_LIKE]
            if world_roots:
                # Prefer children of fixed joints coming out of world-like roots
                candidates = []
                for j in joints:
                    if j.parent_link.lower() in self.WORLD_LIKE and j.type == 'fixed':
                        if j.child_link in link_names:
                            candidates.append(j.child_link)

                if candidates:
                    # Prefer names that look like base/base_link
                    for cand in candidates:
                        cand_lower = cand.lower()
                        if 'base_link' in cand_lower or cand_lower.endswith('base') or 'base' in cand_lower:
                            return cand
                    # Otherwise pick the first candidate deterministically
                    return sorted(candidates)[0]

        # Check root links for base keywords
        for link in root_links:
            link_lower = link.lower()
            for keyword in self.BASE_KEYWORDS:
                if keyword in link_lower:
                    return link

        # Return first root link if any
        if root_links:
            return list(root_links)[0]

        # Fallback: first link
        if links:
            return links[0].name

        return "base_link"

    def detect_ee_link(self, joints: List[JointInfo], links: List[LinkInfo]) -> str:
        """Auto-detect the end-effector link."""
        link_names = {link.name for link in links}

        # Find links that are only children (not parents of any joint)
        parent_links = {j.parent_link for j in joints}
        leaf_links = link_names - parent_links

        # Check leaf links for EE keywords
        for link in leaf_links:
            link_lower = link.lower()
            for keyword in self.EE_KEYWORDS:
                if keyword in link_lower:
                    return link

        # Check all links for EE keywords
        for link in links:
            link_lower = link.name.lower()
            for keyword in self.EE_KEYWORDS:
                if keyword in link_lower:
                    return link.name

        # Find the longest kinematic chain
        if joints:
            chain = self._find_longest_chain(joints, links)
            if chain:
                return chain[-1]

        # Fallback: last leaf link
        if leaf_links:
            return list(leaf_links)[-1]

        return links[-1].name if links else "ee_link"

    def _find_longest_chain(self, joints: List[JointInfo], links: List[LinkInfo]) -> List[str]:
        """Find the longest kinematic chain in the robot."""
        # Build adjacency graph
        link_names = {link.name for link in links}
        child_to_parent = {}
        parent_to_children = {}

        for joint in joints:
            if joint.parent_link in link_names and joint.child_link in link_names:
                child_to_parent[joint.child_link] = joint.parent_link
                if joint.parent_link not in parent_to_children:
                    parent_to_children[joint.parent_link] = []
                parent_to_children[joint.parent_link].append(joint.child_link)

        # Find leaf links
        leaf_links = link_names - set(parent_to_children.keys())

        # Find longest chain from any leaf to root
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
        """Get the kinematic chain from base to end-effector."""
        # Build parent-child mapping
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
        Parse URDF and return robot configuration.

        Args:
            base_link: Base link name (auto-detect if empty)
            ee_link: End-effector link name (auto-detect if empty)

        Returns:
            RobotConfig with extracted information
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
    """Generate cuRobo configuration from robot config."""

    def __init__(self, robot_config: RobotConfig):
        """
        Initialize config generator.

        Args:
            robot_config: Parsed robot configuration
        """
        self.robot = robot_config

    def generate(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate cuRobo configuration.

        Args:
            output_path: Optional path to save YAML config

        Returns:
            Configuration dictionary
        """
        # Get kinematic chain
        chain_joints = self.robot.active_joints

        # Generate joint names
        joint_names = [j.name for j in chain_joints]

        # Generate default configuration (middle of joint limits)
        default_config = []
        for joint in chain_joints:
            mid = (joint.lower_limit + joint.upper_limit) / 2
            default_config.append(float(mid))

        # Generate retract configuration (same as default for safety)
        retract_config = default_config.copy()

        # Generate CSPACE configuration
        cspace_config = {
            'joint_names': joint_names,
            'retract_config': retract_config,
            'null_space_weight': [1.0] * len(joint_names),
            'cspace_distance_weight': [1.0] * len(joint_names),
            'max_jerk': 500.0,
            'max_acceleration': 15.0,
        }

        # Generate kinematics configuration
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

        # Generate self-collision configuration
        collision_spheres = self._generate_collision_spheres()
        collision_config = {
            'collision_spheres': collision_spheres,
            'buffer_distance': 0.005,
            'self_collision_buffer': {
                'default': 0.02,
            }
        }

        # Complete configuration
        config = {
            'robot_cfg': {
                'kinematics': kinematics_config,
                'collision': collision_config,
            }
        }

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config

    def _generate_collision_spheres(self) -> Dict[str, List[Dict]]:
        """Generate basic collision spheres for each link."""
        spheres = {}

        for link in self.robot.links:
            # Create basic spheres for each link
            spheres[link.name] = [
                {'center': [0.0, 0.0, 0.0], 'radius': 0.05}
            ]

        return spheres

    def generate_ik_config(self) -> Dict[str, Any]:
        """Generate IK solver configuration for cuRobo."""
        chain_joints = self.robot.active_joints
        joint_names = [j.name for j in chain_joints]

        # Build position and velocity limits
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
    Convenience function to load URDF and generate cuRobo config.

    Args:
        urdf_path: Path to URDF file
        base_link: Optional base link name
        ee_link: Optional end-effector link name

    Returns:
        Tuple of (RobotConfig, cuRobo config dict)
    """
    parser = URDFParser(urdf_path)
    robot_config = parser.parse(base_link, ee_link)

    generator = CuroboConfigGenerator(robot_config)
    curobo_config = generator.generate()

    return robot_config, curobo_config
