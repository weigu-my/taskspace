"""
ROS 2 / RViz2 可视化模块

提供将 ReachabilityResult 发布到 RViz2 的功能：
- PointCloud2 格式点云（高性能）
- 颜色编码灵活度，透明度编码可操作度
- 机器人 URDF 模型发布
- 支持双臂可达空间可视化
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
import struct
import os
import math


def _import_ros2():
    """延迟导入 ROS 2 模块，保持非 ROS 环境兼容"""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField, JointState
    from std_msgs.msg import Header, String
    return rclpy, Node, QoSProfile, DurabilityPolicy, PointCloud2, PointField, JointState, Header, String


@dataclass
class ArmConfig:
    """单臂配置"""
    name: str                          # 臂名称 (left/right)
    urdf_path: Optional[str] = None    # URDF 文件路径
    data_dir: Optional[str] = None     # 数据目录
    data_prefix: str = "reachability"  # 数据文件前缀
    topic_suffix: str = ""             # Topic 后缀
    color_override: Optional[tuple] = None  # 颜色覆盖 (R, G, B)，用于区分左右臂


@dataclass
class RVizPublisherConfig:
    """RViz 发布器配置"""
    # 坐标系
    frame_id: str = "world"

    # 发布频率 (Hz)，0 表示只发布一次
    publish_rate: float = 1.0

    # 是否使用透明度编码可操作度
    use_alpha_for_manipulability: bool = True

    # 颜色模式:
    # - "dexterity": 使用灵活度颜色映射
    # - "arm": 固定使用每个臂的 color_override
    # - "tint": 灵活度颜色映射 + 臂颜色轻微偏色
    color_mode: str = "dexterity"

    # 臂配置列表
    arms: List[ArmConfig] = field(default_factory=list)

    # 基础 Topic 名称
    base_topic: str = "/reachability"

    # 是否发布 TF（让 RViz 的 RobotModel 能正确摆放所有 link）
    publish_tf: bool = True

    # 是否使用结果中的 base_transform 将点云转换到 frame_id
    # 当 TF 不完整或 base_link 在 world 中位置不正确时很有用
    use_base_transform: bool = True

    # 是否在 base_transform 中应用旋转（False 表示仅平移）
    use_base_transform_rotation: bool = True

    # 是否订阅外部 JointState 消息（用于动态姿态控制）
    subscribe_joint_states: bool = False

    # JointState 订阅 Topic
    joint_states_topic: str = "/joint_states"

    # 动态可达性过滤：根据对侧臂姿态实时过滤可达点
    dynamic_filter: bool = False

    # URDF 路径（动态过滤需要用于 pinocchio FK）
    urdf_path: str = ""


class RVizReachabilityPublisher:
    """
    将可达性结果发布到 RViz2

    支持：
    - 单臂或双臂点云发布
    - 机器人 URDF 模型发布
    - RGBA 颜色编码
    """

    def __init__(self, config: Optional[RVizPublisherConfig] = None):
        self.config = config or RVizPublisherConfig()
        self._ros_initialized = False
        self._node = None
        self._publishers = {}  # {arm_name: publisher}
        self._robot_desc_publishers = {}  # {arm_name: publisher}
        self._joint_state_publishers = {}  # {arm_name: publisher}
        self._results = {}  # {arm_name: result}
        self._robot_desc_published = False  # 跟踪是否已发布 robot_description（多臂模式）
        self._tf_broadcaster = None
        self._robot_configs = {}  # {arm_name: RobotConfig}
        self._last_world_transforms = {}  # {arm_name: {link_name: 4x4}}

        # JointState 订阅相关
        self._joint_state_subscriber = None
        self._arm_joint_states = {}  # {arm_name: {joint_name: position}}
        self._arm_joint_names = {}  # {arm_name: [joint_names]} 缓存各臂关节名
        self._joint_state_callbacks = []  # 外部回调列表

        # 动态过滤器 {arm_name: DynamicReachabilityFilter}
        self._dynamic_filters = {}

    def _ensure_ros_init(self):
        """确保 ROS 2 已初始化"""
        if self._ros_initialized:
            return

        (rclpy, Node, QoSProfile, DurabilityPolicy,
         PointCloud2, PointField, JointState, Header, String) = _import_ros2()

        self._rclpy = rclpy
        self._Node = Node
        self._QoSProfile = QoSProfile
        self._DurabilityPolicy = DurabilityPolicy
        self._PointCloud2 = PointCloud2
        self._PointField = PointField
        self._JointState = JointState
        self._Header = Header
        self._String = String

        # 初始化 ROS 2（如果尚未初始化）
        if not rclpy.ok():
            rclpy.init()

        # 创建节点
        self._node = rclpy.create_node('reachability_visualizer')

        # TF broadcaster（可选）
        try:
            import tf2_ros
            from geometry_msgs.msg import TransformStamped
            self._tf2_ros = tf2_ros
            self._TransformStamped = TransformStamped
            self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)
        except Exception as e:
            # 没有 TF 支持时 RobotModel 会缺姿态，但点云仍可用
            self._node.get_logger().warn(f"TF 广播器不可用（RobotModel 可能无法显示完整姿态）: {e}")
            self._tf_broadcaster = None

        # Transient Local QoS for robot_description (latched)
        self._latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # 多臂模式：只发布一个完整的 robot_description
        is_multi_arm = len(self.config.arms) > 1
        robot_desc_published = False

        # 为每个臂创建发布器
        for arm in self.config.arms:
            suffix = f"_{arm.name}" if arm.name else ""
            # 允许显式空字符串保持无后缀；仅在 None 时回退
            topic_suffix = arm.topic_suffix if arm.topic_suffix is not None else suffix

            # 点云发布器
            topic = f"{self.config.base_topic}/points{topic_suffix}"
            self._publishers[arm.name] = self._node.create_publisher(
                PointCloud2, topic, 10
            )

            # 机器人描述发布器 (如果有URDF)
            # 多臂模式：只发布一次完整的 robot_description（不带后缀）
            # 单臂模式：可以发布带后缀的 robot_description
            if arm.urdf_path and os.path.exists(arm.urdf_path):
                if is_multi_arm:
                    # 多臂模式：只发布一次到标准 topic
                    if not robot_desc_published:
                        desc_topic = "/robot_description"
                        self._robot_desc_publishers[arm.name] = self._node.create_publisher(
                            String, desc_topic, self._latched_qos
                        )
                        robot_desc_published = True
                        self._node.get_logger().info(f"多臂模式：发布完整机器人URDF到 {desc_topic}")
                else:
                    # 单臂模式：发布带后缀的 topic
                    desc_topic = f"/robot_description{topic_suffix}"
                    self._robot_desc_publishers[arm.name] = self._node.create_publisher(
                        String, desc_topic, self._latched_qos
                    )

                # 关节状态发布器
                js_topic = f"/joint_states{topic_suffix}"
                self._joint_state_publishers[arm.name] = self._node.create_publisher(
                    JointState, js_topic, 10
                )

        # JointState 订阅器（用于接收外部姿态控制，如 joint_state_publisher_gui）
        if self.config.subscribe_joint_states:
            self._joint_state_subscriber = self._node.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._on_joint_state_received,
                10
            )
            self._node.get_logger().info(
                f'已订阅 JointState: {self.config.joint_states_topic}'
            )
            # 初始化各臂关节名缓存
            for arm in self.config.arms:
                joint_names = self._get_joint_names_from_urdf(arm.urdf_path)
                if joint_names:
                    self._arm_joint_names[arm.name] = joint_names
                    self._arm_joint_states[arm.name] = {n: 0.0 for n in joint_names}

        self._ros_initialized = True
        self._node.get_logger().info(
            f'RViz 可达性发布器已初始化，臂数量: {len(self.config.arms)}'
        )

    def _get_robot_config(self, arm: ArmConfig):
        """解析并缓存 URDF（用于发布 TF）"""
        if arm.name in self._robot_configs:
            return self._robot_configs[arm.name]
        if not arm.urdf_path or not os.path.exists(arm.urdf_path):
            return None
        try:
            from .urdf_parser import URDFParser
            robot_config = URDFParser(arm.urdf_path).parse()
            self._robot_configs[arm.name] = robot_config
            return robot_config
        except Exception as e:
            self._node.get_logger().warn(f"解析 URDF 失败（无法发布 TF）: {e}")

    def _on_joint_state_received(self, msg) -> None:
        """
        处理接收到的 JointState 消息

        将关节位置按臂名分类存储，更新 TF 使点云跟随机器人姿态变化，
        并触发外部回调。
        """
        # 解析消息中的关节位置
        received_joints = dict(zip(msg.name, msg.position))

        # 按臂分类更新关节状态
        changed_arms = []
        for arm_name, arm_joints in self._arm_joint_names.items():
            updated = False
            for joint_name in arm_joints:
                if joint_name in received_joints:
                    new_pos = received_joints[joint_name]
                    old_pos = self._arm_joint_states[arm_name].get(joint_name, 0.0)
                    if abs(new_pos - old_pos) > 1e-6:
                        self._arm_joint_states[arm_name][joint_name] = new_pos
                        updated = True
            if updated:
                changed_arms.append(arm_name)

        # 当关节状态变化时，更新 TF 使机器人模型和点云跟随姿态变化
        if changed_arms and self.config.publish_tf:
            for arm in self.config.arms:
                if arm.name in changed_arms:
                    joint_positions = self.get_arm_joint_positions(arm.name)
                    if joint_positions is not None:
                        self.publish_tf(arm, joint_positions=joint_positions)

        # 触发外部回调
        if changed_arms:
            for callback in self._joint_state_callbacks:
                try:
                    callback(changed_arms, self._arm_joint_states)
                except Exception as e:
                    self._node.get_logger().warn(f"JointState 回调执行失败: {e}")

    def register_joint_state_callback(self, callback) -> None:
        """
        注册 JointState 变化回调

        回调函数签名: callback(changed_arms: List[str], all_states: Dict[str, Dict[str, float]])
        - changed_arms: 本次变化的臂名列表
        - all_states: 所有臂的当前关节状态 {arm_name: {joint_name: position}}
        """
        if callback not in self._joint_state_callbacks:
            self._joint_state_callbacks.append(callback)

    def unregister_joint_state_callback(self, callback) -> None:
        """取消注册 JointState 变化回调"""
        if callback in self._joint_state_callbacks:
            self._joint_state_callbacks.remove(callback)

    def get_arm_joint_positions(self, arm_name: str) -> Optional[List[float]]:
        """
        获取指定臂的当前关节位置列表

        返回:
            关节位置列表（按URDF中的顺序），如果臂不存在则返回None
        """
        if arm_name not in self._arm_joint_states:
            return None
        if arm_name not in self._arm_joint_names:
            return None
        joint_names = self._arm_joint_names[arm_name]
        return [self._arm_joint_states[arm_name].get(n, 0.0) for n in joint_names]

    def get_all_joint_states(self) -> dict:
        """获取所有臂的当前关节状态"""
        return self._arm_joint_states.copy()

    def _init_dynamic_filters(self):
        """
        初始化动态可达性过滤器

        从各臂数据目录加载 dynamic_filter_config.json，
        创建 DynamicReachabilityFilter 实例。
        """
        from .dynamic_filter import DynamicReachabilityFilter

        urdf_path = self.config.urdf_path
        if not urdf_path:
            # 尝试从臂配置获取
            for arm in self.config.arms:
                if arm.urdf_path and os.path.exists(arm.urdf_path):
                    urdf_path = arm.urdf_path
                    break

        if not urdf_path or not os.path.exists(urdf_path):
            self._node.get_logger().warn(
                '[动态过滤] 无法找到 URDF 文件，跳过动态过滤器初始化'
            )
            return

        for arm in self.config.arms:
            if not arm.data_dir or arm.name not in self._results:
                continue

            config_data = DynamicReachabilityFilter.load_config(arm.data_dir)
            if config_data is None:
                self._node.get_logger().warn(
                    f'[动态过滤] {arm.name} 臂: 未找到 dynamic_filter_config.json，跳过'
                )
                continue

            result = self._results[arm.name]
            try:
                filt = DynamicReachabilityFilter(
                    urdf_path=urdf_path,
                    arm_name=config_data['arm_name'],
                    base_link=config_data['base_link'],
                    max_reachable_mask=result.reachable_mask.copy(),
                    grid_points=result.grid_points,
                    dexterity=result.dexterity,
                    manipulability=result.manipulability,
                    other_arm_spheres_local=config_data['other_arm_spheres_local'],
                )
                self._dynamic_filters[arm.name] = filt
                self._node.get_logger().info(
                    f'[动态过滤] {arm.name} 臂: 过滤器已初始化'
                )
            except Exception as e:
                self._node.get_logger().error(
                    f'[动态过滤] {arm.name} 臂: 初始化失败: {e}'
                )

        if self._dynamic_filters:
            # 注册 joint state 回调用于动态过滤
            self.register_joint_state_callback(self._on_dynamic_filter_update)
            self._node.get_logger().info(
                f'[动态过滤] 已注册动态过滤回调，{len(self._dynamic_filters)} 个过滤器就绪'
            )

    def _on_dynamic_filter_update(self, changed_arms, all_states):
        """
        当关节状态变化时，重新过滤可达点并发布更新后的点云

        动态过滤逻辑：
        - 当臂 A 的姿态变化时，需要更新臂 B 的可达空间（因为 A 的碰撞球位置变了）
        - 每个过滤器从最大可达空间重新开始过滤
        """
        if not self._dynamic_filters:
            return

        # 合并所有臂的关节状态为统一的 {joint_name: position}
        all_joint_positions = {}
        for arm_name, states in all_states.items():
            all_joint_positions.update(states)

        # 对每个有动态过滤器的臂执行过滤
        for arm in self.config.arms:
            if arm.name not in self._dynamic_filters:
                continue

            filt = self._dynamic_filters[arm.name]
            result = self._results.get(arm.name)
            if result is None:
                continue

            try:
                # 用全机器人关节状态进行过滤
                filtered_mask = filt.filter(all_joint_positions)

                # 更新结果中的可达掩码
                result.reachable_mask = filtered_mask
                result.reachable_count = int(np.sum(filtered_mask))

                # 重新发布点云
                self.publish_pointcloud(arm, result)

                n_filtered = filt.n_filtered_out
                if n_filtered > 0:
                    self._node.get_logger().info(
                        f'[动态过滤] {arm.name} 臂: '
                        f'过滤 {n_filtered} 点, 剩余 {result.reachable_count} 点'
                    )
            except Exception as e:
                self._node.get_logger().error(
                    f'[动态过滤] {arm.name} 臂: 过滤失败: {e}'
                )

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
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

    def _matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        # 返回 (x, y, z, w)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            S = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return (x, y, z, w)

    def _base_transform_to_matrix(self, base_transform: dict, use_rotation: bool = True) -> Optional[np.ndarray]:
        """将 base_transform(dict) 转为 4x4 变换矩阵"""
        if not base_transform:
            return None
        try:
            t = np.array(base_transform.get('translation'), dtype=np.float64).reshape(3)
            if use_rotation:
                R = np.array(base_transform.get('rotation'), dtype=np.float64).reshape(3, 3)
            else:
                R = np.eye(3, dtype=np.float64)
        except Exception:
            return None
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _get_root_links(self, robot_config) -> List[str]:
        """获取URDF的根链接列表（不作为任何关节的child）"""
        link_names = {l.name for l in robot_config.links}
        child_links = {j.child_link for j in robot_config.joints}
        roots = list(link_names - child_links)
        if not roots:
            return [robot_config.base_link]
        world_like = {"world", "map", "odom"}
        roots.sort(key=lambda x: (0 if x.lower() in world_like else 1, x))
        return roots

    def _compute_world_link_transforms(self, robot_config, joint_pos: dict) -> dict:
        """计算每个 link 相对 frame_id 的 4x4 变换（从URDF根链接开始）"""
        roots = self._get_root_links(robot_config)
        T = {r: np.eye(4, dtype=np.float64) for r in roots}

        # parent -> joints
        by_parent = {}
        for j in robot_config.joints:
            by_parent.setdefault(j.parent_link, []).append(j)

        queue = list(roots)
        visited = set(roots)
        while queue:
            parent = queue.pop(0)
            for j in by_parent.get(parent, []):
                child = j.child_link
                if child in visited:
                    continue
                # origin
                T_origin = np.eye(4, dtype=np.float64)
                T_origin[:3, 3] = np.array(j.origin_xyz, dtype=np.float64)
                T_origin[:3, :3] = self._rpy_to_matrix(j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2])

                # motion
                q = float(joint_pos.get(j.name, 0.0))
                T_motion = np.eye(4, dtype=np.float64)
                if j.type in ("revolute", "continuous"):
                    R = self._axis_angle_to_matrix(np.array(j.axis, dtype=np.float64), q)
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

    def _compute_parent_child_transforms(self, robot_config, joint_pos: dict) -> List[tuple]:
        """计算每个关节的 parent->child 变换（用于发布正确的 TF 树）"""
        transforms = []
        for j in robot_config.joints:
            # origin
            T_origin = np.eye(4, dtype=np.float64)
            T_origin[:3, 3] = np.array(j.origin_xyz, dtype=np.float64)
            T_origin[:3, :3] = self._rpy_to_matrix(j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2])

            # motion
            q = float(joint_pos.get(j.name, 0.0))
            T_motion = np.eye(4, dtype=np.float64)
            if j.type in ("revolute", "continuous"):
                R = self._axis_angle_to_matrix(np.array(j.axis, dtype=np.float64), q)
                T_motion[:3, :3] = R
            elif j.type == "prismatic":
                axis = np.array(j.axis, dtype=np.float64)
                n = np.linalg.norm(axis)
                if n > 1e-12:
                    axis = axis / n
                T_motion[:3, 3] = axis * q

            T_parent_child = T_origin @ T_motion
            transforms.append((j.parent_link, j.child_link, T_parent_child))

        return transforms

    def publish_tf(self, arm: ArmConfig, joint_positions: Optional[List[float]] = None) -> None:
        """发布 TF（parent->child），让 RobotModel 能正确显示所有 link"""
        if not self.config.publish_tf or self._tf_broadcaster is None:
            return

        robot_config = self._get_robot_config(arm)
        if robot_config is None:
            return

        joint_names = self._get_joint_names_from_urdf(arm.urdf_path)
        if joint_positions is None:
            joint_positions = [0.0] * len(joint_names)

        joint_pos = {n: float(p) for n, p in zip(joint_names, joint_positions)}
        # 计算 parent->child 变换，用于正确的 TF 树
        parent_child = self._compute_parent_child_transforms(robot_config, joint_pos)
        # 保留 world 变换用于调试（不用于发布）
        T = self._compute_world_link_transforms(robot_config, joint_pos)
        self._last_world_transforms[arm.name] = T

        now = self._node.get_clock().now().to_msg()
        msgs = []
        # root links 作为 frame_id 的子节点（若不同）
        roots = self._get_root_links(robot_config)
        for root in roots:
            if root == self.config.frame_id:
                continue
            msg = self._TransformStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.config.frame_id
            msg.child_frame_id = root
            msg.transform.translation.x = 0.0
            msg.transform.translation.y = 0.0
            msg.transform.translation.z = 0.0
            msg.transform.rotation.x = 0.0
            msg.transform.rotation.y = 0.0
            msg.transform.rotation.z = 0.0
            msg.transform.rotation.w = 1.0
            msgs.append(msg)

        # 关节 parent->child 变换
        for parent, child, Tpc in parent_child:
            msg = self._TransformStamped()
            msg.header.stamp = now
            msg.header.frame_id = parent
            msg.child_frame_id = child
            msg.transform.translation.x = float(Tpc[0, 3])
            msg.transform.translation.y = float(Tpc[1, 3])
            msg.transform.translation.z = float(Tpc[2, 3])
            x, y, z, w = self._matrix_to_quaternion(Tpc[:3, :3])
            msg.transform.rotation.x = float(x)
            msg.transform.rotation.y = float(y)
            msg.transform.rotation.z = float(z)
            msg.transform.rotation.w = float(w)
            msgs.append(msg)

        # TransformBroadcaster 支持一次发送 list
        self._tf_broadcaster.sendTransform(msgs)

    def load_arm_data(self, arm: ArmConfig) -> Optional[object]:
        """
        加载单臂的可达性数据

        参数:
            arm: 臂配置

        返回:
            SimpleResult 对象或 None
        """
        if not arm.data_dir or not os.path.exists(arm.data_dir):
            return None

        prefix = arm.data_prefix

        # 检查必要文件
        grid_file = os.path.join(arm.data_dir, f"{prefix}_grid_points.npy")
        mask_file = os.path.join(arm.data_dir, f"{prefix}_reachable_mask.npy")

        if not os.path.exists(grid_file) or not os.path.exists(mask_file):
            self._node.get_logger().warn(f'找不到 {arm.name} 臂的数据文件')
            return None

        # 加载数据
        grid_points = np.load(grid_file)
        reachable_mask = np.load(mask_file)

        # 尝试加载可选数据
        dex_file = os.path.join(arm.data_dir, f"{prefix}_dexterity.npy")
        man_file = os.path.join(arm.data_dir, f"{prefix}_manipulability.npy")

        if os.path.exists(dex_file):
            dexterity = np.load(dex_file)
        else:
            dexterity = np.ones(len(grid_points))

        if os.path.exists(man_file):
            manipulability = np.load(man_file)
        else:
            manipulability = np.ones(len(grid_points))

        # 尝试从 stats.json 加载元数据（包括 base_link）
        base_link = ""
        ee_link = ""
        base_transform = None
        points_frame = "base"
        stats_file = os.path.join(arm.data_dir, f"{prefix}_stats.json")
        if os.path.exists(stats_file):
            try:
                import json
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    base_link = stats.get('base_link', '')
                    ee_link = stats.get('ee_link', '')
                    points_frame = stats.get('points_frame', points_frame)
                    bt = stats.get('base_transform')
                    if bt and 'translation' in bt and 'rotation' in bt:
                        base_transform = {
                            'translation': np.array(bt['translation'], dtype=np.float64),
                            'rotation': np.array(bt['rotation'], dtype=np.float64),
                        }
                self._node.get_logger().info(
                    f'{arm.name} 臂: 从 {stats_file} 加载元数据: '
                    f'base_link="{base_link}", points_frame="{points_frame}"'
                )
            except Exception as e:
                self._node.get_logger().warn(f'无法加载 stats.json: {e}')
        else:
            self._node.get_logger().warn(f'{arm.name} 臂: stats.json 不存在: {stats_file}')

        # 创建简化的 Result 对象
        class SimpleResult:
            pass

        result = SimpleResult()
        result.grid_points = grid_points
        result.reachable_mask = reachable_mask
        result.dexterity = dexterity
        result.manipulability = manipulability
        result.reachable_count = int(np.sum(reachable_mask))
        result.color_override = arm.color_override
        result.base_link = base_link  # 添加 base_link 用于正确的坐标系
        result.ee_link = ee_link
        result.base_transform = base_transform
        result.points_frame = points_frame

        return result

    def _resolve_urdf_package_paths(self, urdf_content: str, urdf_path: str) -> str:
        """
        将 URDF 中的 package:// 路径转换为绝对路径，并补全 mesh 的相对路径。

        参数:
            urdf_content: URDF 文件内容
            urdf_path: URDF 文件路径

        返回:
            转换后的 URDF 内容
        """
        import re
        import xml.etree.ElementTree as ET

        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

        def build_package_map() -> dict:
            package_map = {}
            env_paths = os.environ.get("ROS_PACKAGE_PATH", "")
            for base in [p for p in env_paths.split(":") if p]:
                if os.path.isdir(base):
                    for entry in os.listdir(base):
                        pkg_dir = os.path.join(base, entry)
                        if os.path.isdir(pkg_dir):
                            package_map[entry] = pkg_dir
            # 兜底：URDF 同级或上级可能是 package 根目录
            parent_dir = os.path.dirname(urdf_dir)
            if os.path.isdir(parent_dir):
                package_map.setdefault(os.path.basename(parent_dir), parent_dir)
            package_map.setdefault(os.path.basename(urdf_dir), urdf_dir)
            return package_map

        package_map = build_package_map()

        def _to_file_uri(abs_path: str) -> str:
            # RViz/resource_retriever 走 URI 解析；路径里有空格等字符必须做 URL 编码
            from urllib.parse import quote
            abs_path = os.path.abspath(abs_path)
            abs_path = abs_path.replace("\\", "/")
            return "file://" + quote(abs_path, safe="/:")

        def resolve_path(path: str) -> Optional[str]:
            if path.startswith("package://"):
                remainder = path[len("package://"):]
                parts = remainder.split("/", 1)
                if len(parts) == 2:
                    package_name, relative_path = parts
                    if package_name in package_map:
                        candidate = os.path.join(package_map[package_name], relative_path)
                        if os.path.exists(candidate):
                            return os.path.abspath(candidate)
                return None
            if path.startswith("file://"):
                abs_path = path[len("file://"):]
                return abs_path if os.path.exists(abs_path) else None
            if os.path.isabs(path):
                return path if os.path.exists(path) else None
            # 相对路径：优先 URDF 目录，再上一级
            candidate = os.path.join(urdf_dir, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
            candidate = os.path.join(os.path.dirname(urdf_dir), path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
            return None

        # 1) 先替换所有 package:// 的资源路径
        def replace_package_path(match):
            package_path = match.group(0)
            resolved_path = resolve_path(package_path)
            if resolved_path:
                return _to_file_uri(resolved_path)
            return package_path

        resolved = re.sub(r"package://[^\s\"<>]+", replace_package_path, urdf_content)

        # 2) 再解析 XML，补全 mesh filename 的相对路径
        try:
            root = ET.fromstring(resolved)
            updated = False
            missing = []
            for mesh in root.findall(".//mesh"):
                filename = mesh.get("filename")
                if not filename:
                    continue
                resolved_path = resolve_path(filename)
                if resolved_path:
                    mesh.set("filename", _to_file_uri(resolved_path))
                    updated = True
                else:
                    missing.append(filename)
            if missing:
                self._node.get_logger().warn(
                    f"URDF mesh 路径未找到: {len(missing)} 个（示例: {missing[:3]}）"
                )
            if updated:
                resolved = ET.tostring(root, encoding="unicode")
        except Exception as e:
            self._node.get_logger().warn(f"解析 URDF 以补全 mesh 路径失败: {e}")

        return resolved

    def publish_robot_description(self, arm: ArmConfig) -> None:
        """
        发布机器人描述 (URDF)

        多臂模式：只发布一次完整的机器人URDF
        单臂模式：正常发布

        参数:
            arm: 臂配置
        """
        if arm.name not in self._robot_desc_publishers:
            return

        # 多臂模式：只发布一次
        is_multi_arm = len(self.config.arms) > 1
        if is_multi_arm and self._robot_desc_published:
            return

        if not arm.urdf_path or not os.path.exists(arm.urdf_path):
            return

        with open(arm.urdf_path, 'r', encoding='utf-8') as f:
            urdf_content = f.read()

        # 将 package:// 路径转换为绝对路径
        urdf_content = self._resolve_urdf_package_paths(urdf_content, arm.urdf_path)

        msg = self._String()
        msg.data = urdf_content
        self._robot_desc_publishers[arm.name].publish(msg)

        if is_multi_arm:
            self._node.get_logger().info(f'发布完整机器人 URDF 到 /robot_description')
            self._robot_desc_published = True
        else:
            self._node.get_logger().info(f'发布 {arm.name} 臂 robot_description')

    def publish_joint_states(self, arm: ArmConfig, joint_positions: Optional[List[float]] = None) -> None:
        """
        发布关节状态

        参数:
            arm: 臂配置
            joint_positions: 关节位置列表（默认全为0）
        """
        if arm.name not in self._joint_state_publishers:
            return

        # 从 URDF 获取关节名称
        joint_names = self._get_joint_names_from_urdf(arm.urdf_path)
        if not joint_names:
            return

        if joint_positions is None:
            joint_positions = [0.0] * len(joint_names)

        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = joint_positions

        self._joint_state_publishers[arm.name].publish(msg)

    def _get_joint_names_from_urdf(self, urdf_path: str) -> List[str]:
        """从 URDF 文件获取活动关节名称"""
        if not urdf_path or not os.path.exists(urdf_path):
            return []

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            joint_names = []
            for joint in root.findall('.//joint'):
                joint_type = joint.get('type', 'fixed')
                if joint_type in ['revolute', 'continuous', 'prismatic']:
                    joint_names.append(joint.get('name'))

            return joint_names
        except Exception as e:
            self._node.get_logger().warn(f'解析 URDF 失败: {e}')
            return []

    def publish_pointcloud(self, arm: ArmConfig, result) -> None:
        """
        发布点云

        点云数据是相对于各臂的 base_link 坐标系计算的。
        使用各臂的 base_link 作为 frame_id，ROS TF 会自动将点云显示在正确的 world 位置。

        参数:
            arm: 臂配置
            result: 可达性结果
        """
        if arm.name not in self._publishers:
            return

        # 始终发布在 base_link 坐标系中，依赖 TF 将其变换到 world
        points_frame = getattr(result, 'points_frame', 'base')
        base_link = getattr(result, 'base_link', '')
        base_transform = getattr(result, 'base_transform', None)

        # 确定 frame_id 和变换策略
        if base_link and points_frame != 'world':
            # 点云已在 base_link 坐标系中：直接使用 base_link 作为 frame_id
            frame_id = base_link
            transform = None
            self._node.get_logger().info(
                f'{arm.name} 臂: 点云使用 frame_id={frame_id}（依赖 TF 变换到 world）'
            )
        elif base_link and points_frame == 'world' and base_transform is not None:
            # 点云在 world 坐标系中，但需要发布在 base_link 帧以便跟随机器人姿态
            # 计算逆变换：world -> base_link
            frame_id = base_link
            T_world_base = self._base_transform_to_matrix(
                base_transform,
                self.config.use_base_transform_rotation
            )
            if T_world_base is not None:
                # 逆变换: T_base_world = inv(T_world_base)
                R_inv = T_world_base[:3, :3].T
                t_inv = -R_inv @ T_world_base[:3, 3]
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = R_inv
                transform[:3, 3] = t_inv
                self._node.get_logger().info(
                    f'{arm.name} 臂: 点云从 world 逆变换到 frame_id={frame_id}（跟随 TF 变化）'
                )
            else:
                transform = None
                self._node.get_logger().warn(
                    f'{arm.name} 臂: 逆变换失败，点云在 base_link 帧中可能位置不正确'
                )
        elif base_link and points_frame == 'world':
            # 点云在 world 坐标系且无 base_transform，直接使用 world 帧
            frame_id = self.config.frame_id
            transform = None
            self._node.get_logger().info(
                f'{arm.name} 臂: 点云在 world 坐标系，使用 frame_id={frame_id}'
            )
        elif base_transform is not None:
            # Fallback: base_link 为空但有 base_transform，手动变换到 world
            frame_id = self.config.frame_id
            transform = self._base_transform_to_matrix(
                base_transform,
                self.config.use_base_transform_rotation
            )
            self._node.get_logger().warn(
                f'{arm.name} 臂: base_link 为空，使用 base_transform 手动变换到 {frame_id}'
            )
        else:
            # 无法确定正确变换，使用配置的 frame_id（可能位置不对）
            frame_id = self.config.frame_id
            transform = None
            self._node.get_logger().warn(
                f'{arm.name} 臂: base_link 为空且无 base_transform，'
                f'使用 frame_id={frame_id}！点云位置可能不正确。'
            )

        msg = self._create_pointcloud2(result, transform=transform, frame_id=frame_id)
        if msg is None:
            return
        self._publishers[arm.name].publish(msg)
        self._node.get_logger().info(f'发布 {arm.name} 臂 PointCloud2: {result.reachable_count} 个点 (frame: {frame_id})')

    def _create_pointcloud2(self, result, transform: Optional[np.ndarray] = None, frame_id: Optional[str] = None) -> 'PointCloud2':
        """
        创建 PointCloud2 消息

        如果提供了 transform 4x4矩阵，将点云从 arm 局部坐标系变换到 world 坐标系。
        变换公式: p_world = R @ p_arm + t
        """
        from .utils import colormap_dexterity

        # 获取可达点（复制以避免修改原始数据）
        mask = result.reachable_mask
        points = result.grid_points[mask].copy().astype(np.float64)
        dexterity = result.dexterity[mask]
        manipulability = result.manipulability[mask]

        if len(points) == 0:
            self._node.get_logger().warn('可达点数为 0，跳过点云发布')
            return None

        # 应用完整变换：旋转 + 平移
        if transform is not None:
            R = transform[:3, :3]
            t = transform[:3, 3]
            # p_world = R @ p_arm + t
            points = (R @ points.T).T + t
            self._node.get_logger().info(
                f'变换后坐标范围: X[{points[:,0].min():.2f},{points[:,0].max():.2f}] '
                f'Y[{points[:,1].min():.2f},{points[:,1].max():.2f}] '
                f'Z[{points[:,2].min():.2f},{points[:,2].max():.2f}]'
            )
        
        # 确保是 float32 用于 PointCloud2
        points = points.astype(np.float32)

        # 计算颜色
        color_mode = getattr(self.config, "color_mode", "dexterity")
        if color_mode not in ("dexterity", "arm", "tint"):
            color_mode = "dexterity"
        override = getattr(result, 'color_override', None)

        if color_mode == "arm" and override is not None:
            # 使用覆盖颜色
            r, g, b = override
            colors = np.tile([[r, g, b]], (len(points), 1))
        else:
            colors = colormap_dexterity(dexterity)  # [n, 3] RGB in [0,1]
            if color_mode == "tint" and override is not None:
                # 轻微偏色，仍保留灵活度色阶
                tint = np.array(override, dtype=np.float64).reshape(1, 3)
                colors = 0.7 * colors + 0.3 * tint
                colors = np.clip(colors, 0.0, 1.0)

        # 计算透明度（编码可操作度）
        if self.config.use_alpha_for_manipulability:
            m_min, m_max = manipulability.min(), manipulability.max()
            if m_max > m_min:
                alpha = 0.3 + 0.7 * (manipulability - m_min) / (m_max - m_min)
            else:
                alpha = np.ones_like(manipulability)
        else:
            alpha = np.ones(len(points))

        # 构建点云数据
        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            a = alpha[i]

            # 打包为二进制数据 (x, y, z, rgba)
            rgba = self._pack_rgba(r, g, b, a)
            cloud_data.append(struct.pack('fffI', x, y, z, rgba))

        # 创建 PointCloud2 消息
        msg = self._PointCloud2()
        # 使用臂的 base_link 作为坐标系（如果可用），否则使用配置的 frame_id
        # 这样每个臂的点云都在其自己的 base_link 坐标系中
        if frame_id:
            msg.header.frame_id = frame_id
        elif hasattr(result, 'base_link') and result.base_link:
            msg.header.frame_id = result.base_link
        else:
            msg.header.frame_id = self.config.frame_id
        # 日志输出，方便确认是否选错 base_link
        if hasattr(self, "_node") and self._node is not None:
            self._node.get_logger().info(
                f"PointCloud2 frame_id 选择: {msg.header.frame_id} "
                f"(result.base_link={getattr(result, 'base_link', '') or 'N/A'})"
            )
        msg.header.stamp = self._node.get_clock().now().to_msg()

        # 字段定义
        msg.fields = [
            self._PointField(name='x', offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name='y', offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name='z', offset=8, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name='rgba', offset=12, datatype=self._PointField.UINT32, count=1),
        ]

        msg.point_step = 16  # 4 bytes * 4 fields
        msg.row_step = msg.point_step * len(points)
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.data = b''.join(cloud_data)

        return msg

    def _pack_rgba(self, r: float, g: float, b: float, a: float) -> int:
        """将 RGBA 打包为 uint32"""
        r_int = int(r * 255) & 0xFF
        g_int = int(g * 255) & 0xFF
        b_int = int(b * 255) & 0xFF
        a_int = int(a * 255) & 0xFF
        return (a_int << 24) | (r_int << 16) | (g_int << 8) | b_int

    def publish_all(self) -> None:
        """发布所有臂的数据"""
        self._ensure_ros_init()

        for arm in self.config.arms:
            # 发布机器人描述
            self.publish_robot_description(arm)

            # 获取关节位置：优先使用从 joint_state_publisher_gui 订阅到的值
            joint_positions = None
            if self.config.subscribe_joint_states:
                joint_positions = self.get_arm_joint_positions(arm.name)

            # 发布关节状态（当未使用外部 robot_state_publisher 时）
            if not self.config.subscribe_joint_states:
                self.publish_joint_states(arm, joint_positions)

            self.publish_tf(arm, joint_positions=joint_positions)

            # 加载并发布点云
            if arm.name in self._results:
                result = self._results[arm.name]
            else:
                result = self.load_arm_data(arm)
                if result:
                    self._results[arm.name] = result

            if result:
                self.publish_pointcloud(arm, result)
                self._node.get_logger().info(
                    f'已发布 {arm.name} 臂点云到 {self.config.base_topic}/points{arm.topic_suffix or ""} '
                    f'({result.reachable_count} 点)'
                )

    def spin(self) -> None:
        """持续发布模式"""
        self._ensure_ros_init()

        # 加载所有数据
        for arm in self.config.arms:
            # 若已有内存结果且未指定数据目录，直接复用；否则尝试从磁盘加载
            if arm.name in self._results and arm.data_dir is None:
                result = self._results[arm.name]
                self._node.get_logger().info(
                    f'使用内存中 {arm.name} 臂结果: {result.reachable_count} 个可达点'
                )
            else:
                result = self.load_arm_data(arm)
                if result:
                    self._results[arm.name] = result
                    self._node.get_logger().info(
                        f'加载 {arm.name} 臂数据: {result.reachable_count} 个可达点'
                    )

        # 初始化动态过滤器（如果启用）
        if self.config.dynamic_filter:
            self._init_dynamic_filters()

        # 发布一次机器人描述（latched）
        for arm in self.config.arms:
            self.publish_robot_description(arm)

        if self.config.publish_rate <= 0:
            # 只发布一次，然后保持节点运行
            self.publish_all()
            self._rclpy.spin(self._node)
        else:
            # 定时发布
            timer_period = 1.0 / self.config.publish_rate
            self._node.create_timer(timer_period, self.publish_all)
            self._rclpy.spin(self._node)

    def destroy(self) -> None:
        """销毁节点"""
        if self._node is not None:
            self._node.destroy_node()
        if self._ros_initialized:
            self._rclpy.shutdown()


# ============== 便捷函数 ==============

def publish_reachability_to_rviz(
    result,
    frame_id: str = "world",
    topic: str = "/reachability/points"
) -> None:
    """
    便捷函数：将可达性结果发布到 RViz（单臂）

    参数:
        result: ReachabilityResult 对象
        frame_id: 坐标系名称
        topic: Topic 名称
    """
    arm = ArmConfig(name="default")
    config = RVizPublisherConfig(
        frame_id=frame_id,
        base_topic=topic.rsplit('/', 1)[0] if '/' in topic else "/reachability",
        arms=[arm]
    )
    publisher = RVizReachabilityPublisher(config)
    publisher._ensure_ros_init()
    publisher._results["default"] = result
    publisher.publish_pointcloud(arm, result)


def load_and_publish(
    data_dir: str = "./reachability_output",
    prefix: str = "reachability",
    frame_id: str = "world",
    keep_alive: bool = True,
    urdf_path: Optional[str] = None
) -> None:
    """
    从文件加载数据并发布到 RViz（单臂）

    参数:
        data_dir: 数据目录
        prefix: 文件前缀
        frame_id: 坐标系
        keep_alive: 是否保持节点运行
        urdf_path: URDF 文件路径（可选）
    """
    arm = ArmConfig(
        name="default",
        urdf_path=urdf_path,
        data_dir=data_dir,
        data_prefix=prefix
    )

    config = RVizPublisherConfig(
        frame_id=frame_id,
        publish_rate=1.0 if keep_alive else 0,
        arms=[arm]
    )

    publisher = RVizReachabilityPublisher(config)
    publisher.spin()


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="从结果文件发布点云到 RViz2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./reachability_output",
        help="结果目录（包含 *_grid_points.npy 等文件）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="reachability",
        help="结果文件前缀",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="world",
        help="RViz 固定坐标系",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/reachability/points",
        help="点云 Topic 名称",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="URDF 文件路径（用于发布 robot_description，可选）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只发布一次后退出（默认持续发布）",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 解析 topic，得到 base_topic 和可选后缀（保持与 visualize_rviz 一致）
    if "/" in args.topic:
        base_topic, last = args.topic.rsplit("/", 1)
        base_topic = base_topic or "/"
        if last.startswith("points"):
            topic_suffix = last[len("points"):]
        else:
            topic_suffix = f"_{last}"
    else:
        base_topic = f"/{args.topic}" if not args.topic.startswith("/") else args.topic
        topic_suffix = ""

    arm = ArmConfig(
        name="default",
        urdf_path=args.urdf,
        data_dir=args.data_dir,
        data_prefix=args.prefix,
        topic_suffix=topic_suffix,
    )

    config = RVizPublisherConfig(
        frame_id=args.frame_id,
        publish_rate=0.0 if args.once else 1.0,
        base_topic=base_topic,
        arms=[arm],
    )

    publisher = RVizReachabilityPublisher(config)
    publisher.spin()


class MultiArmRVizPublisher(RVizReachabilityPublisher):
    """
    多臂 RViz 发布器（兼容别名）

    这是 RVizReachabilityPublisher 的别名，提供更直观的命名。
    """

    def run(self, publish_once: bool = False):
        """
        运行发布器

        参数:
            publish_once: 是否只发布一次
        """
        if publish_once:
            self.config.publish_rate = 0.0
        self.spin()


if __name__ == "__main__":
    main()
