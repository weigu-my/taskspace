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

    # 臂配置列表
    arms: List[ArmConfig] = field(default_factory=list)

    # 基础 Topic 名称
    base_topic: str = "/reachability"


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

        # Transient Local QoS for robot_description (latched)
        self._latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # 为每个臂创建发布器
        for arm in self.config.arms:
            suffix = f"_{arm.name}" if arm.name else ""
            topic_suffix = arm.topic_suffix or suffix

            # 点云发布器
            topic = f"{self.config.base_topic}/points{topic_suffix}"
            self._publishers[arm.name] = self._node.create_publisher(
                PointCloud2, topic, 10
            )

            # 机器人描述发布器 (如果有URDF)
            if arm.urdf_path and os.path.exists(arm.urdf_path):
                desc_topic = f"/robot_description{topic_suffix}"
                self._robot_desc_publishers[arm.name] = self._node.create_publisher(
                    String, desc_topic, self._latched_qos
                )

                # 关节状态发布器
                js_topic = f"/joint_states{topic_suffix}"
                self._joint_state_publishers[arm.name] = self._node.create_publisher(
                    JointState, js_topic, 10
                )

        self._ros_initialized = True
        self._node.get_logger().info(
            f'RViz 可达性发布器已初始化，臂数量: {len(self.config.arms)}'
        )

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

        return result

    def publish_robot_description(self, arm: ArmConfig) -> None:
        """
        发布机器人描述 (URDF)

        参数:
            arm: 臂配置
        """
        if arm.name not in self._robot_desc_publishers:
            return

        if not arm.urdf_path or not os.path.exists(arm.urdf_path):
            return

        with open(arm.urdf_path, 'r') as f:
            urdf_content = f.read()

        msg = self._String()
        msg.data = urdf_content
        self._robot_desc_publishers[arm.name].publish(msg)
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

        参数:
            arm: 臂配置
            result: 可达性结果
        """
        if arm.name not in self._publishers:
            return

        msg = self._create_pointcloud2(result)
        self._publishers[arm.name].publish(msg)
        self._node.get_logger().info(f'发布 {arm.name} 臂 PointCloud2: {result.reachable_count} 个点')

    def _create_pointcloud2(self, result) -> 'PointCloud2':
        """创建 PointCloud2 消息"""
        from .utils import colormap_dexterity

        # 获取可达点
        mask = result.reachable_mask
        points = result.grid_points[mask]
        dexterity = result.dexterity[mask]
        manipulability = result.manipulability[mask]

        # 计算颜色
        if hasattr(result, 'color_override') and result.color_override is not None:
            # 使用覆盖颜色
            r, g, b = result.color_override
            colors = np.tile([[r, g, b]], (len(points), 1))
        else:
            colors = colormap_dexterity(dexterity)  # [n, 3] RGB in [0,1]

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
        msg.header.frame_id = self.config.frame_id
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

            # 发布关节状态
            self.publish_joint_states(arm)

            # 加载并发布点云
            if arm.name in self._results:
                result = self._results[arm.name]
            else:
                result = self.load_arm_data(arm)
                if result:
                    self._results[arm.name] = result

            if result:
                self.publish_pointcloud(arm, result)

    def spin(self) -> None:
        """持续发布模式"""
        self._ensure_ros_init()

        # 加载所有数据
        for arm in self.config.arms:
            result = self.load_arm_data(arm)
            if result:
                self._results[arm.name] = result
                self._node.get_logger().info(
                    f'加载 {arm.name} 臂数据: {result.reachable_count} 个可达点'
                )

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
