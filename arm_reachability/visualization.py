"""
可达性分析可视化模块

本模块提供以下可视化功能：
- 从 URDF 渲染机器人网格模型
- 可达性点云可视化：
  - 点的 大小 表示可操作度
  - 点的 颜色 表示灵活度
"""

import os
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET

# 强制使用无界面后端，避免 Qt/xcb 依赖导致崩溃
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

from .urdf_parser import RobotConfig, URDFParser
from .reachability import ReachabilityResult
from .utils import colormap_dexterity, size_from_manipulability


@dataclass
class MeshData:
    """网格数据容器"""
    vertices: np.ndarray       # 顶点
    faces: np.ndarray          # 面
    link_name: str             # 链接名称
    transform: np.ndarray      # 4x4 变换矩阵
    scale: Optional[np.ndarray] = None  # (3,) 网格缩放


class URDFMeshLoader:
    """从 URDF 加载和变换网格"""

    def __init__(self, urdf_path: str):
        """
        初始化网格加载器

        参数:
            urdf_path: URDF 文件路径
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)
        self.package_paths: Dict[str, str] = {}

        # 尝试自动检测包路径
        self._detect_package_paths()

    def _detect_package_paths(self):
        """自动检测 ROS 包路径"""
        # 查找常见的包位置
        search_dirs = [
            self.urdf_dir,
            os.path.dirname(self.urdf_dir),
            os.path.join(os.path.dirname(self.urdf_dir), 'meshes'),
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # 尝试查找 package.xml
                package_xml = os.path.join(search_dir, 'package.xml')
                if os.path.exists(package_xml):
                    try:
                        tree = ET.parse(package_xml)
                        root = tree.getroot()
                        name_elem = root.find('name')
                        if name_elem is not None and name_elem.text:
                            self.package_paths[name_elem.text] = search_dir
                    except:
                        pass

    def resolve_mesh_path(self, mesh_filename: str) -> Optional[str]:
        """
        将网格文件名解析为实际文件路径

        参数:
            mesh_filename: URDF 中的网格文件名（可能使用 package://）

        返回:
            解析后的文件路径，未找到则返回 None
        """
        if not mesh_filename:
            return None

        # 处理 package:// URL
        if mesh_filename.startswith('package://'):
            # 提取包名和相对路径
            remainder = mesh_filename[len('package://'):]
            parts = remainder.split('/', 1)

            if len(parts) == 2:
                package_name, relative_path = parts

                # 检查已注册的包路径
                if package_name in self.package_paths:
                    resolved = os.path.join(self.package_paths[package_name], relative_path)
                    if os.path.exists(resolved):
                        return resolved

                # 尝试常见的相对路径
                search_paths = [
                    os.path.join(self.urdf_dir, '..', relative_path),
                    os.path.join(self.urdf_dir, '..', '..', package_name, relative_path),
                    os.path.join(self.urdf_dir, relative_path),
                    os.path.join(self.urdf_dir, package_name, relative_path),
                    os.path.join(self.urdf_dir, '..', package_name, relative_path),
                    # 尝试在当前目录及子目录中搜索
                    os.path.join(os.path.dirname(self.urdf_dir), package_name, relative_path),
                    os.path.join(os.path.dirname(os.path.dirname(self.urdf_dir)), package_name, relative_path),
                ]

                for path in search_paths:
                    resolved = os.path.abspath(path)
                    if os.path.exists(resolved):
                        return resolved

                # 最后尝试：在 URDF 附近查找 meshes 目录
                # 向上搜索最多 3 级目录
                current_dir = self.urdf_dir
                for _ in range(3):
                    meshes_dir = os.path.join(current_dir, 'meshes')
                    if os.path.exists(meshes_dir):
                        # 从 relative_path 提取文件名
                        mesh_basename = os.path.basename(relative_path)
                        potential_path = os.path.join(meshes_dir, mesh_basename)
                        if os.path.exists(potential_path):
                            return potential_path
                    current_dir = os.path.dirname(current_dir)

            return None

        # 处理 file:// URL
        if mesh_filename.startswith('file://'):
            return mesh_filename[len('file://'):]

        # 处理相对路径
        if not os.path.isabs(mesh_filename):
            resolved = os.path.join(self.urdf_dir, mesh_filename)
            if os.path.exists(resolved):
                return resolved

            # 尝试父目录
            for parent in ['..', '../..', '../../..']:
                resolved = os.path.join(self.urdf_dir, parent, mesh_filename)
                resolved = os.path.abspath(resolved)
                if os.path.exists(resolved):
                    return resolved

        return mesh_filename if os.path.exists(mesh_filename) else None

    def load_mesh(self, mesh_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从文件加载网格

        参数:
            mesh_path: 网格文件路径

        返回:
            (顶点, 面) 元组，失败则返回 None
        """
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, force='mesh')

            if isinstance(mesh, trimesh.Scene):
                # 合并场景中的所有网格
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return None

            return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

        except Exception as e:
            print(f"[警告] 加载网格失败 {mesh_path}: {e}")
            return None

    def load_robot_meshes(
        self,
        robot_config: RobotConfig,
        joint_positions: Optional[np.ndarray] = None
    ) -> List[MeshData]:
        """
        加载所有机器人网格及其变换

        参数:
            robot_config: 机器人配置
            joint_positions: FK 的关节位置（默认：零配置）

        返回:
            MeshData 对象列表
        """
        meshes = []
        loaded_count = 0
        failed_count = 0

        # 使用 FK 获取每个链接的变换
        link_transforms = self._compute_link_transforms(robot_config, joint_positions)

        for link in robot_config.links:
            # 优先使用可视化网格，其次是碰撞网格
            mesh_filename = link.visual_mesh or link.collision_mesh
            mesh_scale = link.visual_scale if link.visual_mesh else link.collision_scale

            if not mesh_filename:
                continue

            mesh_path = self.resolve_mesh_path(mesh_filename)
            if not mesh_path:
                failed_count += 1
                if failed_count <= 5:  # 只打印前 5 个警告以避免刷屏
                    print(f"[警告] 无法解析网格: {mesh_filename}")
                continue

            result = self.load_mesh(mesh_path)
            if result is None:
                failed_count += 1
                if failed_count <= 5:
                    print(f"[警告] 加载网格失败: {mesh_path}")
                continue

            vertices, faces = result
            loaded_count += 1

            # 获取该链接的变换
            transform = link_transforms.get(link.name, np.eye(4))

            # 如果存在可视化原点，应用它
            if link.visual_origin is not None:
                visual_transform = self._origin_to_transform(link.visual_origin)
                transform = transform @ visual_transform

            meshes.append(MeshData(
                vertices=vertices,
                faces=faces,
                link_name=link.name,
                transform=transform,
                scale=np.array(mesh_scale, dtype=np.float32) if mesh_scale else None
            ))

        print(f"[网格加载器] 已加载 {loaded_count} 个网格，{failed_count} 个失败")
        return meshes

    def _compute_link_transforms(
        self,
        robot_config: RobotConfig,
        joint_positions: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """计算所有链接的 FK 变换"""
        if joint_positions is None:
            joint_positions = np.zeros(robot_config.num_dof)

        transforms = {}

        # 尝试使用 pinocchio 进行精确的 FK
        try:
            import pinocchio as pin

            model = pin.buildModelFromUrdf(robot_config.urdf_path)
            data = model.createData()

            # 构建完整配置
            q = np.zeros(model.nq)
            joint_idx = 0
            for i, name in enumerate(model.names[1:]):
                for j, joint in enumerate(robot_config.active_joints):
                    if joint.name == name and joint_idx < len(joint_positions):
                        q[i] = joint_positions[joint_idx]
                        joint_idx += 1
                        break

            # 计算 FK
            pin.framesForwardKinematics(model, data, q)

            # 获取每个帧的变换
            for i, frame in enumerate(model.frames):
                transforms[frame.name] = data.oMf[i].homogeneous.copy()

            return transforms

        except Exception as e:
            print(f"[警告] Pinocchio FK 失败: {e}")

        # 回退：从关节链计算变换
        transforms[robot_config.base_link] = np.eye(4)

        # 构建父子关系
        parent_to_joint = {}
        for joint in robot_config.joints:
            parent_to_joint[joint.child_link] = joint

        # BFS 计算所有变换
        visited = {robot_config.base_link}
        queue = [robot_config.base_link]
        joint_idx = 0

        while queue:
            current = queue.pop(0)

            for joint in robot_config.joints:
                if joint.parent_link == current and joint.child_link not in visited:
                    # 计算关节变换
                    joint_transform = self._joint_to_transform(
                        joint,
                        joint_positions[joint_idx] if joint.type in ['revolute', 'continuous', 'prismatic'] and joint_idx < len(joint_positions) else 0
                    )

                    if joint.type in ['revolute', 'continuous', 'prismatic']:
                        joint_idx += 1

                    # 链接变换
                    transforms[joint.child_link] = transforms[current] @ joint_transform

                    visited.add(joint.child_link)
                    queue.append(joint.child_link)

        return transforms

    def _joint_to_transform(self, joint, q: float) -> np.ndarray:
        """将关节信息和位置转换为 4x4 变换"""
        # 原点变换
        T = self._origin_to_transform(joint.origin_xyz + joint.origin_rpy)

        # 关节运动
        if joint.type == 'revolute' or joint.type == 'continuous':
            axis = np.array(joint.axis)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            R = self._axis_angle_to_matrix(axis, q)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T = T @ T_joint

        elif joint.type == 'prismatic':
            axis = np.array(joint.axis)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            T_joint = np.eye(4)
            T_joint[:3, 3] = axis * q
            T = T @ T_joint

        return T

    def _origin_to_transform(self, origin: List[float]) -> np.ndarray:
        """将原点 (xyz + rpy) 转换为 4x4 变换"""
        T = np.eye(4)

        if len(origin) >= 3:
            T[:3, 3] = origin[:3]

        if len(origin) >= 6:
            roll, pitch, yaw = origin[3:6]
            T[:3, :3] = self._rpy_to_matrix(roll, pitch, yaw)

        return T

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """将 roll-pitch-yaw 转换为旋转矩阵"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        return R

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """使用罗德里格斯公式将轴角转换为旋转矩阵"""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R


class ReachabilityVisualizer:
    """
    可达性分析结果可视化器

    功能：
    - 机器人网格模型渲染
    - 可达性点云可视化
    - 点的 大小 = 可操作度
    - 点的 颜色 = 灵活度
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        result: ReachabilityResult
    ):
        """
        初始化可视化器

        参数:
            robot_config: 机器人配置
            result: 可达性分析结果
        """
        self.robot_config = robot_config
        self.result = result
        self.mesh_loader = URDFMeshLoader(robot_config.urdf_path)

    def visualize_plotly(
        self,
        show_robot: bool = True,
        show_points: bool = True,
        point_size_range: Tuple[float, float] = (2.0, 15.0),
        robot_opacity: float = 0.5,
        robot_color: str = 'lightgray',
        save_html: Optional[str] = None
    ):
        """
        使用 Plotly 创建交互式 3D 可视化

        参数:
            show_robot: 是否显示机器人网格
            show_points: 是否显示可达性点
            point_size_range: (最小大小, 最大大小) 用于点缩放
            robot_opacity: 机器人网格不透明度
            robot_color: 机器人网格颜色
            save_html: HTML 文件保存路径
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # 添加机器人网格
        if show_robot:
            print(f"[可视化] 从 {self.robot_config.urdf_path} 加载机器人网格...")
            meshes = self.mesh_loader.load_robot_meshes(self.robot_config)

            if len(meshes) == 0:
                print(f"[警告] 未加载任何网格！机器人模型将在可视化中不可见。")
                print(f"[警告] 请检查 URDF 中的网格文件是否存在且路径正确。")
            else:
                print(f"[可视化] 向可视化添加 {len(meshes)} 个网格...")

            for mesh_data in meshes:
                # 变换顶点
                vertices = mesh_data.vertices
                if mesh_data.scale is not None:
                    vertices = vertices * mesh_data.scale.reshape(1, 3)
                ones = np.ones((len(vertices), 1))
                vertices_h = np.hstack([vertices, ones])
                transformed = (mesh_data.transform @ vertices_h.T).T[:, :3]

                # 添加网格
                fig.add_trace(go.Mesh3d(
                    x=transformed[:, 0],
                    y=transformed[:, 1],
                    z=transformed[:, 2],
                    i=mesh_data.faces[:, 0],
                    j=mesh_data.faces[:, 1],
                    k=mesh_data.faces[:, 2],
                    color=robot_color,
                    opacity=robot_opacity,
                    name=mesh_data.link_name,
                    showlegend=False,
                    flatshading=True
                ))

        # 添加可达性点
        if show_points and self.result.reachable_count > 0:
            # 获取可达点及其指标
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]
            manipulability = self.result.manipulability[reachable_mask]

            # 计算颜色（灵活度）
            colors = colormap_dexterity(dexterity)

            # 计算大小（可操作度）
            sizes = size_from_manipulability(
                manipulability,
                min_size=point_size_range[0],
                max_size=point_size_range[1]
            )

            # 创建悬停文本
            hover_text = [
                f"位置: ({x:.3f}, {y:.3f}, {z:.3f})<br>"
                f"灵活度: {d}<br>"
                f"可操作度: {m:.4f}"
                for (x, y, z), d, m in zip(points, dexterity, manipulability)
            ]

            # 转换颜色为 plotly 格式
            color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]

            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color_strings,
                    opacity=0.8,
                    line=dict(width=0)
                ),
                text=hover_text,
                hoverinfo='text',
                name='可达点'
            ))

        # 配置布局
        fig.update_layout(
            title=dict(
                text=f"可达性分析: {self.robot_config.name}",
                x=0.5
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 为灵活度添加颜色条
        if show_points and self.result.reachable_count > 0:
            # 创建用于颜色条的虚拟轨迹
            fig.add_trace(go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(
                    colorscale=[
                        [0, 'rgb(0,0,255)'],      # 蓝色
                        [0.25, 'rgb(0,255,255)'], # 青色
                        [0.5, 'rgb(0,255,0)'],    # 绿色
                        [0.75, 'rgb(255,255,0)'], # 黄色
                        [1, 'rgb(255,0,0)']       # 红色
                    ],
                    cmin=0,
                    cmax=int(np.max(self.result.dexterity)),
                    colorbar=dict(
                        title="灵活度",
                        x=1.02,
                        thickness=20
                    ),
                    showscale=True
                ),
                showlegend=False
            ))

        # 显示或保存
        if save_html:
            fig.write_html(save_html)
            print(f"已保存可视化到 {save_html}")
        else:
            # 无显示环境时避免调用 fig.show() 触发 Qt/X11
            if os.environ.get("DISPLAY") in (None, ""):
                print("[提示] 无显示环境，跳过交互式窗口。使用 --no-visualization 或查看保存的 HTML。")
            else:
                fig.show()

        return fig

    def visualize_matplotlib(
        self,
        show_robot: bool = False,  # Matplotlib 3D 不能很好地处理网格
        point_size_range: Tuple[float, float] = (5.0, 100.0),
        elevation: float = 20,
        azimuth: float = 45,
        save_path: Optional[str] = None
    ):
        """
        使用 Matplotlib 创建静态 3D 可视化

        参数:
            show_robot: 是否显示机器人骨架
            point_size_range: (最小大小, 最大大小) 用于点缩放
            elevation: 视角仰角
            azimuth: 视角方位角
            save_path: 图像保存路径
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 添加可达性点
        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]
            manipulability = self.result.manipulability[reachable_mask]

            # 计算颜色和大小
            colors = colormap_dexterity(dexterity)
            sizes = size_from_manipulability(
                manipulability,
                min_size=point_size_range[0],
                max_size=point_size_range[1]
            )

            # 散点图
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=sizes,
                alpha=0.6,
                depthshade=True
            )

        # 如果需要，添加机器人骨架
        if show_robot:
            self._draw_robot_skeleton(ax)

        # 配置视图
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'可达性: {self.robot_config.name}\n'
                    f'点数: {self.result.reachable_count}/{self.result.total_points} '
                    f'({self.result.reachability_ratio*100:.1f}%)')

        # 添加颜色条
        if self.result.reachable_count > 0:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.jet,
                norm=plt.Normalize(0, np.max(self.result.dexterity))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('灵活度')

        # 添加大小图例
        if self.result.reachable_count > 0:
            ax.text2D(0.02, 0.98, "大小 = 可操作度",
                     transform=ax.transAxes, fontsize=10,
                     verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存可视化到 {save_path}")
        else:
            plt.show()

        return fig, ax

    def visualize_rviz(
        self,
        frame_id: str = "world",
        topic: str = "/reachability/points",
        keep_alive: bool = True
    ):
        """
        将当前可达性结果发布到 RViz2

        参数:
            frame_id: 坐标系名称
            topic: 点云 Topic 名称（例如 /reachability/points）
            keep_alive: 是否保持节点运行（持续发布）

        注意: 需要在已安装并 source 的 ROS 2 环境下运行。
        """
        from .ros2_visualization import (
            RVizReachabilityPublisher,
            RVizPublisherConfig,
            ArmConfig,
        )

        # 解析 topic，得到基础前缀和可选后缀
        # 目标: base_topic + "/points" + topic_suffix == 输入的 topic
        if "/" in topic:
            base_topic, last = topic.rsplit("/", 1)
            base_topic = base_topic or "/"
            if last.startswith("points"):
                topic_suffix = last[len("points"):]  # "" 或 "_left"
            else:
                # 非标准命名，退化为直接挂在 base_topic 下
                topic_suffix = f"_{last}"
        else:
            base_topic = f"/{topic}" if not topic.startswith("/") else topic
            topic_suffix = ""

        arm = ArmConfig(
            name="default",
            urdf_path=self.robot_config.urdf_path,
            data_dir=None,            # 直接使用内存中的结果
            data_prefix="reachability",
            topic_suffix=topic_suffix,
        )

        config = RVizPublisherConfig(
            frame_id=frame_id,
            publish_rate=1.0 if keep_alive else 0.0,
            base_topic=base_topic,
            arms=[arm],
        )

        publisher = RVizReachabilityPublisher(config)

        # 直接注入当前结果，避免落盘再加载
        publisher._ensure_ros_init()
        publisher._results[arm.name] = self.result

        print(f"[RViz] 准备发布点云到 {base_topic}/points{topic_suffix}，frame_id={frame_id}，"
              f"点数={self.result.reachable_count}，保持运行={keep_alive}")

        if keep_alive:
            publisher.spin()
        else:
            publisher.publish_all()
            publisher.destroy()

    def _draw_robot_skeleton(self, ax):
        """绘制机器人骨架（关节原点之间的连线）"""
        try:
            import pinocchio as pin

            model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            data = model.createData()

            q = np.zeros(model.nq)
            pin.framesForwardKinematics(model, data, q)

            # 将链接绘制为线段
            for i in range(1, len(model.frames)):
                frame = model.frames[i]
                pos = data.oMf[i].translation

                # 查找父节点
                parent_id = frame.parent
                if parent_id > 0 and parent_id < len(data.oMf):
                    parent_pos = data.oMf[parent_id].translation

                    ax.plot3D(
                        [parent_pos[0], pos[0]],
                        [parent_pos[1], pos[1]],
                        [parent_pos[2], pos[2]],
                        'k-', linewidth=2, alpha=0.7
                    )

                # 绘制关节位置
                ax.scatter(pos[0], pos[1], pos[2],
                          c='black', s=30, marker='o')

        except Exception as e:
            print(f"[警告] 无法绘制机器人骨架: {e}")

    def create_summary_plot(self, save_path: Optional[str] = None):
        """
        创建多视图汇总可视化

        参数:
            save_path: 图像保存路径
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))

        # 主 3D 视图
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_points_3d(ax1, elev=30, azim=45)
        ax1.set_title('3D 视图（等轴测）')

        # 俯视图 (XY)
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_points_2d(ax2, 'x', 'y', '俯视图 (XY)')

        # 正视图 (XZ)
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_points_2d(ax3, 'x', 'z', '正视图 (XZ)')

        # 侧视图 (YZ)
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_points_2d(ax4, 'y', 'z', '侧视图 (YZ)')

        plt.suptitle(f'可达性分析: {self.robot_config.name}\n'
                    f'可达: {self.result.reachable_count}/{self.result.total_points} '
                    f'({self.result.reachability_ratio*100:.1f}%)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存汇总图到 {save_path}")
        else:
            plt.show()

        return fig

    def _plot_points_3d(self, ax, elev=30, azim=45):
        """绘制 3D 点的辅助函数"""
        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]

            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=dexterity, cmap='jet', s=5, alpha=0.5
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def _plot_points_2d(self, ax, x_axis: str, y_axis: str, title: str):
        """绘制 2D 投影的辅助函数"""
        axis_map = {'x': 0, 'y': 1, 'z': 2}

        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]

            xi = axis_map[x_axis]
            yi = axis_map[y_axis]

            scatter = ax.scatter(
                points[:, xi], points[:, yi],
                c=dexterity, cmap='jet', s=3, alpha=0.5
            )
            import matplotlib.pyplot as plt
            plt.colorbar(scatter, ax=ax, label='灵活度')

        ax.set_xlabel(f'{x_axis.upper()} (m)')
        ax.set_ylabel(f'{y_axis.upper()} (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


def visualize_results(
    urdf_path: str,
    result: ReachabilityResult,
    output_dir: str = "./",
    show: bool = True
):
    """
    可视化结果的便捷函数

    参数:
        urdf_path: URDF 文件路径
        result: ReachabilityResult
        output_dir: 保存可视化的目录
        show: 是否显示交互式可视化
    """
    parser = URDFParser(urdf_path)
    robot_config = parser.parse()

    visualizer = ReachabilityVisualizer(robot_config, result)

    # 保存 HTML 可视化
    html_path = os.path.join(output_dir, "reachability_visualization.html")
    visualizer.visualize_plotly(
        show_robot=True,
        save_html=html_path
    )

    # 保存汇总图
    summary_path = os.path.join(output_dir, "reachability_summary.png")
    visualizer.create_summary_plot(save_path=summary_path)

    if show:
        visualizer.visualize_plotly(show_robot=True)
