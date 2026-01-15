#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
交互式 3D 可达性可视化工具
===========================================================

功能：
1. 加载并显示 URDF 机器人模型
2. 显示可达空间点云
3. 支持鼠标交互（旋转、缩放、平移）
4. 类似 MuJoCo 的交互体验

依赖：
- Open3D: pip install open3d
- 或 PyVista: pip install pyvista
- 或 Plotly: pip install plotly (生成 HTML 交互式页面)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Union
import numpy as np

# 尝试导入不同的可视化库
OPEN3D_AVAILABLE = False
PYVISTA_AVAILABLE = False
PLOTLY_AVAILABLE = False
YOURDFPY_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    pass

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    from yourdfpy import URDF
    YOURDFPY_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# URDF 加载器
# ===================================================================

class URDFLoader:
    """URDF 机器人模型加载器"""

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.meshes = []
        self.link_transforms = {}

        if YOURDFPY_AVAILABLE:
            self._load_with_yourdfpy()
        else:
            print("[WARN] yourdfpy 未安装，尝试手动解析 URDF")
            self._load_manual()

    def _load_with_yourdfpy(self):
        """使用 yourdfpy 加载 URDF"""
        try:
            self.robot = URDF.load(self.urdf_path)
            # 获取所有 link 的网格
            scene = self.robot.scene
            if scene is not None:
                for name, geometry in scene.geometry.items():
                    if hasattr(geometry, 'vertices'):
                        self.meshes.append({
                            'name': name,
                            'vertices': np.array(geometry.vertices),
                            'faces': np.array(geometry.faces) if hasattr(geometry, 'faces') else None,
                        })
            print(f"[INFO] 成功加载 URDF: {self.urdf_path}")
            print(f"[INFO] 找到 {len(self.meshes)} 个网格")
        except Exception as e:
            print(f"[WARN] yourdfpy 加载失败: {e}")
            self._load_manual()

    def _load_manual(self):
        """手动解析 URDF (简化版)"""
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()

            urdf_dir = Path(self.urdf_path).parent

            for link in root.findall('.//link'):
                link_name = link.get('name')

                # 查找 visual 元素
                for visual in link.findall('visual'):
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            filename = mesh.get('filename')
                            if filename:
                                # 处理 package:// 路径
                                if filename.startswith('package://'):
                                    filename = filename.replace('package://', '')
                                mesh_path = urdf_dir / filename

                                if mesh_path.exists():
                                    self._load_mesh_file(str(mesh_path), link_name)

            print(f"[INFO] 手动解析完成，找到 {len(self.meshes)} 个网格")

        except Exception as e:
            print(f"[ERROR] URDF 解析失败: {e}")

    def _load_mesh_file(self, mesh_path: str, name: str):
        """加载网格文件 (STL, OBJ, DAE)"""
        try:
            if OPEN3D_AVAILABLE:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if mesh.has_vertices():
                    self.meshes.append({
                        'name': name,
                        'vertices': np.asarray(mesh.vertices),
                        'faces': np.asarray(mesh.triangles),
                        'o3d_mesh': mesh,
                    })
        except Exception as e:
            print(f"[WARN] 无法加载网格 {mesh_path}: {e}")

    def get_combined_mesh_o3d(self) -> Optional['o3d.geometry.TriangleMesh']:
        """获取合并后的 Open3D 网格"""
        if not OPEN3D_AVAILABLE:
            return None

        combined = o3d.geometry.TriangleMesh()
        for mesh_data in self.meshes:
            if 'o3d_mesh' in mesh_data:
                combined += mesh_data['o3d_mesh']

        if combined.has_vertices():
            combined.compute_vertex_normals()
            combined.paint_uniform_color([0.7, 0.7, 0.7])
            return combined
        return None


# ===================================================================
# Open3D 可视化器
# ===================================================================

class Open3DVisualizer:
    """基于 Open3D 的交互式可视化器"""

    def __init__(self):
        if not OPEN3D_AVAILABLE:
            raise RuntimeError("Open3D 未安装，请运行: pip install open3d")

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        robot_mesh: Optional['o3d.geometry.TriangleMesh'] = None,
        point_size: float = 2.0,
        title: str = "Reachability Visualization",
    ):
        """
        交互式可视化

        Controls:
        - 左键拖动: 旋转
        - 滚轮: 缩放
        - Shift + 左键: 平移
        - R: 重置视角
        - Q: 退出
        """
        geometries = []

        # 添加可达点云 (绿色)
        if len(reachable_points) > 0:
            pcd_reach = o3d.geometry.PointCloud()
            pcd_reach.points = o3d.utility.Vector3dVector(reachable_points)
            pcd_reach.paint_uniform_color([0.0, 0.8, 0.0])  # 绿色
            geometries.append(pcd_reach)

        # 添加不可达点云 (红色，半透明效果通过稀疏采样实现)
        if unreachable_points is not None and len(unreachable_points) > 0:
            # 稀疏采样以减少视觉干扰
            if len(unreachable_points) > 5000:
                indices = np.random.choice(len(unreachable_points), 5000, replace=False)
                unreachable_sample = unreachable_points[indices]
            else:
                unreachable_sample = unreachable_points

            pcd_unreach = o3d.geometry.PointCloud()
            pcd_unreach.points = o3d.utility.Vector3dVector(unreachable_sample)
            pcd_unreach.paint_uniform_color([0.8, 0.0, 0.0])  # 红色
            geometries.append(pcd_unreach)

        # 添加机器人网格
        if robot_mesh is not None:
            geometries.append(robot_mesh)

        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(coord_frame)

        # 创建可视化窗口
        print("\n" + "="*50)
        print("Open3D 交互式可视化")
        print("="*50)
        print("鼠标操作:")
        print("  - 左键拖动: 旋转视角")
        print("  - 滚轮: 缩放")
        print("  - Shift + 左键: 平移")
        print("  - Ctrl + 左键: 旋转 (另一种模式)")
        print("键盘操作:")
        print("  - R: 重置视角")
        print("  - Q/Esc: 退出")
        print("  - H: 显示帮助")
        print("  - P: 截图")
        print("="*50 + "\n")

        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1280,
            height=720,
            point_show_normal=False,
        )


# ===================================================================
# PyVista 可视化器
# ===================================================================

class PyVistaVisualizer:
    """基于 PyVista 的交互式可视化器"""

    def __init__(self):
        if not PYVISTA_AVAILABLE:
            raise RuntimeError("PyVista 未安装，请运行: pip install pyvista")

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        robot_mesh_path: Optional[str] = None,
        point_size: float = 5.0,
        title: str = "Reachability Visualization",
    ):
        """交互式可视化"""
        plotter = pv.Plotter(title=title)
        plotter.set_background('white')

        # 添加可达点云
        if len(reachable_points) > 0:
            cloud_reach = pv.PolyData(reachable_points)
            plotter.add_mesh(
                cloud_reach,
                color='green',
                point_size=point_size,
                render_points_as_spheres=True,
                label='Reachable'
            )

        # 添加不可达点云
        if unreachable_points is not None and len(unreachable_points) > 0:
            # 稀疏采样
            if len(unreachable_points) > 5000:
                indices = np.random.choice(len(unreachable_points), 5000, replace=False)
                unreachable_sample = unreachable_points[indices]
            else:
                unreachable_sample = unreachable_points

            cloud_unreach = pv.PolyData(unreachable_sample)
            plotter.add_mesh(
                cloud_unreach,
                color='red',
                point_size=point_size * 0.5,
                opacity=0.3,
                render_points_as_spheres=True,
                label='Unreachable'
            )

        # 添加坐标轴
        plotter.add_axes()
        plotter.add_legend()

        print("\n" + "="*50)
        print("PyVista 交互式可视化")
        print("="*50)
        print("鼠标操作:")
        print("  - 左键拖动: 旋转")
        print("  - 滚轮: 缩放")
        print("  - 中键拖动: 平移")
        print("键盘操作:")
        print("  - Q: 退出")
        print("  - V: 切换视角")
        print("  - S: 保存截图")
        print("="*50 + "\n")

        plotter.show()


# ===================================================================
# Plotly 可视化器 (生成 HTML)
# ===================================================================

class PlotlyVisualizer:
    """基于 Plotly 的可视化器，生成交互式 HTML"""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise RuntimeError("Plotly 未安装，请运行: pip install plotly")

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        robot_vertices: Optional[np.ndarray] = None,
        robot_faces: Optional[np.ndarray] = None,
        title: str = "Reachability Visualization",
        output_html: str = "reachability_3d.html",
        point_size: float = 2.0,
    ):
        """生成交互式 HTML 可视化"""
        traces = []

        # 可达点云
        if len(reachable_points) > 0:
            # 采样以提高性能
            if len(reachable_points) > 10000:
                indices = np.random.choice(len(reachable_points), 10000, replace=False)
                points_sample = reachable_points[indices]
            else:
                points_sample = reachable_points

            traces.append(go.Scatter3d(
                x=points_sample[:, 0],
                y=points_sample[:, 1],
                z=points_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color='green',
                    opacity=0.6,
                ),
                name=f'Reachable ({len(reachable_points)} pts)',
            ))

        # 不可达点云
        if unreachable_points is not None and len(unreachable_points) > 0:
            if len(unreachable_points) > 5000:
                indices = np.random.choice(len(unreachable_points), 5000, replace=False)
                unreach_sample = unreachable_points[indices]
            else:
                unreach_sample = unreachable_points

            traces.append(go.Scatter3d(
                x=unreach_sample[:, 0],
                y=unreach_sample[:, 1],
                z=unreach_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size * 0.5,
                    color='red',
                    opacity=0.2,
                ),
                name=f'Unreachable ({len(unreachable_points)} pts)',
            ))

        # 机器人网格
        if robot_vertices is not None and robot_faces is not None:
            traces.append(go.Mesh3d(
                x=robot_vertices[:, 0],
                y=robot_vertices[:, 1],
                z=robot_vertices[:, 2],
                i=robot_faces[:, 0],
                j=robot_faces[:, 1],
                k=robot_faces[:, 2],
                color='gray',
                opacity=0.5,
                name='Robot',
            ))

        # 添加坐标轴指示
        axis_length = 0.3
        for axis, color, name in [
            ([axis_length, 0, 0], 'red', 'X'),
            ([0, axis_length, 0], 'green', 'Y'),
            ([0, 0, axis_length], 'blue', 'Z'),
        ]:
            traces.append(go.Scatter3d(
                x=[0, axis[0]],
                y=[0, axis[1]],
                z=[0, axis[2]],
                mode='lines',
                line=dict(color=color, width=5),
                name=f'{name} axis',
                showlegend=False,
            ))

        # 创建图形
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',  # 保持真实比例
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        # 保存为 HTML
        fig.write_html(output_html)
        print(f"\n[SAVE] 交互式可视化已保存到: {output_html}")
        print("用浏览器打开该文件即可进行交互式查看")
        print("  - 左键拖动: 旋转")
        print("  - 滚轮: 缩放")
        print("  - 右键拖动: 平移")

        return fig


# ===================================================================
# 统一可视化接口
# ===================================================================

class ReachabilityViewer3D:
    """统一的 3D 可达性可视化器"""

    def __init__(self, backend: str = "auto"):
        """
        初始化可视化器

        Args:
            backend: 可视化后端
                - "auto": 自动选择可用的后端
                - "open3d": 使用 Open3D
                - "pyvista": 使用 PyVista
                - "plotly": 使用 Plotly (生成 HTML)
        """
        self.backend = self._select_backend(backend)
        self.urdf_loader: Optional[URDFLoader] = None

    def _select_backend(self, backend: str) -> str:
        """选择可视化后端"""
        if backend == "auto":
            if OPEN3D_AVAILABLE:
                return "open3d"
            elif PYVISTA_AVAILABLE:
                return "pyvista"
            elif PLOTLY_AVAILABLE:
                return "plotly"
            else:
                raise RuntimeError(
                    "没有可用的可视化后端，请安装其中之一:\n"
                    "  pip install open3d\n"
                    "  pip install pyvista\n"
                    "  pip install plotly"
                )
        else:
            available = {
                "open3d": OPEN3D_AVAILABLE,
                "pyvista": PYVISTA_AVAILABLE,
                "plotly": PLOTLY_AVAILABLE,
            }
            if not available.get(backend, False):
                raise RuntimeError(f"{backend} 未安装")
            return backend

    def load_robot(self, urdf_path: str):
        """加载机器人 URDF 模型"""
        self.urdf_loader = URDFLoader(urdf_path)

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        title: str = "Arm Reachability",
        output_html: str = "reachability_3d.html",
        **kwargs,
    ):
        """
        可视化可达空间

        Args:
            reachable_points: 可达点云 (N, 3)
            unreachable_points: 不可达点云 (M, 3)，可选
            title: 标题
            output_html: Plotly 输出文件名
        """
        print(f"[INFO] 使用后端: {self.backend}")

        if self.backend == "open3d":
            viz = Open3DVisualizer()
            robot_mesh = None
            if self.urdf_loader is not None:
                robot_mesh = self.urdf_loader.get_combined_mesh_o3d()
            viz.visualize(
                reachable_points,
                unreachable_points,
                robot_mesh=robot_mesh,
                title=title,
                **kwargs,
            )

        elif self.backend == "pyvista":
            viz = PyVistaVisualizer()
            viz.visualize(
                reachable_points,
                unreachable_points,
                title=title,
                **kwargs,
            )

        elif self.backend == "plotly":
            viz = PlotlyVisualizer()
            robot_vertices = None
            robot_faces = None
            if self.urdf_loader is not None and len(self.urdf_loader.meshes) > 0:
                # 合并所有网格
                all_vertices = []
                all_faces = []
                vertex_offset = 0
                for mesh in self.urdf_loader.meshes:
                    if mesh['vertices'] is not None:
                        all_vertices.append(mesh['vertices'])
                        if mesh['faces'] is not None:
                            all_faces.append(mesh['faces'] + vertex_offset)
                        vertex_offset += len(mesh['vertices'])
                if all_vertices:
                    robot_vertices = np.vstack(all_vertices)
                if all_faces:
                    robot_faces = np.vstack(all_faces)

            viz.visualize(
                reachable_points,
                unreachable_points,
                robot_vertices=robot_vertices,
                robot_faces=robot_faces,
                title=title,
                output_html=output_html,
                **kwargs,
            )


# ===================================================================
# 命令行接口
# ===================================================================

def load_points(path: str) -> np.ndarray:
    """加载点云数据"""
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.csv':
        import pandas as pd
        return pd.read_csv(path).values
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="交互式 3D 可达性可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Open3D 可视化
  python reachability_viewer_3d.py reachable.npy --backend open3d

  # 使用 Plotly 生成 HTML
  python reachability_viewer_3d.py reachable.npy --backend plotly -o result.html

  # 加载机器人模型
  python reachability_viewer_3d.py reachable.npy --urdf robot.urdf
        """
    )

    parser.add_argument("data_path", type=str, help="可达点云文件 (.npy 或 .csv)")
    parser.add_argument("--unreachable", "-u", type=str, help="不可达点云文件")
    parser.add_argument("--urdf", type=str, help="机器人 URDF 文件路径")
    parser.add_argument("--backend", "-b", type=str, default="auto",
                        choices=["auto", "open3d", "pyvista", "plotly"],
                        help="可视化后端")
    parser.add_argument("--output", "-o", type=str, default="reachability_3d.html",
                        help="Plotly 输出 HTML 文件名")
    parser.add_argument("--title", "-t", type=str, default="Arm Reachability",
                        help="可视化标题")

    args = parser.parse_args()

    # 加载数据
    print(f"[INFO] 加载可达点云: {args.data_path}")
    reachable = load_points(args.data_path)
    print(f"[INFO] 可达点数: {len(reachable)}")

    unreachable = None
    if args.unreachable:
        print(f"[INFO] 加载不可达点云: {args.unreachable}")
        unreachable = load_points(args.unreachable)
        print(f"[INFO] 不可达点数: {len(unreachable)}")

    # 创建可视化器
    viewer = ReachabilityViewer3D(backend=args.backend)

    # 加载机器人模型
    if args.urdf:
        print(f"[INFO] 加载机器人模型: {args.urdf}")
        viewer.load_robot(args.urdf)

    # 可视化
    viewer.visualize(
        reachable,
        unreachable,
        title=args.title,
        output_html=args.output,
    )


if __name__ == "__main__":
    main()
