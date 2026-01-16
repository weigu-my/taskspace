#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
交互式 3D 可达性可视化工具
===========================================================

功能：
1. 加载并显示真实 URDF 机器人网格模型（STL、OBJ、DAE）
2. 显示可达空间点云（支持灵巧性着色）
3. 支持鼠标交互（旋转、缩放、平移）
4. 类似 MuJoCo 的交互体验

依赖：
- trimesh: pip install trimesh (加载网格文件)
- Open3D: pip install open3d
- 或 PyVista: pip install pyvista
- 或 Plotly: pip install plotly (生成 HTML 交互式页面)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np
import xml.etree.ElementTree as ET

# 尝试导入不同的可视化库
OPEN3D_AVAILABLE = False
PYVISTA_AVAILABLE = False
PLOTLY_AVAILABLE = False
TRIMESH_AVAILABLE = False

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
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# 网格加载工具
# ===================================================================

def load_mesh_file(mesh_path: str, scale: List[float] = None) -> Optional[dict]:
    """
    加载网格文件（STL、OBJ、DAE 等）

    Returns:
        dict with 'vertices' (N, 3) and 'faces' (M, 3)
    """
    if not TRIMESH_AVAILABLE:
        return None

    try:
        mesh = trimesh.load(mesh_path, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                return None

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        if scale is not None:
            vertices = vertices * np.array(scale)

        return {'vertices': vertices, 'faces': faces}

    except Exception as e:
        print(f"[WARN] 无法加载网格 {mesh_path}: {e}")
        return None


def transform_mesh(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """应用 4x4 变换矩阵到顶点"""
    n = len(vertices)
    homogeneous = np.hstack([vertices, np.ones((n, 1))])
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


# ===================================================================
# URDF 加载器（支持真实网格模型）
# ===================================================================

class URDFLoader:
    """URDF 机器人模型加载器 - 支持真实网格渲染"""

    def __init__(self, urdf_path: str):
        self.urdf_path = Path(urdf_path)
        self.urdf_dir = self.urdf_path.parent

        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        self.links = {}
        self.joints = {}
        self.meshes = []

        self._parse_urdf()

    def _parse_urdf(self):
        """解析 URDF 文件"""
        # 解析所有链接
        for link in self.root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = {'visual': []}

            for visual in link.findall('visual'):
                visual_info = self._parse_visual(visual)
                if visual_info:
                    self.links[link_name]['visual'].append(visual_info)

        # 解析所有关节
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')

            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            origin = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if origin.get('xyz'):
                    xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    rpy = [float(x) for x in origin.get('rpy').split()]

            axis = [0, 0, 1]
            axis_elem = joint.find('axis')
            if axis_elem is not None and axis_elem.get('xyz'):
                axis = [float(x) for x in axis_elem.get('xyz').split()]

            self.joints[joint_name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
            }

        print(f"[INFO] URDF 解析完成: {len(self.links)} 个链接, {len(self.joints)} 个关节")

    def _parse_visual(self, visual) -> Optional[dict]:
        """解析 visual 元素"""
        geometry = visual.find('geometry')
        if geometry is None:
            return None

        origin = visual.find('origin')
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if origin is not None:
            if origin.get('xyz'):
                xyz = [float(x) for x in origin.get('xyz').split()]
            if origin.get('rpy'):
                rpy = [float(x) for x in origin.get('rpy').split()]

        color = [0.7, 0.7, 0.7, 1.0]
        material = visual.find('material')
        if material is not None:
            color_elem = material.find('color')
            if color_elem is not None and color_elem.get('rgba'):
                color = [float(x) for x in color_elem.get('rgba').split()]

        result = {
            'xyz': xyz, 'rpy': rpy, 'color': color,
            'type': None, 'mesh_path': None, 'scale': [1, 1, 1], 'size': None,
        }

        mesh = geometry.find('mesh')
        box = geometry.find('box')
        cylinder = geometry.find('cylinder')
        sphere = geometry.find('sphere')

        if mesh is not None:
            result['type'] = 'mesh'
            filename = mesh.get('filename')
            if filename:
                if filename.startswith('package://'):
                    package_path = filename.replace('package://', '')
                    possible_paths = [
                        self.urdf_dir / package_path,
                        self.urdf_dir.parent / package_path,
                        self.urdf_dir / package_path.split('/', 1)[-1] if '/' in package_path else None,
                    ]
                    for p in possible_paths:
                        if p and p.exists():
                            result['mesh_path'] = str(p)
                            break
                    if not result['mesh_path']:
                        result['mesh_path'] = str(self.urdf_dir / package_path)
                else:
                    result['mesh_path'] = str(self.urdf_dir / filename)
            if mesh.get('scale'):
                result['scale'] = [float(x) for x in mesh.get('scale').split()]

        elif box is not None:
            result['type'] = 'box'
            result['size'] = [float(x) for x in box.get('size').split()]

        elif cylinder is not None:
            result['type'] = 'cylinder'
            result['size'] = {'radius': float(cylinder.get('radius')), 'length': float(cylinder.get('length'))}

        elif sphere is not None:
            result['type'] = 'sphere'
            result['size'] = {'radius': float(sphere.get('radius'))}

        return result

    def get_base_link(self) -> str:
        """获取基座链接名"""
        child_links = {joint['child'] for joint in self.joints.values()}
        for link_name in self.links.keys():
            if link_name not in child_links:
                return link_name
        return list(self.links.keys())[0]

    def compute_link_transforms(self, joint_positions: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """计算所有链接的世界坐标变换矩阵"""
        if joint_positions is None:
            joint_positions = {}

        transforms = {}
        base_link = self.get_base_link()
        transforms[base_link] = np.eye(4)

        children = {link: [] for link in self.links}
        for joint_name, joint in self.joints.items():
            children[joint['parent']].append((joint_name, joint['child']))

        queue = [base_link]
        visited = {base_link}

        while queue:
            parent_link = queue.pop(0)
            parent_transform = transforms[parent_link]

            for joint_name, child_link in children.get(parent_link, []):
                if child_link in visited:
                    continue

                joint = self.joints[joint_name]
                joint_transform = self._make_transform(joint['xyz'], joint['rpy'])

                angle = joint_positions.get(joint_name, 0.0)
                if joint['type'] in ['revolute', 'continuous']:
                    axis = np.array(joint['axis'])
                    rotation = self._axis_angle_to_matrix(axis, angle)
                    joint_motion = np.eye(4)
                    joint_motion[:3, :3] = rotation
                elif joint['type'] == 'prismatic':
                    axis = np.array(joint['axis'])
                    joint_motion = np.eye(4)
                    joint_motion[:3, 3] = axis * angle
                else:
                    joint_motion = np.eye(4)

                transforms[child_link] = parent_transform @ joint_transform @ joint_motion
                visited.add(child_link)
                queue.append(child_link)

        return transforms

    def _make_transform(self, xyz: List[float], rpy: List[float]) -> np.ndarray:
        """从 xyz 和 rpy 创建 4x4 变换矩阵"""
        T = np.eye(4)
        T[:3, 3] = xyz
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        T[:3, :3] = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return T

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """轴角转旋转矩阵"""
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def _create_box_mesh(self, size: List[float]) -> dict:
        """创建长方体网格"""
        sx, sy, sz = [s/2 for s in size]
        vertices = np.array([
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
        return {'vertices': vertices, 'faces': faces}

    def _create_cylinder_mesh(self, radius: float, length: float, segments: int = 16) -> dict:
        """创建圆柱体网格"""
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        h = length / 2
        top_v = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(segments, h)])
        bot_v = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(segments, -h)])
        vertices = np.vstack([top_v, bot_v, [[0, 0, h]], [[0, 0, -h]]])

        faces = []
        for i in range(segments):
            ni = (i + 1) % segments
            faces.extend([[i, ni, segments + ni], [i, segments + ni, segments + i]])
            faces.extend([[2*segments, i, ni], [2*segments + 1, segments + ni, segments + i]])
        return {'vertices': vertices, 'faces': np.array(faces)}

    def _create_sphere_mesh(self, radius: float, segments: int = 16) -> dict:
        """创建球体网格"""
        phi = np.linspace(0, np.pi, segments)
        theta = np.linspace(0, 2*np.pi, segments)
        phi, theta = np.meshgrid(phi, theta)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

        faces = []
        for i in range(segments - 1):
            for j in range(segments - 1):
                idx = i * segments + j
                faces.extend([[idx, idx + 1, idx + segments], [idx + 1, idx + segments + 1, idx + segments]])
        return {'vertices': vertices, 'faces': np.array(faces)}

    def get_all_meshes(self, joint_positions: Dict[str, float] = None) -> List[dict]:
        """获取所有链接的网格（已变换到世界坐标）"""
        link_transforms = self.compute_link_transforms(joint_positions)
        all_meshes = []

        for link_name, link_info in self.links.items():
            if link_name not in link_transforms:
                continue

            link_transform = link_transforms[link_name]

            for visual in link_info['visual']:
                visual_local = self._make_transform(visual['xyz'], visual['rpy'])
                full_transform = link_transform @ visual_local

                mesh_data = None

                if visual['type'] == 'mesh' and visual['mesh_path']:
                    mesh_data = load_mesh_file(visual['mesh_path'], visual['scale'])
                elif visual['type'] == 'box' and visual['size']:
                    mesh_data = self._create_box_mesh(visual['size'])
                elif visual['type'] == 'cylinder' and visual['size']:
                    mesh_data = self._create_cylinder_mesh(visual['size']['radius'], visual['size']['length'])
                elif visual['type'] == 'sphere' and visual['size']:
                    mesh_data = self._create_sphere_mesh(visual['size']['radius'])

                if mesh_data:
                    transformed_vertices = transform_mesh(mesh_data['vertices'], full_transform)
                    all_meshes.append({
                        'vertices': transformed_vertices,
                        'faces': mesh_data['faces'],
                        'color': visual['color'],
                        'link_name': link_name,
                    })

        return all_meshes

    def get_combined_mesh_o3d(self) -> Optional['o3d.geometry.TriangleMesh']:
        """获取合并后的 Open3D 网格"""
        if not OPEN3D_AVAILABLE:
            return None

        meshes = self.get_all_meshes()
        if not meshes:
            return None

        combined = o3d.geometry.TriangleMesh()
        for mesh_data in meshes:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_data['vertices'])
            mesh.triangles = o3d.utility.Vector3iVector(mesh_data['faces'])
            color = mesh_data.get('color', [0.7, 0.7, 0.7, 1.0])[:3]
            mesh.paint_uniform_color(color)
            combined += mesh

        if combined.has_vertices():
            combined.compute_vertex_normals()
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
    """基于 Plotly 的可视化器，生成交互式 HTML - 支持真实网格渲染"""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise RuntimeError("Plotly 未安装，请运行: pip install plotly")

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        robot_meshes: Optional[List[dict]] = None,
        dexterity: Optional[np.ndarray] = None,
        title: str = "Reachability Visualization",
        output_html: str = "reachability_3d.html",
        point_size: float = 2.0,
        robot_opacity: float = 0.7,
    ):
        """生成交互式 HTML 可视化（支持真实网格模型）"""
        traces = []

        # 可达点云
        if len(reachable_points) > 0:
            if len(reachable_points) > 10000:
                indices = np.random.choice(len(reachable_points), 10000, replace=False)
                points_sample = reachable_points[indices]
                dex_sample = dexterity[indices] if dexterity is not None else None
            else:
                points_sample = reachable_points
                dex_sample = dexterity

            # 根据是否有灵巧性数据选择颜色
            if dex_sample is not None:
                color_values = dex_sample
                colorbar_title = 'Dexterity'
                colorscale = 'RdYlGn'
            else:
                color_values = points_sample[:, 2]
                colorbar_title = 'Z (m)'
                colorscale = 'Viridis'

            traces.append(go.Scatter3d(
                x=points_sample[:, 0],
                y=points_sample[:, 1],
                z=points_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_values,
                    colorscale=colorscale,
                    opacity=0.6,
                    colorbar=dict(title=colorbar_title, x=1.02),
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

        # 机器人网格（真实模型）
        if robot_meshes is not None and len(robot_meshes) > 0:
            for i, mesh in enumerate(robot_meshes):
                vertices = mesh['vertices']
                faces = mesh['faces']
                color = mesh.get('color', [0.7, 0.7, 0.7, 1.0])

                r, g, b = [int(c * 255) for c in color[:3]]
                color_str = f'rgb({r},{g},{b})'

                traces.append(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color_str,
                    opacity=robot_opacity * color[3] if len(color) > 3 else robot_opacity,
                    name=f"Robot_{mesh.get('link_name', i)}",
                    showlegend=False,
                    flatshading=True,
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.8,
                        specular=0.2,
                        roughness=0.5,
                    ),
                    lightposition=dict(x=1, y=1, z=2),
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
                mode='lines+text',
                line=dict(color=color, width=6),
                text=['', name],
                textposition='top center',
                name=f'{name} axis',
                showlegend=False,
            ))

        fig = go.Figure(data=traces)

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=50, b=0),
        )

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
    """统一的 3D 可达性可视化器 - 支持真实网格模型"""

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
        if not TRIMESH_AVAILABLE:
            print("[WARN] trimesh 未安装，网格加载可能受限，运行: pip install trimesh")

    def visualize(
        self,
        reachable_points: np.ndarray,
        unreachable_points: Optional[np.ndarray] = None,
        dexterity: Optional[np.ndarray] = None,
        title: str = "Arm Reachability",
        output_html: str = "reachability_3d.html",
        robot_opacity: float = 0.7,
        **kwargs,
    ):
        """
        可视化可达空间

        Args:
            reachable_points: 可达点云 (N, 3)
            unreachable_points: 不可达点云 (M, 3)，可选
            dexterity: 灵巧性数据 (N,)，可选
            title: 标题
            output_html: Plotly 输出文件名
            robot_opacity: 机器人模型透明度
        """
        print(f"[INFO] 使用后端: {self.backend}")

        # 获取机器人网格
        robot_meshes = None
        if self.urdf_loader is not None:
            robot_meshes = self.urdf_loader.get_all_meshes()
            if robot_meshes:
                print(f"[INFO] 加载了 {len(robot_meshes)} 个机器人网格")

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
            viz.visualize(
                reachable_points,
                unreachable_points,
                robot_meshes=robot_meshes,
                dexterity=dexterity,
                title=title,
                output_html=output_html,
                robot_opacity=robot_opacity,
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
        description="交互式 3D 可达性可视化工具（支持真实机器人网格模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Open3D 可视化
  python reachability_viewer_3d.py reachable.npy --backend open3d

  # 使用 Plotly 生成 HTML（带真实机器人模型）
  python reachability_viewer_3d.py reachable.npy --backend plotly --urdf robot.urdf -o result.html

  # 加载灵巧性数据（颜色表示灵巧性）
  python reachability_viewer_3d.py reachable.npy --dexterity dexterity.npy --urdf robot.urdf

依赖:
  pip install trimesh plotly  # 推荐
  pip install open3d          # 可选，本地窗口显示
        """
    )

    parser.add_argument("data_path", type=str, help="可达点云文件 (.npy 或 .csv)")
    parser.add_argument("--unreachable", "-u", type=str, help="不可达点云文件")
    parser.add_argument("--dexterity", "-d", type=str, help="灵巧性数据文件 (.npy)")
    parser.add_argument("--urdf", type=str, help="机器人 URDF 文件路径（显示真实模型）")
    parser.add_argument("--backend", "-b", type=str, default="auto",
                        choices=["auto", "open3d", "pyvista", "plotly"],
                        help="可视化后端")
    parser.add_argument("--output", "-o", type=str, default="reachability_3d.html",
                        help="Plotly 输出 HTML 文件名")
    parser.add_argument("--title", "-t", type=str, default="Arm Reachability & Dexterity",
                        help="可视化标题")
    parser.add_argument("--robot-opacity", type=float, default=0.7,
                        help="机器人模型透明度 (0-1)")

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

    # 加载灵巧性数据
    dexterity = None
    dex_path = args.dexterity
    if dex_path is None:
        # 尝试自动查找
        auto_path = args.data_path.replace("_reachable.npy", "_dexterity.npy")
        if Path(auto_path).exists():
            dex_path = auto_path
    if dex_path and Path(dex_path).exists():
        dexterity = np.load(dex_path)
        print(f"[INFO] 加载灵巧性数据: {len(dexterity)} 点")
        print(f"[INFO] 灵巧性范围: {dexterity.min()} - {dexterity.max()}")

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
        dexterity=dexterity,
        title=args.title,
        output_html=args.output,
        robot_opacity=args.robot_opacity,
    )


if __name__ == "__main__":
    main()
