#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
可达性可视化 + 真实机械臂模型渲染
===========================================================

功能：
1. 加载 URDF 中的网格模型（STL、OBJ、DAE）
2. 计算 FK 获取各链接的变换矩阵
3. 在 Plotly 中渲染真实的机械臂模型
4. 用灵巧性着色显示可达空间

依赖：
- trimesh: pip install trimesh
- yourdfpy: pip install yourdfpy
- plotly: pip install plotly
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import xml.etree.ElementTree as ET

# 尝试导入网格处理库
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARN] trimesh 未安装，运行: pip install trimesh")

try:
    from yourdfpy import URDF
    YOURDFPY_AVAILABLE = True
except ImportError:
    YOURDFPY_AVAILABLE = False


# ===================================================================
# 网格加载和处理
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
            # 如果是场景，合并所有网格
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

        # 应用缩放
        if scale is not None:
            vertices = vertices * np.array(scale)

        return {
            'vertices': vertices,
            'faces': faces,
        }

    except Exception as e:
        print(f"[WARN] 无法加载网格 {mesh_path}: {e}")
        return None


def transform_mesh(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    应用 4x4 变换矩阵到顶点

    Args:
        vertices: (N, 3) 顶点坐标
        transform: (4, 4) 变换矩阵

    Returns:
        transformed vertices (N, 3)
    """
    n = len(vertices)
    # 转换为齐次坐标
    homogeneous = np.hstack([vertices, np.ones((n, 1))])
    # 应用变换
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


# ===================================================================
# URDF 解析和 FK 计算
# ===================================================================

class URDFMeshLoader:
    """从 URDF 加载机械臂网格模型"""

    def __init__(self, urdf_path: str, package_paths: Dict[str, str] = None):
        """
        初始化 URDF 网格加载器

        Args:
            urdf_path: URDF 文件路径
            package_paths: ROS 包名到路径的映射，如 {"robot_model": "/path/to/robot_model"}
        """
        self.urdf_path = Path(urdf_path)
        self.urdf_dir = self.urdf_path.parent
        self.package_paths = package_paths or {}

        # 自动检测常见的包路径
        self._auto_detect_package_paths()

        # 解析 URDF
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        # 提取信息
        self.links = {}
        self.joints = {}
        self.link_meshes = {}

        self._parse_urdf()

    def _auto_detect_package_paths(self):
        """自动检测常见的 ROS 包路径"""
        # 检查 URDF 目录的父目录结构
        # 常见结构：workspace/src/package_name/urdf/robot.urdf
        #         或：package_name/urdf/robot.urdf

        # 检查是否有 meshes 目录在同级或父级
        search_dirs = [
            self.urdf_dir,
            self.urdf_dir.parent,
            self.urdf_dir.parent.parent,
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # 查找包含 meshes 目录的文件夹
            for item in search_dir.iterdir():
                if item.is_dir():
                    meshes_dir = item / "meshes"
                    if meshes_dir.exists():
                        package_name = item.name
                        if package_name not in self.package_paths:
                            self.package_paths[package_name] = str(item)
                            print(f"[INFO] 自动检测到包路径: {package_name} -> {item}")

            # 也检查当前目录是否直接包含 meshes
            meshes_dir = search_dir / "meshes"
            if meshes_dir.exists():
                # 使用父目录名作为包名
                package_name = search_dir.name
                if package_name not in self.package_paths:
                    self.package_paths[package_name] = str(search_dir)
                    print(f"[INFO] 自动检测到包路径: {package_name} -> {search_dir}")

    def _parse_urdf(self):
        """解析 URDF 文件"""
        # 解析所有链接
        for link in self.root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = {
                'visual': [],
                'collision': [],
            }

            # 解析 visual 元素
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

        # 获取原点变换
        origin = visual.find('origin')
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if origin is not None:
            if origin.get('xyz'):
                xyz = [float(x) for x in origin.get('xyz').split()]
            if origin.get('rpy'):
                rpy = [float(x) for x in origin.get('rpy').split()]

        # 获取材质颜色
        color = [0.7, 0.7, 0.7, 1.0]  # 默认灰色
        material = visual.find('material')
        if material is not None:
            color_elem = material.find('color')
            if color_elem is not None and color_elem.get('rgba'):
                color = [float(x) for x in color_elem.get('rgba').split()]

        result = {
            'xyz': xyz,
            'rpy': rpy,
            'color': color,
            'type': None,
            'mesh_path': None,
            'scale': [1, 1, 1],
            'size': None,
        }

        # 解析几何体
        mesh = geometry.find('mesh')
        box = geometry.find('box')
        cylinder = geometry.find('cylinder')
        sphere = geometry.find('sphere')

        if mesh is not None:
            result['type'] = 'mesh'
            filename = mesh.get('filename')
            if filename:
                # 处理 package:// 路径
                if filename.startswith('package://'):
                    # 解析 package://package_name/path/to/file
                    package_path = filename.replace('package://', '')
                    parts = package_path.split('/', 1)
                    package_name = parts[0]
                    relative_path = parts[1] if len(parts) > 1 else ''

                    # 首先检查是否有已知的包路径
                    if package_name in self.package_paths:
                        resolved_path = Path(self.package_paths[package_name]) / relative_path
                        if resolved_path.exists():
                            result['mesh_path'] = str(resolved_path)

                    # 如果没有找到，尝试其他可能的路径
                    if not result['mesh_path']:
                        possible_paths = [
                            self.urdf_dir / package_path,
                            self.urdf_dir / relative_path,
                            self.urdf_dir.parent / package_path,
                            self.urdf_dir.parent / relative_path,
                            self.urdf_dir.parent / package_name / relative_path,
                        ]
                        for p in possible_paths:
                            if p and p.exists():
                                result['mesh_path'] = str(p)
                                break

                    if not result['mesh_path']:
                        # 最后使用默认路径（可能不存在）
                        result['mesh_path'] = str(self.urdf_dir / package_path)

                elif filename.startswith('file://'):
                    result['mesh_path'] = filename.replace('file://', '')
                else:
                    result['mesh_path'] = str(self.urdf_dir / filename)

            if mesh.get('scale'):
                result['scale'] = [float(x) for x in mesh.get('scale').split()]

        elif box is not None:
            result['type'] = 'box'
            result['size'] = [float(x) for x in box.get('size').split()]

        elif cylinder is not None:
            result['type'] = 'cylinder'
            result['size'] = {
                'radius': float(cylinder.get('radius')),
                'length': float(cylinder.get('length')),
            }

        elif sphere is not None:
            result['type'] = 'sphere'
            result['size'] = {'radius': float(sphere.get('radius'))}

        return result

    def get_base_link(self) -> str:
        """获取基座链接名"""
        child_links = set()
        for joint in self.joints.values():
            child_links.add(joint['child'])

        for link_name in self.links.keys():
            if link_name not in child_links:
                return link_name

        return list(self.links.keys())[0]

    def compute_link_transforms(self, joint_positions: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        计算所有链接的世界坐标变换矩阵

        Args:
            joint_positions: 关节角度字典 {joint_name: angle}

        Returns:
            {link_name: 4x4 transform matrix}
        """
        if joint_positions is None:
            joint_positions = {}

        transforms = {}
        base_link = self.get_base_link()
        transforms[base_link] = np.eye(4)

        # 构建父子关系
        children = {link: [] for link in self.links}
        for joint_name, joint in self.joints.items():
            children[joint['parent']].append((joint_name, joint['child']))

        # BFS 遍历计算变换
        queue = [base_link]
        visited = {base_link}

        while queue:
            parent_link = queue.pop(0)
            parent_transform = transforms[parent_link]

            for joint_name, child_link in children.get(parent_link, []):
                if child_link in visited:
                    continue

                joint = self.joints[joint_name]

                # 关节原点变换
                joint_transform = self._make_transform(joint['xyz'], joint['rpy'])

                # 关节旋转/平移
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

                # 计算子链接的变换
                transforms[child_link] = parent_transform @ joint_transform @ joint_motion

                visited.add(child_link)
                queue.append(child_link)

        return transforms

    def _make_transform(self, xyz: List[float], rpy: List[float]) -> np.ndarray:
        """从 xyz 和 rpy 创建 4x4 变换矩阵"""
        T = np.eye(4)

        # 平移
        T[:3, 3] = xyz

        # 旋转 (rpy: roll, pitch, yaw)
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        # ZYX 欧拉角旋转矩阵
        T[:3, :3] = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        return T

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """轴角转旋转矩阵（Rodrigues公式）"""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def get_all_meshes(self, joint_positions: Dict[str, float] = None) -> List[dict]:
        """
        获取所有链接的网格（已变换到世界坐标）

        Returns:
            List of {'vertices': (N,3), 'faces': (M,3), 'color': [r,g,b,a]}
        """
        link_transforms = self.compute_link_transforms(joint_positions)
        all_meshes = []

        for link_name, link_info in self.links.items():
            if link_name not in link_transforms:
                continue

            link_transform = link_transforms[link_name]

            for visual in link_info['visual']:
                # 计算 visual 的完整变换
                visual_local = self._make_transform(visual['xyz'], visual['rpy'])
                full_transform = link_transform @ visual_local

                mesh_data = None

                if visual['type'] == 'mesh' and visual['mesh_path']:
                    mesh_data = load_mesh_file(visual['mesh_path'], visual['scale'])

                elif visual['type'] == 'box' and visual['size']:
                    mesh_data = self._create_box_mesh(visual['size'])

                elif visual['type'] == 'cylinder' and visual['size']:
                    mesh_data = self._create_cylinder_mesh(
                        visual['size']['radius'],
                        visual['size']['length']
                    )

                elif visual['type'] == 'sphere' and visual['size']:
                    mesh_data = self._create_sphere_mesh(visual['size']['radius'])

                if mesh_data:
                    # 应用变换
                    transformed_vertices = transform_mesh(mesh_data['vertices'], full_transform)
                    all_meshes.append({
                        'vertices': transformed_vertices,
                        'faces': mesh_data['faces'],
                        'color': visual['color'],
                        'link_name': link_name,
                    })

        return all_meshes

    def _create_box_mesh(self, size: List[float]) -> dict:
        """创建长方体网格"""
        sx, sy, sz = [s/2 for s in size]
        vertices = np.array([
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        return {'vertices': vertices, 'faces': faces}

    def _create_cylinder_mesh(self, radius: float, length: float, segments: int = 16) -> dict:
        """创建圆柱体网格"""
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        h = length / 2

        # 顶面和底面顶点
        top_vertices = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(segments, h)])
        bottom_vertices = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(segments, -h)])

        # 中心点
        top_center = np.array([[0, 0, h]])
        bottom_center = np.array([[0, 0, -h]])

        vertices = np.vstack([top_vertices, bottom_vertices, top_center, bottom_center])

        faces = []
        top_center_idx = 2 * segments
        bottom_center_idx = 2 * segments + 1

        for i in range(segments):
            next_i = (i + 1) % segments
            # 侧面
            faces.append([i, next_i, segments + next_i])
            faces.append([i, segments + next_i, segments + i])
            # 顶面
            faces.append([top_center_idx, i, next_i])
            # 底面
            faces.append([bottom_center_idx, segments + next_i, segments + i])

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
                faces.append([idx, idx + 1, idx + segments])
                faces.append([idx + 1, idx + segments + 1, idx + segments])

        return {'vertices': vertices, 'faces': np.array(faces)}


# ===================================================================
# Plotly 可视化
# ===================================================================

def create_mesh_traces(
    meshes: List[dict],
    opacity: float = 0.8,
    name_prefix: str = "Robot"
) -> List[go.Mesh3d]:
    """从网格数据创建 Plotly traces"""
    traces = []

    for i, mesh in enumerate(meshes):
        vertices = mesh['vertices']
        faces = mesh['faces']
        color = mesh.get('color', [0.7, 0.7, 0.7, 1.0])

        # 转换颜色为 Plotly 格式
        r, g, b = [int(c * 255) for c in color[:3]]
        color_str = f'rgb({r},{g},{b})'

        trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color_str,
            opacity=opacity * color[3] if len(color) > 3 else opacity,
            name=f"{name_prefix}_{mesh.get('link_name', i)}",
            showlegend=False,
            flatshading=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.2,
                roughness=0.5,
            ),
            lightposition=dict(x=1, y=1, z=2),
        )
        traces.append(trace)

    return traces


def create_robot_skeleton_traces(urdf_loader: URDFMeshLoader, joint_positions: Dict[str, float] = None) -> List:
    """创建机械臂骨架 traces（备用方案）"""
    transforms = urdf_loader.compute_link_transforms(joint_positions)

    positions = []
    for link_name, transform in transforms.items():
        pos = transform[:3, 3]
        positions.append(pos)

    positions = np.array(positions)

    traces = []

    # 绘制连杆
    if len(positions) > 1:
        traces.append(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=5, color='darkblue'),
            name='Robot Skeleton',
        ))

    return traces


def visualize_reachability_with_robot(
    reachable_points_path: str,
    urdf_path: str = None,
    dexterity_path: str = None,
    joint_positions: Dict[str, float] = None,
    output_html: str = "reachability_with_robot.html",
    title: str = "Arm Reachability with Robot Model",
    robot_opacity: float = 0.7,
    package_paths: Dict[str, str] = None,
):
    """
    可视化可达空间和真实机械臂模型

    Args:
        reachable_points_path: 可达点文件路径 (.npy)
        urdf_path: URDF 文件路径（可选）
        dexterity_path: 灵巧性数据路径 (.npy)
        joint_positions: 关节角度字典（可选）
        output_html: 输出 HTML 文件路径
        title: 图表标题
        robot_opacity: 机器人模型透明度
        package_paths: ROS 包名到路径的映射，如 {"robot_model": "/path/to/robot_model"}
    """
    # 加载可达点
    points = np.load(reachable_points_path)
    print(f"[INFO] 加载可达点: {len(points)}")

    # 尝试加载灵巧性数据
    dexterity = None
    if dexterity_path is None:
        auto_path = reachable_points_path.replace("_reachable.npy", "_dexterity.npy")
        if Path(auto_path).exists():
            dexterity_path = auto_path

    if dexterity_path and Path(dexterity_path).exists():
        dexterity = np.load(dexterity_path)
        print(f"[INFO] 加载灵巧性数据: {len(dexterity)} 点")
        print(f"[INFO] 灵巧性范围: {dexterity.min()} - {dexterity.max()}")

    traces = []

    # 添加可达点云
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_sample = points[indices]
        dexterity_sample = dexterity[indices] if dexterity is not None else None
    else:
        points_sample = points
        dexterity_sample = dexterity

    # 根据是否有灵巧性数据选择颜色
    if dexterity_sample is not None:
        color_values = dexterity_sample
        colorbar_title = 'Dexterity<br>(# orientations)'
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
            size=2,
            color=color_values,
            colorscale=colorscale,
            opacity=0.6,
            colorbar=dict(title=colorbar_title, x=1.02),
        ),
        name=f'Reachable ({len(points)} pts)',
    ))

    # 加载并绘制机械臂模型
    if urdf_path and Path(urdf_path).exists():
        print(f"[INFO] 加载 URDF 模型: {urdf_path}")
        try:
            loader = URDFMeshLoader(urdf_path, package_paths=package_paths)
            meshes = loader.get_all_meshes(joint_positions)

            if meshes:
                print(f"[INFO] 加载了 {len(meshes)} 个网格")
                robot_traces = create_mesh_traces(meshes, opacity=robot_opacity)
                traces.extend(robot_traces)
            else:
                print("[WARN] 未能加载网格，使用骨架表示")
                skeleton_traces = create_robot_skeleton_traces(loader, joint_positions)
                traces.extend(skeleton_traces)

        except Exception as e:
            print(f"[WARN] URDF 加载失败: {e}")

    # 添加坐标轴
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

    # 创建图形
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

    # 保存
    fig.write_html(output_html)
    print(f"[SAVE] 已保存到: {output_html}")
    print("用浏览器打开查看（支持鼠标旋转、缩放、平移）")

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="可达性可视化 + 真实机械臂模型")
    parser.add_argument("data_path", type=str, nargs='?',
                        default="./reachability_output/robot_arm_reachable.npy",
                        help="可达点云文件")
    parser.add_argument("-u", "--urdf", type=str, default=None,
                        help="URDF 文件路径（显示真实机器人模型）")
    parser.add_argument("-d", "--dexterity", type=str, default=None,
                        help="灵巧性数据文件 (可选，默认自动查找)")
    parser.add_argument("-o", "--output", type=str, default="reachability_with_robot.html",
                        help="输出 HTML 文件")
    parser.add_argument("-t", "--title", type=str, default="Arm Reachability & Dexterity",
                        help="图表标题")
    parser.add_argument("--robot-opacity", type=float, default=0.7,
                        help="机器人模型透明度 (0-1)")
    parser.add_argument("--mesh-dir", "-m", type=str, default=None,
                        help="网格文件目录路径 (用于解析 package:// 路径)")
    parser.add_argument("--package", "-p", type=str, action='append', default=[],
                        help="ROS 包映射，格式: package_name=/path/to/package (可多次使用)")

    args = parser.parse_args()

    # 解析包路径映射
    package_paths = {}
    for pkg_mapping in args.package:
        if '=' in pkg_mapping:
            name, path = pkg_mapping.split('=', 1)
            package_paths[name] = path
            print(f"[INFO] 包路径映射: {name} -> {path}")

    # 如果指定了 mesh-dir，将其作为默认包路径
    if args.mesh_dir:
        mesh_dir = Path(args.mesh_dir)
        if mesh_dir.exists():
            # 使用目录名作为默认包名
            package_paths[mesh_dir.name] = str(mesh_dir)
            print(f"[INFO] 使用网格目录: {mesh_dir}")

    visualize_reachability_with_robot(
        args.data_path,
        urdf_path=args.urdf,
        dexterity_path=args.dexterity,
        output_html=args.output,
        title=args.title,
        robot_opacity=args.robot_opacity,
        package_paths=package_paths if package_paths else None,
    )


if __name__ == "__main__":
    main()
