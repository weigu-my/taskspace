"""
从 URDF 的 collision mesh 自动生成 cuRobo 碰撞球配置

使用 trimesh 加载 STL/DAE 网格，通过体素化 + 球覆盖生成包络球，
导出为 cuRobo 兼容的 YAML 配置文件。

用法:
    python3 -m arm_reachability.collision_spheres_generator \
        --urdf /path/to/robot.urdf \
        --output collision_spheres.yaml \
        --max-spheres-per-link 8
"""

import os
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LinkCollisionInfo:
    """单个 link 的碰撞信息"""
    name: str
    mesh_path: str = ""
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))


def resolve_package_path(mesh_filename: str, urdf_dir: str) -> str:
    """解析 package:// 路径为绝对路径"""
    if mesh_filename.startswith("package://"):
        # package://robot_model/meshes/xxx.stl
        # 假设 urdf 在 package_name/urdf/ 目录下
        rel_path = mesh_filename.replace("package://", "")
        package_name = rel_path.split("/")[0]
        rest_path = "/".join(rel_path.split("/")[1:])
        # 尝试从 urdf_dir 的父目录找 package
        package_dir = os.path.dirname(urdf_dir)  # 上一级 = package root
        full_path = os.path.join(package_dir, rest_path)
        if os.path.exists(full_path):
            return full_path
        # 尝试从 urdf_dir 的父父目录
        full_path2 = os.path.join(os.path.dirname(package_dir), rel_path)
        if os.path.exists(full_path2):
            return full_path2
    elif not os.path.isabs(mesh_filename):
        return os.path.join(urdf_dir, mesh_filename)
    return mesh_filename


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """RPY 欧拉角转旋转矩阵"""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def parse_urdf_collision_meshes(urdf_path: str) -> Dict[str, LinkCollisionInfo]:
    """从 URDF 解析所有 link 的 collision mesh 信息"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name", "")
        collision = link_elem.find("collision")
        if collision is None:
            continue

        geometry = collision.find("geometry")
        if geometry is None:
            continue

        mesh_elem = geometry.find("mesh")
        if mesh_elem is None:
            continue

        filename = mesh_elem.get("filename", "")
        mesh_path = resolve_package_path(filename, urdf_dir)

        # 解析 origin
        origin_xyz = np.zeros(3)
        origin_rpy = np.zeros(3)
        origin = collision.find("origin")
        if origin is not None:
            xyz_str = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            origin_xyz = np.array([float(x) for x in xyz_str.split()])
            origin_rpy = np.array([float(x) for x in rpy_str.split()])

        # 解析 scale
        scale = np.ones(3)
        scale_str = mesh_elem.get("scale", "")
        if scale_str:
            scale = np.array([float(x) for x in scale_str.split()])

        links[name] = LinkCollisionInfo(
            name=name,
            mesh_path=mesh_path,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            scale=scale,
        )

    return links


def generate_spheres_from_mesh(
    mesh_path: str,
    origin_xyz: np.ndarray,
    origin_rpy: np.ndarray,
    scale: np.ndarray,
    max_spheres: int = 8,
    min_radius: float = 0.01,
) -> List[Dict]:
    """
    从 mesh 文件生成碰撞球

    方法: 加载 mesh -> 应用 collision origin 变换 -> 使用 K-means 聚类
    将顶点分组 -> 每组用一个球包围

    参数:
        mesh_path: STL/DAE 文件路径
        origin_xyz: collision origin 位移
        origin_rpy: collision origin 旋转
        scale: mesh 缩放
        max_spheres: 每个 link 最大球数
        min_radius: 最小球半径
    """
    import trimesh

    if not os.path.exists(mesh_path):
        print(f"  [WARN] mesh 不存在: {mesh_path}")
        return [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]

    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        print(f"  [WARN] 加载 mesh 失败: {mesh_path}: {e}")
        return [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]

    # 应用缩放
    vertices = mesh.vertices * scale

    # 应用 collision origin 变换（旋转 + 平移）
    R = rpy_to_matrix(origin_rpy)
    vertices = (R @ vertices.T).T + origin_xyz

    if len(vertices) < 3:
        return [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]

    # 计算 mesh 整体尺寸
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_dim = bbox_size.max()

    # 对于非常小的 link，使用单个球
    if max_dim < 0.03:
        center = (bbox_min + bbox_max) / 2
        radius = max(max_dim / 2, min_radius)
        return [{"center": center.tolist(), "radius": float(radius)}]

    # 根据 mesh 大小决定球数量
    n_spheres = min(max_spheres, max(1, int(max_dim / 0.05)))

    if n_spheres == 1:
        center = (bbox_min + bbox_max) / 2
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        return [{"center": center.tolist(), "radius": max(radius, min_radius)}]

    # 使用表面采样点进行 K-means 聚类
    try:
        sample_points, _ = trimesh.sample.sample_surface(mesh, count=min(2000, len(vertices) * 3))
        sample_points = sample_points * scale
        sample_points = (R @ sample_points.T).T + origin_xyz
    except Exception:
        sample_points = vertices

    # 简单 K-means
    centers, labels = _kmeans(sample_points, n_spheres, max_iter=30)

    spheres = []
    for i in range(n_spheres):
        cluster_mask = labels == i
        if not cluster_mask.any():
            continue
        cluster_pts = sample_points[cluster_mask]
        center = centers[i]
        # 球半径 = 最远点到中心的距离 + padding
        distances = np.linalg.norm(cluster_pts - center, axis=1)
        radius = float(distances.max()) + 0.005  # 5mm padding
        radius = max(radius, min_radius)
        spheres.append({
            "center": [round(float(c), 5) for c in center],
            "radius": round(radius, 4),
        })

    return spheres if spheres else [{"center": [0.0, 0.0, 0.0], "radius": min_radius}]


def _kmeans(points: np.ndarray, k: int, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """简单 K-means 聚类"""
    n = len(points)
    if n <= k:
        return points.copy(), np.arange(n)

    # 随机初始化（使用 k-means++ 风格）
    rng = np.random.default_rng(42)
    centers = np.zeros((k, 3))
    centers[0] = points[rng.integers(n)]
    for i in range(1, k):
        dists = np.min([np.linalg.norm(points - centers[j], axis=1) for j in range(i)], axis=0)
        probs = dists / dists.sum()
        centers[i] = points[rng.choice(n, p=probs)]

    for _ in range(max_iter):
        # 分配
        dists = np.stack([np.linalg.norm(points - c, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        # 更新中心
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = points[mask].mean(axis=0)
            else:
                new_centers[i] = centers[i]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return centers, labels


def parse_urdf_joints(urdf_path: str) -> List[Tuple[str, str]]:
    """解析 URDF joint 的 parent-child 关系"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            joints.append((parent.get("link", ""), child.get("link", "")))
    return joints


def generate_self_collision_ignore(
    collision_spheres: Dict[str, List],
    joints: List[Tuple[str, str]],
    skip_distance: int = 1,
) -> Dict[str, List[str]]:
    """
    生成自碰撞忽略列表

    只忽略直接相邻的 link 对（距离=1），因为有了真实的碰撞球包络，
    不需要像简单球那样忽略更远的 link。

    参数:
        collision_spheres: link -> spheres 映射
        joints: (parent, child) 列表
        skip_distance: 忽略的 link 距离（1=仅直接相邻）
    """
    sphere_links = set(collision_spheres.keys())
    adjacency = {}
    for parent, child in joints:
        if parent in sphere_links and child in sphere_links:
            adjacency.setdefault(parent, set()).add(child)
            adjacency.setdefault(child, set()).add(parent)

    # BFS 扩展到 skip_distance
    ignore = {}
    for link in sphere_links:
        to_ignore = set()
        visited = {link}
        frontier = {link}
        for _ in range(skip_distance):
            next_frontier = set()
            for f in frontier:
                next_frontier |= adjacency.get(f, set())
            next_frontier -= visited
            to_ignore |= next_frontier
            visited |= next_frontier
            frontier = next_frontier
        to_ignore -= {link}
        if to_ignore:
            ignore[link] = sorted(to_ignore)

    return ignore


def generate_collision_spheres_yaml(
    urdf_path: str,
    output_path: str = None,
    max_spheres_per_link: int = 8,
    include_links: Optional[List[str]] = None,
    exclude_links: Optional[List[str]] = None,
) -> Dict:
    """
    从 URDF mesh 生成碰撞球配置并保存为 YAML

    参数:
        urdf_path: URDF 文件路径
        output_path: 输出 YAML 路径（None 则不保存）
        max_spheres_per_link: 每个 link 的最大球数
        include_links: 只包含这些 link（None=全部）
        exclude_links: 排除这些 link

    返回:
        碰撞球配置字典
    """
    print(f"从 URDF mesh 生成碰撞球配置...")
    print(f"  URDF: {urdf_path}")
    print(f"  最大球数/link: {max_spheres_per_link}")

    # 解析 URDF
    link_infos = parse_urdf_collision_meshes(urdf_path)
    joints = parse_urdf_joints(urdf_path)

    collision_spheres = {}
    total_spheres = 0

    for link_name, info in link_infos.items():
        if include_links and link_name not in include_links:
            continue
        if exclude_links and link_name in exclude_links:
            continue

        print(f"  处理 {link_name}...", end="")
        spheres = generate_spheres_from_mesh(
            mesh_path=info.mesh_path,
            origin_xyz=info.origin_xyz,
            origin_rpy=info.origin_rpy,
            scale=info.scale,
            max_spheres=max_spheres_per_link,
        )
        collision_spheres[link_name] = spheres
        total_spheres += len(spheres)
        print(f" {len(spheres)} 个球")

    # 生成自碰撞忽略（忽略距离 ≤2 的相邻 link，避免 mesh 碰撞球过大导致误报）
    self_collision_ignore = generate_self_collision_ignore(
        collision_spheres, joints, skip_distance=2
    )

    config = {
        "collision_spheres": collision_spheres,
        "self_collision_ignore": self_collision_ignore,
        "self_collision_buffer": {},
    }

    print(f"\n碰撞球生成完成:")
    print(f"  Link 数: {len(collision_spheres)}")
    print(f"  总球数: {total_spheres}")
    print(f"  自碰撞忽略对: {sum(len(v) for v in self_collision_ignore.values())}")

    if output_path:
        # 转换 numpy 类型为 Python 原生类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            return convert(obj)

        with open(output_path, 'w') as f:
            yaml.dump(deep_convert(config), f, default_flow_style=False, sort_keys=False)
        print(f"  已保存: {output_path}")

    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从 URDF mesh 生成 cuRobo 碰撞球配置")
    parser.add_argument("--urdf", required=True, help="URDF 文件路径")
    parser.add_argument("--output", default="collision_spheres.yaml", help="输出 YAML 路径")
    parser.add_argument("--max-spheres", type=int, default=8, help="每个 link 最大球数")
    args = parser.parse_args()

    generate_collision_spheres_yaml(
        urdf_path=args.urdf,
        output_path=args.output,
        max_spheres_per_link=args.max_spheres,
    )
