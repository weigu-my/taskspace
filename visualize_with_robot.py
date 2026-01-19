#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
可达性结果 + 机器人模型可视化（Plotly）
===========================================================

功能:
1. 渲染 URDF 中的机器人模型
2. 可达点云可视化
3. 点大小 = manipulability，点颜色 = dexterity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import importlib.util
import numpy as np

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
YOURDFPY_AVAILABLE = importlib.util.find_spec("yourdfpy") is not None
TRIMESH_AVAILABLE = importlib.util.find_spec("trimesh") is not None

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
if YOURDFPY_AVAILABLE:
    from yourdfpy import URDF
if TRIMESH_AVAILABLE:
    import trimesh


def _normalize_visual_fk(data) -> Iterable[Tuple[object, np.ndarray]]:
    if isinstance(data, dict):
        for mesh, transform in data.items():
            yield mesh, transform
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                yield item[0], item[1]


def load_robot_meshes(urdf_path: str) -> list[trimesh.Trimesh]:
    if not (YOURDFPY_AVAILABLE and TRIMESH_AVAILABLE):
        return []

    robot = URDF.load(urdf_path)
    meshes = []

    if hasattr(robot, "visual_trimesh_fk"):
        visual_fk = robot.visual_trimesh_fk()
        for mesh, transform in _normalize_visual_fk(visual_fk):
            mesh_copy = mesh.copy()
            if transform is not None:
                mesh_copy.apply_transform(transform)
            meshes.append(mesh_copy)

    return meshes


def load_metrics(path: str) -> np.ndarray:
    data_path = Path(path)
    if data_path.suffix == ".npy":
        return np.load(data_path)

    return np.loadtxt(data_path, delimiter=",", skiprows=1)


def build_plot(robot_meshes: list[trimesh.Trimesh], metrics: np.ndarray, output_html: str | None):
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("未安装 plotly")

    fig = go.Figure()

    for mesh in robot_meshes:
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="lightgray",
                opacity=0.6,
                name="robot",
                showscale=False,
            )
        )

    points = metrics[:, :3]
    dexterity = metrics[:, 3]
    manipulability = metrics[:, 5]

    min_size = 3
    max_size = 14
    if len(manipulability) > 0:
        m_min = float(np.min(manipulability))
        m_max = float(np.max(manipulability))
        scale = (manipulability - m_min) / (m_max - m_min + 1e-9)
        sizes = min_size + scale * (max_size - min_size)
    else:
        sizes = np.full(len(points), min_size)

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=dexterity,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Dexterity"),
            ),
            name="reachability",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        title="Reachability + Dexterity + Manipulability",
    )

    if output_html:
        fig.write_html(output_html)
        print(f"[SAVE] 可视化已保存: {output_html}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="可达性结果可视化")
    parser.add_argument("--urdf", required=True, help="URDF 文件")
    parser.add_argument("--metrics", required=True, help="可达性结果 CSV/NPY")
    parser.add_argument("--output", default=None, help="输出 HTML 文件")

    args = parser.parse_args()

    meshes = load_robot_meshes(args.urdf)
    metrics = load_metrics(args.metrics)

    build_plot(meshes, metrics, args.output)


if __name__ == "__main__":
    main()
