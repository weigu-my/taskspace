#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
可达性分析结果可视化工具
===========================================================

支持功能：
1. 3D 点云可视化
2. 多视角切片图
3. 可达率热力图
4. 对比分析（多臂/多配置）
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_reachability_data(
    data_path: Union[str, Path],
) -> np.ndarray:
    """加载可达性点云数据"""
    data_path = Path(data_path)

    if data_path.suffix == '.npy':
        return np.load(data_path)
    elif data_path.suffix == '.csv':
        import pandas as pd
        return pd.read_csv(data_path).values
    else:
        raise ValueError(f"不支持的文件格式: {data_path.suffix}")


def plot_3d_scatter(
    points: np.ndarray,
    title: str = "Reachability Point Cloud",
    color: str = 'blue',
    alpha: float = 0.3,
    size: float = 1,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """绘制 3D 散点图"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if len(points) > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            alpha=alpha,
            s=size,
        )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # 设置等比例坐标轴
    if len(points) > 0:
        max_range = np.max([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]) / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
        mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
        mid_z = (points[:, 2].max() + points[:, 2].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] 图片已保存: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_multi_view(
    points: np.ndarray,
    title: str = "Reachability Multi-View",
    color: str = 'blue',
    alpha: float = 0.3,
    size: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """绘制多视角投影图（XY, XZ, YZ, 3D）"""
    fig = plt.figure(figsize=(14, 10))

    # XY 平面 (俯视图)
    ax1 = fig.add_subplot(2, 2, 1)
    if len(points) > 0:
        ax1.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=size)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # XZ 平面 (前视图)
    ax2 = fig.add_subplot(2, 2, 2)
    if len(points) > 0:
        ax2.scatter(points[:, 0], points[:, 2], c=color, alpha=alpha, s=size)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Front View (XZ)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # YZ 平面 (侧视图)
    ax3 = fig.add_subplot(2, 2, 3)
    if len(points) > 0:
        ax3.scatter(points[:, 1], points[:, 2], c=color, alpha=alpha, s=size)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (YZ)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)

    # 3D 视图
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    if len(points) > 0:
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=color, alpha=alpha, s=size)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D View')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] 图片已保存: {save_path}")

    if show:
        plt.show()

    return fig


def plot_slice_heatmap(
    points: np.ndarray,
    voxel_size: float = 0.05,
    slice_axis: str = 'z',
    slice_values: Optional[list] = None,
    num_slices: int = 6,
    title: str = "Reachability Slice Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """绘制切片热力图"""
    if len(points) == 0:
        print("[WARN] 无数据点，跳过热力图绘制")
        return None

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(slice_axis.lower(), 2)
    other_axes = [i for i in range(3) if i != axis_idx]

    # 确定切片位置
    if slice_values is None:
        min_val = points[:, axis_idx].min()
        max_val = points[:, axis_idx].max()
        slice_values = np.linspace(min_val, max_val, num_slices + 2)[1:-1]

    n_slices = len(slice_values)
    cols = min(3, n_slices)
    rows = (n_slices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_slices == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    axis_labels = ['X', 'Y', 'Z']

    for i, slice_val in enumerate(slice_values):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        # 选择在切片范围内的点
        mask = np.abs(points[:, axis_idx] - slice_val) < voxel_size
        slice_points = points[mask]

        if len(slice_points) > 0:
            ax.scatter(
                slice_points[:, other_axes[0]],
                slice_points[:, other_axes[1]],
                c='blue',
                alpha=0.5,
                s=2,
            )

        ax.set_xlabel(f'{axis_labels[other_axes[0]]} (m)')
        ax.set_ylabel(f'{axis_labels[other_axes[1]]} (m)')
        ax.set_title(f'{axis_labels[axis_idx]} = {slice_val:.2f}m (n={len(slice_points)})')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_slices, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] 图片已保存: {save_path}")

    if show:
        plt.show()

    return fig


def plot_comparison(
    data_dict: dict[str, np.ndarray],
    title: str = "Reachability Comparison",
    colors: Optional[dict] = None,
    alpha: float = 0.4,
    size: float = 1,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """对比多组可达性数据"""
    if colors is None:
        color_cycle = plt.cm.tab10.colors
        colors = {name: color_cycle[i % 10] for i, name in enumerate(data_dict.keys())}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for name, points in data_dict.items():
        if len(points) > 0:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=[colors.get(name, 'gray')],
                alpha=alpha,
                s=size,
                label=f'{name} (n={len(points)})',
            )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] 图片已保存: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_reachability_volume(
    points: np.ndarray,
    voxel_size: float = 0.05,
    title: str = "Reachability Volume Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """分析和绘制可达空间的体积分布"""
    if len(points) == 0:
        print("[WARN] 无数据点")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # X 方向分布
    axes[0].hist(points[:, 0], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('X Distribution')
    axes[0].grid(True, alpha=0.3)

    # Y 方向分布
    axes[1].hist(points[:, 1], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Y (m)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Y Distribution')
    axes[1].grid(True, alpha=0.3)

    # Z 方向分布
    axes[2].hist(points[:, 2], bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Z (m)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Z Distribution')
    axes[2].grid(True, alpha=0.3)

    # 计算统计信息
    volume = len(points) * (voxel_size ** 3)

    fig.suptitle(f'{title}\nTotal Points: {len(points)}, Estimated Volume: {volume:.4f} m³',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] 图片已保存: {save_path}")

    if show:
        plt.show()

    return fig


def generate_report(
    output_dir: str,
    arm_names: list[str],
    voxel_size: float = 0.05,
):
    """生成完整的可达性分析报告"""
    output_dir = Path(output_dir)

    for arm_name in arm_names:
        npy_path = output_dir / f"{arm_name}_reachable.npy"

        if not npy_path.exists():
            print(f"[WARN] 未找到数据文件: {npy_path}")
            continue

        print(f"\n{'='*60}")
        print(f"生成报告: {arm_name}")
        print('='*60)

        points = load_reachability_data(npy_path)
        print(f"加载点数: {len(points)}")

        # 生成各种可视化
        plot_3d_scatter(
            points,
            title=f"{arm_name} Reachability 3D",
            save_path=str(output_dir / f"{arm_name}_3d.png"),
            show=False,
        )
        plt.close()

        plot_multi_view(
            points,
            title=f"{arm_name} Reachability Views",
            save_path=str(output_dir / f"{arm_name}_views.png"),
            show=False,
        )
        plt.close()

        plot_slice_heatmap(
            points,
            voxel_size=voxel_size,
            slice_axis='z',
            title=f"{arm_name} Z-Slices",
            save_path=str(output_dir / f"{arm_name}_z_slices.png"),
            show=False,
        )
        plt.close()

        plot_reachability_volume(
            points,
            voxel_size=voxel_size,
            title=f"{arm_name} Volume Analysis",
            save_path=str(output_dir / f"{arm_name}_volume.png"),
            show=False,
        )
        plt.close()

    # 对比图（如果有多个臂）
    if len(arm_names) > 1:
        data_dict = {}
        for arm_name in arm_names:
            npy_path = output_dir / f"{arm_name}_reachable.npy"
            if npy_path.exists():
                data_dict[arm_name] = load_reachability_data(npy_path)

        if len(data_dict) > 1:
            plot_comparison(
                data_dict,
                title="Multi-Arm Reachability Comparison",
                save_path=str(output_dir / "comparison.png"),
                show=False,
            )
            plt.close()

    print(f"\n[DONE] 报告已生成到: {output_dir}")


# ===================================================================
# 命令行入口
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="可达性分析结果可视化工具")
    parser.add_argument("data_path", type=str, help="可达性数据文件 (.npy 或 .csv)")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出图片路径")
    parser.add_argument("--mode", "-m", type=str, default="3d",
                        choices=["3d", "multi", "slice", "volume"],
                        help="可视化模式")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素大小")

    args = parser.parse_args()

    # 加载数据
    points = load_reachability_data(args.data_path)
    print(f"加载点数: {len(points)}")

    # 根据模式绘图
    if args.mode == "3d":
        plot_3d_scatter(points, save_path=args.output)
    elif args.mode == "multi":
        plot_multi_view(points, save_path=args.output)
    elif args.mode == "slice":
        plot_slice_heatmap(points, voxel_size=args.voxel_size, save_path=args.output)
    elif args.mode == "volume":
        plot_reachability_volume(points, voxel_size=args.voxel_size, save_path=args.output)


if __name__ == "__main__":
    main()
