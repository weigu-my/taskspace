#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct visualizer for reachability .npy results.
No arguments needed.

- Always saves:
  workspace_3d.png
  workspace_views.png

- Additionally (NEW):
  Starts a Meshcat web viewer for interactive rotation/zoom:
    http://localhost:7000

Note:
- In headless docker, matplotlib plt.show() won't pop up windows.
  We keep saving PNGs and skip interactive show automatically.
- For Meshcat in docker: make sure your container exposes port 7000, e.g.
  docker run ... -p 7000:7000 ...
"""

import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =========================
# Config (直接改这里也行)
# =========================
DATA_DIR = "/anyverse/taskspace/ik_reachability_out"
LEFT_NPY = os.path.join(DATA_DIR, "left_reachable.npy")
RIGHT_NPY = os.path.join(DATA_DIR, "right_reachable.npy")

# 显示优化关键：体素下采样（2cm 通常能显著消除“摩尔纹/放射纹”）
DOWNSAMPLE_VOXEL = 0.02   # meters; 0 disables

# 防止点太多导致渲染很慢
MAX_POINTS_PER_ARM = 80000  # <=0 disables
RANDOM_SEED = 0

# 3D 视角（Matplotlib）
VIEW_ELEV = 20
VIEW_AZIM = -60

# 输出图片
SAVE_3D = os.path.join(DATA_DIR, "workspace_3d.png")
SAVE_VIEWS = os.path.join(DATA_DIR, "workspace_views.png")

# --------- Meshcat (NEW) ---------
ENABLE_MESHCAT = True
MESHCAT_PORT = 7000         # docker 里注意映射 -p 7000:7000
MESHCAT_POINT_SIZE = 0.006  # 点越小越清爽：0.003~0.01
KEEP_MESHCAT_ALIVE = True   # True: 让脚本一直挂着方便你在浏览器看；Ctrl+C 退出
# 如果你不想一直占着终端，可设为 False（但脚本退出后 Meshcat 服务也会停）


# =========================
# Helpers
# =========================
def load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    pts = np.load(path)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{path} should have shape (N,3), got {pts.shape}")
    return pts


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """
    体素下采样：把落在同一个 voxel 的点合并（保留第一个点）。
    voxel<=0 表示不下采样。
    """
    if voxel <= 0 or len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int32)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]


def random_subsample(pts: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """
    随机抽样：限制最大点数，避免渲染卡顿。
    max_points<=0 表示不抽样。
    """
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[idx]


def set_equal_aspect_3d(ax, xyz: np.ndarray):
    """
    Matplotlib 3D 轴等比例（否则会视觉拉伸，看起来像“椭球被压扁/拉长”）。
    """
    if xyz is None or len(xyz) == 0:
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    spans = (maxs - mins)
    radius = float(spans.max() / 2.0) if spans.max() > 0 else 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_3d(left: np.ndarray, right: np.ndarray, elev: float, azim: float):
    """
    3D 点云图：左/右臂分别散点显示。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 点大小：建议小一点更干净
    s = 1.0

    if len(left) > 0:
        ax.scatter(left[:, 0], left[:, 1], left[:, 2], s=s, marker=".", alpha=0.9, label="Left")
    if len(right) > 0:
        ax.scatter(right[:, 0], right[:, 1], right[:, 2], s=s, marker=".", alpha=0.9, label="Right")

    if len(left) + len(right) > 0:
        all_pts = np.vstack([p for p in [left, right] if len(p) > 0])
        set_equal_aspect_3d(ax, all_pts)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Reachability Workspace (Dual Arms)")

    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig


def plot_views(left: np.ndarray, right: np.ndarray):
    """
    三视图：XY / XZ / YZ
    给 mentor 发图更稳、更容易对比范围。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def draw(ax, x, y, xlabel, ylabel):
        if len(left) > 0:
            ax.scatter(left[:, x], left[:, y], s=1.0, marker=".", alpha=0.9, label="Left")
        if len(right) > 0:
            ax.scatter(right[:, x], right[:, y], s=1.0, marker=".", alpha=0.9, label="Right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

    draw(axes[0], 0, 1, "X (m)", "Y (m)")
    axes[0].set_title("XY (top view)")

    draw(axes[1], 0, 2, "X (m)", "Z (m)")
    axes[1].set_title("XZ (front view)")

    draw(axes[2], 1, 2, "Y (m)", "Z (m)")
    axes[2].set_title("YZ (side view)")

    axes[0].legend(loc="best")
    fig.suptitle("Reachability Views (Dual Arms)")
    fig.tight_layout()
    return fig


# -------------------------
# Meshcat Visualization (NEW)
# -------------------------
def meshcat_show_pointcloud(left: np.ndarray, right: np.ndarray, port: int = 7000, point_size: float = 0.006):
    """
    启动 Meshcat 并显示左右臂点云（可旋转缩放）。
    注意：脚本退出后 Meshcat 服务器会停止，所以一般需要 KEEP_MESHCAT_ALIVE=True 挂着。
    """
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
    except Exception as e:
        print("[WARN] Meshcat not available. Install it via: pip install meshcat")
        print("       Error:", repr(e))
        return None

    # 让 Meshcat 绑定 0.0.0.0，方便 docker 宿主机访问（端口映射后）
    vis = meshcat.Visualizer(zmq_url=f"tcp://0.0.0.0:{port}")

    print("[INFO] Meshcat started. Open in browser:")
    print(f"       http://localhost:{port}")
    print("       (If running inside docker: ensure -p 7000:7000 or use the mapped port.)")

    # 清理旧对象，避免多次运行叠加
    for name in ["reach_left", "reach_right", "axes"]:
        try:
            vis[name].delete()
        except Exception:
            pass

    # 坐标轴
    vis["axes"].set_object(g.Axes(length=0.5))

    # 点云：Meshcat 的 PointCloud 需要 3×N float32
    def solid_color(n: int, rgb):
        c = np.zeros((3, n), dtype=np.float32)
        c[0, :] = rgb[0]
        c[1, :] = rgb[1]
        c[2, :] = rgb[2]
        return c

    if len(left) > 0:
        pts = left.T.astype(np.float32)
        colors = solid_color(pts.shape[1], (0.0, 0.2, 1.0))  # Left: blue
        vis["reach_left"].set_object(g.PointCloud(position=pts, color=colors, size=point_size))

    if len(right) > 0:
        pts = right.T.astype(np.float32)
        colors = solid_color(pts.shape[1], (1.0, 0.2, 0.0))  # Right: orange/red
        vis["reach_right"].set_object(g.PointCloud(position=pts, color=colors, size=point_size))

    # 简单设置相机视角（可选）
    # 注意：不同 meshcat 版本相机节点名字略有差异，失败也不影响显示
    try:
        cam_T = tf.translation_matrix([2.5, 0.0, 1.5]) @ tf.euler_matrix(0.0, -0.5, 1.57)
        vis["/Cameras/default"].set_transform(cam_T)
    except Exception:
        pass

    return vis


# =========================
# Main
# =========================
def main():
    print("[INFO] Loading:")
    print("  left :", LEFT_NPY)
    print("  right:", RIGHT_NPY)

    left = load_npy(LEFT_NPY)
    right = load_npy(RIGHT_NPY)

    print(f"[INFO] Raw points: left={len(left)}, right={len(right)}")

    # 体素下采样（显示优化核心）
    left = voxel_downsample(left, DOWNSAMPLE_VOXEL)
    right = voxel_downsample(right, DOWNSAMPLE_VOXEL)
    print(f"[INFO] After voxel downsample({DOWNSAMPLE_VOXEL}m): left={len(left)}, right={len(right)}")

    # 限制最大点数（防止渲染太慢）
    left = random_subsample(left, MAX_POINTS_PER_ARM, seed=RANDOM_SEED)
    right = random_subsample(right, MAX_POINTS_PER_ARM, seed=RANDOM_SEED + 1)
    print(f"[INFO] After random cap({MAX_POINTS_PER_ARM}/arm): left={len(left)}, right={len(right)}")

    # ---- Matplotlib: 保存 PNG ----
    os.makedirs(DATA_DIR, exist_ok=True)

    fig3d = plot_3d(left, right, elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig3d.savefig(SAVE_3D, dpi=300, bbox_inches="tight")
    print(f"[SAVE] 3D figure -> {SAVE_3D}")

    figv = plot_views(left, right)
    figv.savefig(SAVE_VIEWS, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Views figure -> {SAVE_VIEWS}")

    # 在 headless 环境不强行 show（避免你之前看到的 FigureCanvasAgg warning）
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        print(f"[INFO] Matplotlib backend='{matplotlib.get_backend()}' is non-interactive; skip plt.show().")

    # ---- Meshcat: 可交互浏览器显示 ----
    vis = None
    if ENABLE_MESHCAT:
        vis = meshcat_show_pointcloud(left, right, port=MESHCAT_PORT, point_size=MESHCAT_POINT_SIZE)

    if ENABLE_MESHCAT and KEEP_MESHCAT_ALIVE and vis is not None:
        print("[INFO] Meshcat is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
    else:
        # 不 keep alive 的话，脚本结束 Meshcat 就停了
        if ENABLE_MESHCAT:
            print("[INFO] KEEP_MESHCAT_ALIVE=False, program will exit and Meshcat will stop.")


if __name__ == "__main__":
    main()
