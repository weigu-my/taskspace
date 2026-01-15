#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
Dual-arm Reachability (Pose-conditioned, Axis-only) via IK (Pinocchio)
======================================================================

【目标】
在“给定全身参考姿态 q_ref + 末端工具轴方向约束（roll 自由）”条件下，
计算左右手末端“可达的位置空间”（position workspace slice）。

与“蒙特卡洛 + FK”不同：
- FK蒙特卡洛：采样关节 q -> FK 得到 (p, R)，得到的是“所有姿态混在一起”的位置包络。
- 本脚本（IK扫描）：固定方向约束（例如手掌法向朝下）-> 扫 p -> 做 IK 判定 -> 得到“方向切片”的可达空间。

【关键改进点】
(1) 方向约束从“完整 R_d”改为“仅工具轴方向 a_d”
    - 这会让末端绕工具轴的 roll 角自由 -> 可达区域更大，IK更容易收敛。
(2) 扫描顺序采用“空间连续（蛇形）遍历”
    - continuation：上一点的解 q_prev 作为下一点初值 q0，在空间连续时非常有效。
(3) 粗到细：先粗体素 VOXEL_COARSE 扫描，再对粗扫可达体素细分成 VOXEL_FINE 再扫
    - 速度通常提升一个量级。
(4) FK warm-start 初值库：
    - 先对每只臂随机采样关节做FK，筛选出工具轴方向接近目标方向的姿态，把对应 q 存入体素哈希。
    - IK扫某个体素时，优先用这些 q 当初值，比随机初值强很多。

【输出】
- left_reachable_coarse.csv / .npy
- right_reachable_coarse.csv / .npy
- left_reachable_fine.csv / .npy
- right_reachable_fine.csv / .npy
（可选）Meshcat 中显示机器人与点云

依赖：
pip install pin meshcat numpy pandas

----------------------------------------------------------------------
如果你 mentor 的“给定姿态”其实是“全身某个姿态”，只需把 q_ref 换成那组关节角即可。
如果 mentor 的“给定姿态”是“末端法向朝下”，改 TARGET_AXIS_WORLD 即可。
----------------------------------------------------------------------

"""

import os
import time
import math
import numpy as np
import pandas as pd
import pinocchio as pin

# ========== 可选 Meshcat 可视化 ==========
USE_MESHCAT = True
try:
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as g
except Exception:
    USE_MESHCAT = False


# ===========================================================
# 1) 用户配置区（优先只改这里）
# ===========================================================

# ---- URDF ----
URDF_PATH = "/anyverse/sub_modules/isaacros/src/wheeled_humanoid/robot_model/urdf/wheel_robot.urdf"

# Root joint：固定底座一般 None；浮动底座用 pin.JointModelFreeFlyer()
ROOT_JOINT = None

# ---- 末端 frame 选择 ----
# 优先：如果你已经知道准确 frame 名，可以直接填死（最稳定）
LEFT_EE_NAME = "left_gripper_base_joint"    # e.g. "left_gripper"
RIGHT_EE_NAME = "right_gripper_base_joint"   # e.g. "right_gripper"

# 否则：用关键词自动匹配（找不到会打印候选）
LEFT_EE_KEYWORDS  = ["left_gripper", "left_hand", "l_gripper", "l_hand", "left_tip", "left_wrist"]
RIGHT_EE_KEYWORDS = ["right_gripper", "right_hand", "r_gripper", "r_hand", "right_tip", "right_wrist"]

# ---- “给定姿态”定义：工具轴方向约束（世界坐标系） ----
# 你需要明确“工具轴”在末端坐标系里是哪根轴：常见取 z 轴 [0,0,1]
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# 目标工具轴方向（世界系），例如“手掌法向朝下”：世界 -Z
TARGET_AXIS_WORLD = np.array([0.0, 0.0, -1.0])

# 方向误差容差：工具轴与目标方向夹角 < 这个阈值（用于判定成功）
AXIS_ANGLE_TOL = np.deg2rad(8.0)   # 8度（你可改 5~15 度）

# ---- IK 参数（阻尼最小二乘） ----
IK_MAX_ITERS = 60
IK_DAMPING  = 2e-3
IK_STEP     = 0.7

EPS_POS = 3e-3   # 位置阈值（3mm）
# 注意：这里不再用“完整姿态误差阈值”，而用 AXIS_ANGLE_TOL 控制方向

# 旋转误差权重（影响“先对齐方向还是先到位位置”）
# 一般方向约束对任务很重要时可取 1~3；如果更想要位置成功率，可取 0.5~1
W_ROT = 1.5

# 每个目标点最多尝试的初值数量（multi-start）
MULTI_START = 8

# ---- 扫描分辨率：粗到细 ----
VOXEL_COARSE = 0.10   # 10cm 粗扫
VOXEL_FINE   = 0.05   # 5cm 细扫
REFINE_NEIGHBOR_PAD = 0  # 细扫时是否同时细分邻近粗体素（0=只细分可达粗体素；1=带一圈邻居）

# ---- 扫描范围 bbox：可以手动给，也可以自动估计 ----
# 建议：真实项目最好手动给你关心的任务空间范围（桌面上方/胸前等），会快很多
LEFT_BBOX_MANUAL  = None
RIGHT_BBOX_MANUAL = None
# 例子：
# LEFT_BBOX_MANUAL  = dict(x=(-0.6, 0.6), y=(-0.8, 0.2), z=(0.2, 1.2))
# RIGHT_BBOX_MANUAL = dict(x=(-0.6, 0.6), y=(-0.2, 0.8), z=(0.2, 1.2))

# 如果不手动给，就用 FK 随机粗估 bbox
BBOX_EST_SAMPLES = 2500
BBOX_MARGIN = 0.05

# ---- FK warm-start 初值库 ----
# 用随机 FK 生成“工具轴方向接近目标”的初值 q，并按粗体素哈希存起来
WARMSTART_FK_SAMPLES = 6000
WARMSTART_AXIS_TOL_FOR_BANK = np.deg2rad(20.0)  # 初值库筛选阈值（比最终成功阈值放宽一些）
BANK_PER_VOXEL_MAX = 12                          # 每个体素最多存多少个 q 初值（防止内存爆）

# ---- 输出 ----
OUT_DIR = "./ik_reachability_out_axis_only"


# ===========================================================
# 2) 基础工具函数（尽量写得“可读 + 可改”）
# ===========================================================

def guess_package_dirs(urdf_path: str):
    """
    用于解析 URDF 中 package://... mesh 路径。
    Pinocchio/meshcat 不知道 ROS package 的搜索根，需要你提供 package_dirs。
    这里做一个“自动猜测”：把 URDF 所在目录及若干上级目录加入候选。
    如果你仍发现模型显示不全，需要手动把真实 package 根目录加入。
    """
    dirs = []
    if not urdf_path:
        return dirs
    urdf_dir = os.path.dirname(urdf_path)
    dirs.append(urdf_dir)

    cur = urdf_dir
    for _ in range(8):
        cur = os.path.dirname(cur)
        if cur and cur not in dirs:
            dirs.append(cur)

    # 你的环境里常见根
    dirs.append("/anyverse")

    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    return dirs


def find_frame_by_keywords(model: pin.Model, keywords):
    """按关键词（不区分大小写）从 model.frames 里找第一个匹配的 frame name。"""
    kw = [k.lower() for k in keywords]
    for f in model.frames:
        n = f.name.lower()
        if any(k in n for k in kw):
            return f.name
    return None


def list_frame_candidates(model: pin.Model, contains_any):
    """列出包含任意关键词的 frame 名，辅助你手动定位末端 frame。"""
    keys = [k.lower() for k in contains_any]
    out = []
    for f in model.frames:
        n = f.name.lower()
        if any(k in n for k in keys):
            out.append(f.name)
    return out


def chain_joint_ids_from_frame(model: pin.Model, frame_name: str):
    """
    从末端 frame 向上追溯 parent joint，直到 universe(0)，得到该 frame 的关节链条 joint ids。
    """
    if not model.existFrame(frame_name):
        raise ValueError(f"Frame '{frame_name}' not found.")
    fid = model.getFrameId(frame_name)
    jid = model.frames[fid].parent
    chain = []
    while jid > 0:
        chain.append(jid)
        jid = model.parents[jid]
    chain.reverse()
    return chain


def joint_ids_to_q_indices(model: pin.Model, joint_ids):
    """
    把 joint ids 转换为 q 向量中的维度下标 active_q_idx。
    Pinocchio: joint.idx_q 是该关节在 q 中的起始下标，joint.nq 是占用长度。
    """
    idx = []
    for jid in joint_ids:
        j = model.joints[jid]
        idx.extend(range(j.idx_q, j.idx_q + j.nq))
    return sorted(set(idx))


def clamp_to_limits(model: pin.Model, q: np.ndarray):
    """对有上下限的关节维度 clamp；无穷限的不处理。"""
    q2 = q.copy()
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    finite = np.isfinite(lo) & np.isfinite(hi)
    q2[finite] = np.minimum(np.maximum(q2[finite], lo[finite]), hi[finite])
    return q2


def rand_in_limits(model: pin.Model, q_ref: np.ndarray, active_q_idx):
    """
    随机生成一个初值：
    - active_q_idx：随机采样
    - 非 active：固定为 q_ref
    """
    q = q_ref.copy()
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    for k in active_q_idx:
        if np.isfinite(lo[k]) and np.isfinite(hi[k]) and hi[k] > lo[k]:
            q[k] = np.random.uniform(lo[k], hi[k])
    return q


def compute_frame_pose_world(model, data, q, frame_id):
    """给定 q，返回末端 frame 在世界系的位姿 oMf（SE3）。"""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def frame_jacobian_LWA(model, data, q, frame_id):
    """
    返回 frame 的 6xN 雅可比，采用 LOCAL_WORLD_ALIGNED（LWA）坐标：
    - 线速度部分：世界系
    - 角速度部分：世界系
    这样我们后面在世界系构造 position error、axis error 更一致。
    """
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    return pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)


def tool_axis_world(oMf: pin.SE3, axis_local: np.ndarray):
    """
    给定末端位姿 oMf（世界系），以及工具轴在末端坐标系的表示 axis_local，
    返回该工具轴在世界系的方向向量（单位向量）。
    """
    a = oMf.rotation @ axis_local
    n = np.linalg.norm(a) + 1e-12
    return a / n


def axis_angle(a: np.ndarray, b: np.ndarray):
    """
    返回两个单位向量 a、b 的夹角（0~pi）。
    """
    aa = a / (np.linalg.norm(a) + 1e-12)
    bb = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return math.acos(c)


def estimate_bbox_via_random_fk(model, data, frame_id, q_ref, active_q_idx,
                                n_samples=2000, margin=0.05):
    """
    用随机 FK 粗估末端位置范围 bbox（只用于确定扫描范围）。
    """
    pts = []
    for _ in range(n_samples):
        q = rand_in_limits(model, q_ref, active_q_idx)
        oMf = compute_frame_pose_world(model, data, q, frame_id)
        pts.append(oMf.translation.copy())
    pts = np.array(pts)
    mn = pts.min(axis=0) - margin
    mx = pts.max(axis=0) + margin
    return {"x": (mn[0], mx[0]), "y": (mn[1], mx[1]), "z": (mn[2], mx[2])}


def build_voxel_grid_ordered(bbox, voxel):
    """
    生成体素中心点，并按“蛇形遍历”排序：
    - 按 z 层遍历
    - 每一层按 y 行遍历
    - 每一行 x 方向交替（正向/反向）
    目的：让连续点尽量空间邻近，continuation (q_prev) 才好用。
    """
    xs = np.arange(bbox["x"][0], bbox["x"][1] + 1e-9, voxel)
    ys = np.arange(bbox["y"][0], bbox["y"][1] + 1e-9, voxel)
    zs = np.arange(bbox["z"][0], bbox["z"][1] + 1e-9, voxel)

    pts = []
    flip_y = False
    for z in zs:
        # 每层遍历 y 行
        y_iter = ys[::-1] if flip_y else ys
        flip_x = False
        for y in y_iter:
            x_iter = xs[::-1] if flip_x else xs
            for x in x_iter:
                pts.append([x, y, z])
            flip_x = not flip_x
        flip_y = not flip_y

    return np.array(pts, dtype=float)


def voxel_key(p: np.ndarray, origin: np.ndarray, voxel: float):
    """
    把连续空间点 p 映射为体素索引（整数三元组），用于哈希。
    origin 通常取 bbox_min，保证索引非负。
    """
    idx = np.floor((p - origin) / voxel + 1e-9).astype(int)
    return (int(idx[0]), int(idx[1]), int(idx[2]))


def save_points(points: np.ndarray, path_prefix: str):
    """同时保存 npy 与 csv。"""
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(path_prefix + ".npy", points)
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(path_prefix + ".csv", index=False)


# ===========================================================
# 3) IK：只约束“位置 + 工具轴方向”（roll 自由）
# ===========================================================

def ik_solve_position_and_axis(
    model, data,
    frame_id: int,
    q_init: np.ndarray,
    p_des: np.ndarray,             # 目标位置（世界系）
    axis_des_world: np.ndarray,    # 目标工具轴方向（世界系，单位向量）
    axis_local: np.ndarray,        # 工具轴在末端坐标系表示（如 [0,0,1]）
    active_q_idx: list,
    q_ref: np.ndarray,
    max_iters=IK_MAX_ITERS,
    damping=IK_DAMPING,
    step=IK_STEP,
    eps_pos=EPS_POS,
    axis_angle_tol=AXIS_ANGLE_TOL,
    w_rot=W_ROT
):
    """
    数值IK（阻尼最小二乘）：
    - 误差向量 err6 = [p_err(3), w_err(3)]
    - p_err = p_des - p_cur （世界系）
    - w_err = (a_cur x a_des)  （世界系）
        这个误差有一个非常重要的性质：
        - 当 a_cur 与 a_des 对齐时，叉积为 0
        - 当绕 a_des 旋转（roll），a_cur 不变 -> 误差仍为 0
        => 只约束“方向轴”，不约束绕轴旋转（roll 自由）

    雅可比：
    - 使用 LOCAL_WORLD_ALIGNED Jacobian（线速度/角速度都在世界系）
    - 取 active_q_idx 的列，解 dq_active
    - 非 active 维度固定为 q_ref，实现“全身给定姿态下只动该臂”

    返回：
    - success, q_sol, (pos_err_norm, axis_angle)
    """
    q = clamp_to_limits(model, q_init)
    axis_des = axis_des_world / (np.linalg.norm(axis_des_world) + 1e-12)

    I6 = np.eye(6)

    for _ in range(max_iters):
        # 当前末端位姿
        oMf = compute_frame_pose_world(model, data, q, frame_id)
        p_cur = oMf.translation
        a_cur = tool_axis_world(oMf, axis_local)

        # 位置误差
        p_err = (p_des - p_cur)

        # 方向误差：用叉积近似小角度旋转误差（世界系）
        # a_cur x a_des 表示把 a_cur 旋到 a_des 所需的角速度方向（小角度下）
        w_err = np.cross(a_cur, axis_des) * w_rot

        pos_norm = float(np.linalg.norm(p_err))
        ang = axis_angle(a_cur, axis_des)

        # 收敛判据：位置误差 + 工具轴夹角误差
        if pos_norm < eps_pos and ang < axis_angle_tol:
            return True, q, (pos_norm, ang)

        # 雅可比（LWA，世界系）
        J6 = frame_jacobian_LWA(model, data, q, frame_id)  # 6 x nq
        J = J6[:, active_q_idx]  # 6 x k

        # 拼误差向量
        err6 = np.hstack([p_err, w_err])  # 6,

        # DLS/LM：dq = J^T (J J^T + λ^2 I)^-1 err
        A = J @ J.T + (damping ** 2) * I6
        dq_active = J.T @ np.linalg.solve(A, err6)

        # 更新 active 维度
        q_new = q.copy()
        q_new[active_q_idx] = q[active_q_idx] + step * dq_active

        # 固定非 active 维度为 q_ref
        non_active = np.ones(model.nq, dtype=bool)
        non_active[active_q_idx] = False
        q_new[non_active] = q_ref[non_active]

        q = clamp_to_limits(model, q_new)

    # 失败：返回最终误差
    oMf = compute_frame_pose_world(model, data, q, frame_id)
    p_cur = oMf.translation
    a_cur = tool_axis_world(oMf, axis_local)
    pos_norm = float(np.linalg.norm(p_des - p_cur))
    ang = axis_angle(a_cur, axis_des)
    return False, q, (pos_norm, ang)


# ===========================================================
# 4) FK warm-start 初值库：按粗体素存 q 初值
# ===========================================================

def build_warmstart_bank(
    model, data,
    frame_id: int,
    q_ref: np.ndarray,
    active_q_idx: list,
    bbox: dict,
    voxel_coarse: float,
    axis_des_world: np.ndarray,
    axis_local: np.ndarray,
    fk_samples: int,
    axis_tol_for_bank: float,
    per_voxel_max: int
):
    """
    生成一个 dict：bank[(ix,iy,iz)] = [q1, q2, ...]
    q 来自随机采样，只保留工具轴方向与目标方向夹角 < axis_tol_for_bank 的样本。
    同时按位置落在哪个粗体素存起来。

    为什么有效：
    - 你扫某个 p 时，需要一个“离解近”的 q0 才容易收敛。
    - FK 采样能提供很多“方向差不多”的 q0，尤其在固定方向任务里非常关键。
    """
    axis_des = axis_des_world / (np.linalg.norm(axis_des_world) + 1e-12)

    origin = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]], dtype=float)
    bank = {}

    for _ in range(fk_samples):
        q = rand_in_limits(model, q_ref, active_q_idx)
        oMf = compute_frame_pose_world(model, data, q, frame_id)
        p = oMf.translation
        a = tool_axis_world(oMf, axis_local)

        ang = axis_angle(a, axis_des)
        if ang > axis_tol_for_bank:
            continue

        k = voxel_key(p, origin, voxel_coarse)
        if k not in bank:
            bank[k] = []
        if len(bank[k]) < per_voxel_max:
            bank[k].append(q.copy())

    return bank


def gather_bank_inits(bank: dict, key0, pad=0):
    """
    从 bank 里取某个体素及其邻域(pad圈)的初值列表。
    pad=0: 只取本体素；pad=1: 取 3x3x3 体素邻域
    """
    inits = []
    x0, y0, z0 = key0
    for dx in range(-pad, pad + 1):
        for dy in range(-pad, pad + 1):
            for dz in range(-pad, pad + 1):
                k = (x0 + dx, y0 + dy, z0 + dz)
                if k in bank:
                    inits.extend(bank[k])
    return inits


# ===========================================================
# 5) 粗到细扫描逻辑
# ===========================================================

def refine_points_from_reachable_voxels(reachable_coarse: np.ndarray, voxel_coarse: float, voxel_fine: float,
                                       pad_voxels=0):
    """
    根据粗扫可达点（它们在粗网格上），生成细扫点集合：
    - 对每个“可达粗体素中心”，生成该粗体素内部的细网格点（2x/4x...）
    - pad_voxels>0 时，连同邻居粗体素也细分（让边界更完整）
    """
    if len(reachable_coarse) == 0:
        return np.zeros((0, 3), dtype=float)

    # 因为 reachable_coarse 就是粗体素中心点，我们可以把它当作“粗体素中心”
    # 粗体素中心 c，对应的粗体素边界范围 [c-voxel/2, c+voxel/2]
    half = voxel_coarse / 2.0

    # 细体素在粗体素内的分割数
    # voxel_coarse 必须是 voxel_fine 的整数倍（否则会有小偏差）
    ratio = int(round(voxel_coarse / voxel_fine))
    if ratio < 1:
        ratio = 1

    # 细分偏移：例如 ratio=2，则偏移为 [-0.25, +0.25]*voxel_coarse
    # 让细点均匀分布在粗体素内部
    offsets_1d = (np.arange(ratio) + 0.5) / ratio - 0.5  # in [-0.5, 0.5)
    offsets_1d = offsets_1d * voxel_coarse

    # 生成细点
    fine_pts = []
    for c in reachable_coarse:
        # 允许对邻居粗体素也细分
        for dx in range(-pad_voxels, pad_voxels + 1):
            for dy in range(-pad_voxels, pad_voxels + 1):
                for dz in range(-pad_voxels, pad_voxels + 1):
                    center = c + np.array([dx, dy, dz], dtype=float) * voxel_coarse
                    for ox in offsets_1d:
                        for oy in offsets_1d:
                            for oz in offsets_1d:
                                fine_pts.append(center + np.array([ox, oy, oz], dtype=float))

    # 去重（哈希到 fine voxel）
    fine_pts = np.array(fine_pts, dtype=float)
    key = np.round(fine_pts / voxel_fine).astype(int)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    return fine_pts[uniq_idx]


def ik_scan_points(
    model, data,
    frame_id: int,
    q_ref: np.ndarray,
    active_q_idx: list,
    points: np.ndarray,                 # N x 3 扫描点（已按空间连续排序更好）
    bbox_origin_for_key: np.ndarray,     # 用于 voxel_key
    bank: dict,                          # warm-start 初值库
    voxel_for_bank_key: float,
    axis_des_world: np.ndarray,
    axis_local: np.ndarray,
    multi_start: int,
    bank_neighbor_pad: int = 0,
    verbose_every: int = 2000
):
    """
    对给定 points 逐点做 IK 判定，返回可达点集。
    初值选择策略（按优先级）：
      1) continuation：上一成功解 q_prev
      2) q_ref
      3) warm-start bank：对应体素及邻居体素的 q
      4) 随机初值（补足到 multi_start）

    为什么这样成功率高：
    - continuation 在空间连续遍历时非常强
    - bank 初值来自 FK 过滤，通常离可行解更近
    """
    reachable = []
    q_prev = q_ref.copy()

    t0 = time.time()
    for i, p_des in enumerate(points):
        # 体素 key，用于取 bank 初值
        k0 = voxel_key(p_des, bbox_origin_for_key, voxel_for_bank_key)
        bank_inits = gather_bank_inits(bank, k0, pad=bank_neighbor_pad)

        # 构建初值列表（去重：用 id 或近似比较都可以，这里简单截断）
        inits = [q_prev, q_ref]
        # bank 初值可能很多，取前若干个（也可以随机挑）
        if len(bank_inits) > 0:
            inits.extend(bank_inits[:max(0, multi_start - len(inits))])

        # 若还不足，补随机初值
        while len(inits) < multi_start:
            inits.append(rand_in_limits(model, q_ref, active_q_idx))

        success = False
        q_best = None

        for q0 in inits[:multi_start]:
            ok, q_sol, (pos_err, ang_err) = ik_solve_position_and_axis(
                model, data,
                frame_id=frame_id,
                q_init=q0,
                p_des=p_des,
                axis_des_world=axis_des_world,
                axis_local=axis_local,
                active_q_idx=active_q_idx,
                q_ref=q_ref
            )
            if ok:
                success = True
                q_best = q_sol
                break

        if success:
            reachable.append(p_des)
            q_prev = q_best  # continuation 更新

        if verbose_every and (i + 1) % verbose_every == 0:
            dt = time.time() - t0
            print(f"  scanned {i+1}/{len(points)} | reachable={len(reachable)} | {dt:.1f}s")

    return np.array(reachable, dtype=float)


# ===========================================================
# 6) 主程序
# ===========================================================

def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    package_dirs = guess_package_dirs(URDF_PATH)
    print("[INFO] URDF:", URDF_PATH)
    print("[INFO] package_dirs:")
    for d in package_dirs:
        print("  -", d)

    # 构建模型（含 visual/collision）
    if ROOT_JOINT is None:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs)
    else:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs, ROOT_JOINT)

    data = model.createData()

    # =======================================================
    # 默认“全身参考姿态 q_ref”：Pinocchio neutral（通常是URDF零位）
    # mentor 如果给了“某种全身姿态”，你应把 q_ref 换成那组关节角
    # =======================================================
    q_ref = pin.neutral(model)

    # =======================================================
    # 寻找左右末端 frame
    # =======================================================
    if LEFT_EE_NAME is None:
        LEFT_EE_NAME = find_frame_by_keywords(model, LEFT_EE_KEYWORDS)
    if RIGHT_EE_NAME is None:
        RIGHT_EE_NAME = find_frame_by_keywords(model, RIGHT_EE_KEYWORDS)

    if LEFT_EE_NAME is None or RIGHT_EE_NAME is None:
        print("[WARN] Cannot auto-find both EE frames.")
        print("  Left candidates:", list_frame_candidates(model, ["left", "l_", "hand", "gripper", "wrist", "tip"])[:80])
        print("  Right candidates:", list_frame_candidates(model, ["right", "r_", "hand", "gripper", "wrist", "tip"])[:80])
        raise RuntimeError("Please set LEFT_EE_NAME / RIGHT_EE_NAME or adjust keyword lists.")

    left_fid = model.getFrameId(LEFT_EE_NAME)
    right_fid = model.getFrameId(RIGHT_EE_NAME)
    print("[INFO] Left EE frame:", LEFT_EE_NAME, "id=", left_fid)
    print("[INFO] Right EE frame:", RIGHT_EE_NAME, "id=", right_fid)

    # =======================================================
    # 获取左右臂链条（只允许该臂关节动，其它固定）
    # 注意：如果你的 URDF 是“从基座一路到手”同一棵树，
    # 这里链条可能包含躯干/肩等关节 —— 这其实是好事（能扩大可达空间）。
    # 如果你希望“只动手臂不动躯干”，你需要自己裁剪 active_q_idx。
    # =======================================================
    left_chain = chain_joint_ids_from_frame(model, LEFT_EE_NAME)
    right_chain = chain_joint_ids_from_frame(model, RIGHT_EE_NAME)

    left_active_q = joint_ids_to_q_indices(model, left_chain)
    right_active_q = joint_ids_to_q_indices(model, right_chain)

    print("[INFO] Left active q dims:", len(left_active_q))
    print("[INFO] Right active q dims:", len(right_active_q))

    # =======================================================
    # 任务空间扫描范围 bbox
    # =======================================================
    if LEFT_BBOX_MANUAL is not None:
        left_bbox = LEFT_BBOX_MANUAL
    else:
        print("[INFO] Estimating LEFT bbox via random FK ...")
        left_bbox = estimate_bbox_via_random_fk(
            model, data, left_fid, q_ref, left_active_q,
            n_samples=BBOX_EST_SAMPLES, margin=BBOX_MARGIN
        )

    if RIGHT_BBOX_MANUAL is not None:
        right_bbox = RIGHT_BBOX_MANUAL
    else:
        print("[INFO] Estimating RIGHT bbox via random FK ...")
        right_bbox = estimate_bbox_via_random_fk(
            model, data, right_fid, q_ref, right_active_q,
            n_samples=BBOX_EST_SAMPLES, margin=BBOX_MARGIN
        )

    print("[INFO] LEFT bbox:", left_bbox)
    print("[INFO] RIGHT bbox:", right_bbox)

    left_origin = np.array([left_bbox["x"][0], left_bbox["y"][0], left_bbox["z"][0]], dtype=float)
    right_origin = np.array([right_bbox["x"][0], right_bbox["y"][0], right_bbox["z"][0]], dtype=float)

    # =======================================================
    # 可视化：显示机器人
    # =======================================================
    viz = None
    if USE_MESHCAT:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(q_ref)

    # =======================================================
    # 先为左右臂建立 warm-start bank
    # =======================================================
    print("[INFO] Building warm-start bank (LEFT) ...")
    left_bank = build_warmstart_bank(
        model, data,
        frame_id=left_fid,
        q_ref=q_ref,
        active_q_idx=left_active_q,
        bbox=left_bbox,
        voxel_coarse=VOXEL_COARSE,
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        fk_samples=WARMSTART_FK_SAMPLES,
        axis_tol_for_bank=WARMSTART_AXIS_TOL_FOR_BANK,
        per_voxel_max=BANK_PER_VOXEL_MAX
    )
    print(f"[INFO] LEFT bank voxels: {len(left_bank)}")

    print("[INFO] Building warm-start bank (RIGHT) ...")
    right_bank = build_warmstart_bank(
        model, data,
        frame_id=right_fid,
        q_ref=q_ref,
        active_q_idx=right_active_q,
        bbox=right_bbox,
        voxel_coarse=VOXEL_COARSE,
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        fk_samples=WARMSTART_FK_SAMPLES,
        axis_tol_for_bank=WARMSTART_AXIS_TOL_FOR_BANK,
        per_voxel_max=BANK_PER_VOXEL_MAX
    )
    print(f"[INFO] RIGHT bank voxels: {len(right_bank)}")

    # =======================================================
    # (阶段1) 粗扫
    # =======================================================
    print("[RUN] Stage-1 COARSE scan LEFT ...")
    left_points_coarse = build_voxel_grid_ordered(left_bbox, VOXEL_COARSE)
    left_reach_coarse = ik_scan_points(
        model, data,
        frame_id=left_fid,
        q_ref=q_ref,
        active_q_idx=left_active_q,
        points=left_points_coarse,
        bbox_origin_for_key=left_origin,
        bank=left_bank,
        voxel_for_bank_key=VOXEL_COARSE,
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        multi_start=MULTI_START,
        bank_neighbor_pad=1,          # 取一圈邻居的bank初值，成功率更高
        verbose_every=2000
    )
    print("[DONE] LEFT coarse reachable:", len(left_reach_coarse))
    save_points(left_reach_coarse, os.path.join(OUT_DIR, "left_reachable_coarse"))

    print("[RUN] Stage-1 COARSE scan RIGHT ...")
    right_points_coarse = build_voxel_grid_ordered(right_bbox, VOXEL_COARSE)
    right_reach_coarse = ik_scan_points(
        model, data,
        frame_id=right_fid,
        q_ref=q_ref,
        active_q_idx=right_active_q,
        points=right_points_coarse,
        bbox_origin_for_key=right_origin,
        bank=right_bank,
        voxel_for_bank_key=VOXEL_COARSE,
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        multi_start=MULTI_START,
        bank_neighbor_pad=1,
        verbose_every=2000
    )
    print("[DONE] RIGHT coarse reachable:", len(right_reach_coarse))
    save_points(right_reach_coarse, os.path.join(OUT_DIR, "right_reachable_coarse"))

    # =======================================================
    # (阶段2) 细扫：只对“粗扫可达体素”细分
    # =======================================================
    print("[RUN] Stage-2 FINE refine LEFT ...")
    left_points_fine = refine_points_from_reachable_voxels(
        left_reach_coarse,
        voxel_coarse=VOXEL_COARSE,
        voxel_fine=VOXEL_FINE,
        pad_voxels=REFINE_NEIGHBOR_PAD
    )
    # 细扫点也按蛇形排序（对 continuation 更友好）
    # 这里为了简单：把细点按 (z,y,x) 排序 + 行内蛇形不严格，但仍然比随机好
    if len(left_points_fine) > 0:
        left_points_fine = left_points_fine[np.lexsort((left_points_fine[:, 0], left_points_fine[:, 1], left_points_fine[:, 2]))]

    left_reach_fine = ik_scan_points(
        model, data,
        frame_id=left_fid,
        q_ref=q_ref,
        active_q_idx=left_active_q,
        points=left_points_fine,
        bbox_origin_for_key=left_origin,
        bank=left_bank,
        voxel_for_bank_key=VOXEL_COARSE,   # bank 仍用粗体素key索引即可
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        multi_start=MULTI_START,
        bank_neighbor_pad=1,
        verbose_every=3000
    )
    print("[DONE] LEFT fine reachable:", len(left_reach_fine))
    save_points(left_reach_fine, os.path.join(OUT_DIR, "left_reachable_fine"))

    print("[RUN] Stage-2 FINE refine RIGHT ...")
    right_points_fine = refine_points_from_reachable_voxels(
        right_reach_coarse,
        voxel_coarse=VOXEL_COARSE,
        voxel_fine=VOXEL_FINE,
        pad_voxels=REFINE_NEIGHBOR_PAD
    )
    if len(right_points_fine) > 0:
        right_points_fine = right_points_fine[np.lexsort((right_points_fine[:, 0], right_points_fine[:, 1], right_points_fine[:, 2]))]

    right_reach_fine = ik_scan_points(
        model, data,
        frame_id=right_fid,
        q_ref=q_ref,
        active_q_idx=right_active_q,
        points=right_points_fine,
        bbox_origin_for_key=right_origin,
        bank=right_bank,
        voxel_for_bank_key=VOXEL_COARSE,
        axis_des_world=TARGET_AXIS_WORLD,
        axis_local=TOOL_AXIS_LOCAL,
        multi_start=MULTI_START,
        bank_neighbor_pad=1,
        verbose_every=3000
    )
    print("[DONE] RIGHT fine reachable:", len(right_reach_fine))
    save_points(right_reach_fine, os.path.join(OUT_DIR, "right_reachable_fine"))

    print("[OUT] saved to:", OUT_DIR)

    # =======================================================
    # Meshcat 点云显示（左红右蓝）
    # =======================================================
    if USE_MESHCAT and viz is not None:
        viewer = viz.viewer

        # 可视化建议：优先显示 fine（更密），也可改为 coarse
        if len(left_reach_fine) > 0:
            pts = left_reach_fine.T
            colors = np.zeros((3, pts.shape[1]))
            colors[0, :] = 1.0
            viewer["reach_left"].set_object(g.PointCloud(position=pts, color=colors, size=0.01))

        if len(right_reach_fine) > 0:
            pts = right_reach_fine.T
            colors = np.zeros((3, pts.shape[1]))
            colors[2, :] = 1.0
            viewer["reach_right"].set_object(g.PointCloud(position=pts, color=colors, size=0.01))

        viz.display(q_ref)
        print("[VIS] Meshcat updated (fine clouds).")


if __name__ == "__main__":
    main()
