#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ik-workspace-collision-surface-fast.py

在你上一版“碰撞检测 + 表面提取”的基础上，针对“运行几十分钟太慢”做了针对性提速：

核心提速手段（默认开启）：
1) 多进程并行扫描体素点（每个进程独立加载模型与数据，避免线程安全/共享数据问题）
2) multi-start 改为“分层尝试”：
   - 先尝试 continuation(q_prev)
   - 再尝试 q_ref
   - 只有都失败，才进入随机初值（避免每点都生成随机）
3) 每个 chunk 内保持“局部连续性”：点按 (z,y,x) 排序，增强 q_prev 的复用

输出与可视化：
- left/right reachable + left/right surface（npy/csv）
- Meshcat 默认只画 surface（干净且更快）

注意：
- 并行会显著提升速度，但也会增加内存占用（每个 worker 都会加载一份模型/mesh）。
- 如果你机器内存紧张，把 N_WORKERS 调小。
"""

import os
import time
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


# ==========================================================
# 1) 用户可配置区
# ==========================================================

URDF_PATH = "/anyverse/sub_modules/isaacros/src/wheeled_humanoid/robot_model/urdf/wheel_robot.urdf"
ROOT_JOINT = None

LEFT_EE_KEYWORDS  = ["left_gripper", "left_hand", "l_gripper", "l_hand", "left_tip", "left_wrist"]
RIGHT_EE_KEYWORDS = ["right_gripper", "right_hand", "r_gripper", "r_hand", "right_tip", "right_wrist"]

IK_MAX_ITERS = 80
IK_DAMPING  = 1e-3
IK_STEP     = 0.6

EPS_POS = 2e-3
EPS_ROT = 3e-2

MULTI_START = 6

VOXEL = 0.05

BBOX_EST_SAMPLES = 2500
BBOX_MARGIN = 0.05

LEFT_BBOX_MANUAL  = None
RIGHT_BBOX_MANUAL = None

OUT_DIR = "./ik_reachability_out"

# ------------------ 提速相关开关 ------------------
USE_PARALLEL = True
# 默认用 (CPU核数-1)；内存吃紧就调小，比如 4/6
N_WORKERS = max(1, (os.cpu_count() or 8) - 1)
# 任务切片数量：越大负载越均匀，但进程间传输开销也会略增。
CHUNK_MULTIPLIER = 4


# ==========================================================
# 2) 小工具函数（大部分与上一版一致）
# ==========================================================

def guess_package_dirs(urdf_path: str):
    dirs = []
    if not urdf_path:
        return dirs
    urdf_dir = os.path.dirname(urdf_path)
    dirs.append(urdf_dir)
    cur = urdf_dir
    for _ in range(6):
        cur = os.path.dirname(cur)
        if cur and cur not in dirs:
            dirs.append(cur)
    dirs.append("/anyverse")
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    return dirs


def find_frame_by_keywords(model: pin.Model, keywords):
    kw = [k.lower() for k in keywords]
    for f in model.frames:
        name = f.name.lower()
        if any(k in name for k in kw):
            return f.name
    return None


def list_frame_candidates(model: pin.Model, contains_any):
    keys = [k.lower() for k in contains_any]
    cands = []
    for f in model.frames:
        n = f.name.lower()
        if any(k in n for k in keys):
            cands.append(f.name)
    return cands


def chain_joint_ids_from_frame(model: pin.Model, frame_name: str):
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
    idx = []
    for jid in joint_ids:
        j = model.joints[jid]
        start = j.idx_q
        nq = j.nq
        idx.extend(range(start, start + nq))
    return sorted(set(idx))


def clamp_to_limits(model: pin.Model, q: np.ndarray):
    q2 = q.copy()
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    finite = np.isfinite(lo) & np.isfinite(hi)
    q2[finite] = np.minimum(np.maximum(q2[finite], lo[finite]), hi[finite])
    return q2


def rand_in_limits(model: pin.Model, q_ref: np.ndarray, active_q_idx):
    q = q_ref.copy()
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    for k in active_q_idx:
        if np.isfinite(lo[k]) and np.isfinite(hi[k]) and hi[k] > lo[k]:
            q[k] = np.random.uniform(lo[k], hi[k])
        else:
            q[k] = q_ref[k]
    return q


def se3_error_local(oMf: pin.SE3, oMd: pin.SE3):
    return pin.log6(oMf.actInv(oMd)).vector


def compute_frame_pose(model, data, q, frame_id):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def compute_frame_jacobian_local(model, data, q, frame_id):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    return pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL)


def ik_solve_pose_for_arm(
    model, data,
    frame_id,
    q_init,
    oMd: pin.SE3,
    active_q_idx,
    q_ref,
    max_iters=IK_MAX_ITERS,
    damping=IK_DAMPING,
    step=IK_STEP,
    eps_pos=EPS_POS,
    eps_rot=EPS_ROT,
):
    q = clamp_to_limits(model, q_init)
    I6 = np.eye(6)
    non_active = np.ones(model.nq, dtype=bool)
    non_active[active_q_idx] = False

    for _ in range(max_iters):
        oMf = compute_frame_pose(model, data, q, frame_id)
        err6 = se3_error_local(oMf, oMd)
        if np.linalg.norm(err6[:3]) < eps_pos and np.linalg.norm(err6[3:]) < eps_rot:
            return True, q, err6

        J6 = compute_frame_jacobian_local(model, data, q, frame_id)
        J = J6[:, active_q_idx]
        A = J @ J.T + (damping ** 2) * I6
        dq_active = J.T @ np.linalg.solve(A, err6)

        q_new = q.copy()
        q_new[active_q_idx] = q[active_q_idx] + step * dq_active
        q_new[non_active] = q_ref[non_active]
        q = clamp_to_limits(model, q_new)

    oMf = compute_frame_pose(model, data, q, frame_id)
    err6 = se3_error_local(oMf, oMd)
    return False, q, err6


def build_voxel_grid(bbox, voxel):
    xs = np.arange(bbox["x"][0], bbox["x"][1] + 1e-9, voxel)
    ys = np.arange(bbox["y"][0], bbox["y"][1] + 1e-9, voxel)
    zs = np.arange(bbox["z"][0], bbox["z"][1] + 1e-9, voxel)
    pts = np.array(np.meshgrid(xs, ys, zs, indexing="xy"), dtype=float)
    return pts.reshape(3, -1).T


def estimate_bbox_via_random_fk(model, data, frame_id, q_ref, active_q_idx,
                                n_samples=2000, margin=0.05):
    pts = []
    for _ in range(n_samples):
        q = rand_in_limits(model, q_ref, active_q_idx)
        oMf = compute_frame_pose(model, data, q, frame_id)
        pts.append(oMf.translation.copy())
    pts = np.array(pts)
    mn = pts.min(axis=0) - margin
    mx = pts.max(axis=0) + margin
    return {"x": (mn[0], mx[0]), "y": (mn[1], mx[1]), "z": (mn[2], mx[2])}


def save_points(points, csv_path, npy_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.save(npy_path, points)
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(csv_path, index=False)


def is_collision_free(model, data, collision_model, collision_data, q, stop_at_first_collision=True) -> bool:
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)
    res = pin.computeCollisions(collision_model, collision_data, stop_at_first_collision)
    if isinstance(res, (bool, np.bool_)):
        return (not bool(res))
    for cr in collision_data.collisionResults:
        if cr.isCollision():
            return False
    return True


def try_prune_adjacent_collision_pairs(model, collision_model) -> int:
    """尽量剔除“同关节/父子关节”上的碰撞对，以减少 computeCollisions 的工作量。

    为什么安全：
    - 很多 URDF/SRDF 本来就会默认忽略相邻连杆的碰撞对（它们通常会贴近甚至相交）。
    - 但你现在没有 SRDF，addAllCollisionPairs() 会把它们全加进去，碰撞对数量暴涨。

    返回：剔除掉的碰撞对数量（如果失败则返回 0）。
    """
    try:
        pairs = list(getattr(collision_model, "collisionPairs", []))
        if not pairs:
            return 0

        # GeometryObject 上通常有 parentJoint 字段
        geoms = collision_model.geometryObjects

        kept = []
        removed = 0
        parents = getattr(model, "parents", None)

        for cp in pairs:
            go1 = geoms[cp.first]
            go2 = geoms[cp.second]
            j1 = getattr(go1, "parentJoint", None)
            j2 = getattr(go2, "parentJoint", None)

            # 拿不到 parentJoint 就不动它
            if j1 is None or j2 is None:
                kept.append(cp)
                continue

            # 同一关节 / 父子关节：跳过
            is_adjacent = False
            if j1 == j2:
                is_adjacent = True
            elif parents is not None:
                if (j1 < len(parents) and parents[j1] == j2) or (j2 < len(parents) and parents[j2] == j1):
                    is_adjacent = True

            if is_adjacent:
                removed += 1
                continue
            kept.append(cp)

        # 尝试写回（不同 pinocchio 版本写法不同，做多层兜底）
        try:
            collision_model.collisionPairs = kept
        except Exception:
            try:
                collision_model.collisionPairs.clear()
                collision_model.collisionPairs.extend(kept)
            except Exception:
                return 0

        return removed
    except Exception:
        return 0


def voxel_index_from_point(p: np.ndarray, bbox: dict, voxel: float) -> tuple:
    xmin, ymin, zmin = bbox["x"][0], bbox["y"][0], bbox["z"][0]
    ix = int(np.round((p[0] - xmin) / voxel))
    iy = int(np.round((p[1] - ymin) / voxel))
    iz = int(np.round((p[2] - zmin) / voxel))
    return (ix, iy, iz)


def point_from_voxel_index(idx: tuple, bbox: dict, voxel: float) -> np.ndarray:
    xmin, ymin, zmin = bbox["x"][0], bbox["y"][0], bbox["z"][0]
    ix, iy, iz = idx
    return np.array([xmin + ix * voxel, ymin + iy * voxel, zmin + iz * voxel], dtype=float)


def extract_surface_points(reachable_points: np.ndarray, bbox: dict, voxel: float, connectivity: int = 6) -> np.ndarray:
    if reachable_points is None or len(reachable_points) == 0:
        return np.zeros((0, 3), dtype=float)
    occ = set(voxel_index_from_point(p, bbox, voxel) for p in reachable_points)
    if connectivity == 26:
        neighbor_offsets = [(dx, dy, dz)
                            for dx in (-1, 0, 1)
                            for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)
                            if not (dx == 0 and dy == 0 and dz == 0)]
    else:
        neighbor_offsets = [(1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]
    surface = []
    for idx in occ:
        ix, iy, iz = idx
        for dx, dy, dz in neighbor_offsets:
            if (ix + dx, iy + dy, iz + dz) not in occ:
                surface.append(point_from_voxel_index(idx, bbox, voxel))
                break
    return np.array(surface, dtype=float)


def _sort_points_for_locality(points: np.ndarray) -> np.ndarray:
    """为了增强 q_prev 的复用，按 (z,y,x) 排序（相邻点更可能相邻解）。"""
    if len(points) == 0:
        return points
    order = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
    return points[order]


# ==========================================================
# 3) 并行 worker 逻辑
# ==========================================================

# 进程内全局变量（每个进程一份）
_G = {}


def _worker_init(cfg: dict):
    """每个进程启动时初始化 Pinocchio 模型与常用变量。"""
    global _G
    np.random.seed(cfg.get("seed", 0) + os.getpid() % 100000)

    urdf_path = cfg["urdf_path"]
    package_dirs = cfg["package_dirs"]
    root_joint = cfg.get("root_joint", None)

    if root_joint is None:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs)
    else:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs, root_joint)
    data = model.createData()

    collision_data = None
    try:
        if hasattr(collision_model, "addAllCollisionPairs"):
            collision_model.addAllCollisionPairs()
        # 没有 SRDF 时，先剔除“同关节/父子关节”碰撞对，通常能大幅提速
        try_prune_adjacent_collision_pairs(model, collision_model)
        collision_data = collision_model.createData()
    except Exception:
        collision_data = None

    q_ref = pin.neutral(model)

    left_frame = cfg["left_frame"]
    right_frame = cfg["right_frame"]
    left_fid = model.getFrameId(left_frame)
    right_fid = model.getFrameId(right_frame)

    left_joint_chain = chain_joint_ids_from_frame(model, left_frame)
    right_joint_chain = chain_joint_ids_from_frame(model, right_frame)
    left_active_q = joint_ids_to_q_indices(model, left_joint_chain)
    right_active_q = joint_ids_to_q_indices(model, right_joint_chain)

    left_oMf_ref = compute_frame_pose(model, data, q_ref, left_fid)
    right_oMf_ref = compute_frame_pose(model, data, q_ref, right_fid)
    LEFT_R_d = left_oMf_ref.rotation.copy()
    RIGHT_R_d = right_oMf_ref.rotation.copy()

    _G = {
        "model": model,
        "data": data,
        "collision_model": collision_model,
        "collision_data": collision_data,
        "q_ref": q_ref,
        "left_fid": left_fid,
        "right_fid": right_fid,
        "left_active_q": left_active_q,
        "right_active_q": right_active_q,
        "LEFT_R_d": LEFT_R_d,
        "RIGHT_R_d": RIGHT_R_d,
    }


def _solve_chunk(args):
    """处理一个 chunk。

    args:
      arm: "left" or "right"
      points: (M,3)
    """
    arm, points = args
    g = _G
    model = g["model"]
    data = g["data"]
    collision_model = g["collision_model"]
    collision_data = g["collision_data"]
    q_ref = g["q_ref"]

    if arm == "left":
        fid = g["left_fid"]
        active_q = g["left_active_q"]
        R_d = g["LEFT_R_d"]
    else:
        fid = g["right_fid"]
        active_q = g["right_active_q"]
        R_d = g["RIGHT_R_d"]

    pts = _sort_points_for_locality(points)
    reachable = []
    q_prev = q_ref.copy()

    for p in pts:
        oMd = pin.SE3(R_d, p)

        # 分层尝试（最省时）：先 q_prev，再 q_ref；都失败才随机。
        init_list = [q_prev, q_ref]
        success = False
        q_sol_best = None

        # 先尝试 q_prev / q_ref
        for q0 in init_list:
            ok, q_sol, _ = ik_solve_pose_for_arm(
                model, data,
                frame_id=fid,
                q_init=q0,
                oMd=oMd,
                active_q_idx=active_q,
                q_ref=q_ref,
            )
            if ok:
                if collision_data is not None:
                    if not is_collision_free(model, data, collision_model, collision_data, q_sol):
                        continue
                success = True
                q_sol_best = q_sol
                break

        # 再尝试随机初值（必要时才做）
        if (not success) and MULTI_START > len(init_list):
            for _ in range(MULTI_START - len(init_list)):
                q0 = rand_in_limits(model, q_ref, active_q)
                ok, q_sol, _ = ik_solve_pose_for_arm(
                    model, data,
                    frame_id=fid,
                    q_init=q0,
                    oMd=oMd,
                    active_q_idx=active_q,
                    q_ref=q_ref,
                )
                if ok:
                    if collision_data is not None:
                        if not is_collision_free(model, data, collision_model, collision_data, q_sol):
                            continue
                    success = True
                    q_sol_best = q_sol
                    break

        if success:
            reachable.append(p)
            q_prev = q_sol_best

    return np.array(reachable, dtype=float)


def _parallel_solve(arm: str, grid_points: np.ndarray):
    """并行求解一个 arm 的可达点。"""
    import multiprocessing as mp

    if len(grid_points) == 0:
        return np.zeros((0, 3), dtype=float)

    n_workers = max(1, int(N_WORKERS))
    if n_workers == 1 or (not USE_PARALLEL):
        return _solve_chunk((arm, grid_points))

    # 分成更多小块以均衡负载
    n_chunks = max(n_workers, n_workers * int(CHUNK_MULTIPLIER))
    chunks = np.array_split(grid_points, n_chunks)
    tasks = [(arm, c) for c in chunks if len(c) > 0]

    ctx = mp.get_context("spawn")
    t0 = time.time()
    out = []
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(_PAR_CFG,)) as pool:
        for i, arr in enumerate(pool.imap_unordered(_solve_chunk, tasks), 1):
            if arr.size:
                out.append(arr)
            if i % max(1, len(tasks)//10) == 0:
                dt = time.time() - t0
                print(f"  [{arm}] chunks {i}/{len(tasks)} done | {dt:.1f}s")
    if not out:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(out)


# 主进程传给 worker 的配置（在 main() 里填充）
_PAR_CFG = {}


# ==========================================================
# 4) 主流程
# ==========================================================

def main():
    global _PAR_CFG
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    package_dirs = guess_package_dirs(URDF_PATH)
    print("[INFO] URDF:", URDF_PATH)
    print("[INFO] package_dirs:")
    for d in package_dirs:
        print("   -", d)

    # 主进程也加载一次（用于：找 frame、估 bbox、meshcat 展示）
    if ROOT_JOINT is None:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs)
    else:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs, ROOT_JOINT)
    data = model.createData()

    q_ref = pin.neutral(model)

    left_frame = find_frame_by_keywords(model, LEFT_EE_KEYWORDS)
    right_frame = find_frame_by_keywords(model, RIGHT_EE_KEYWORDS)
    if left_frame is None or right_frame is None:
        print("[WARN] Cannot auto-find both EE frames.")
        print("  Left candidates:", list_frame_candidates(model, ["left", "l_", "hand", "gripper", "wrist", "tip"])[:60])
        print("  Right candidates:", list_frame_candidates(model, ["right", "r_", "hand", "gripper", "wrist", "tip"])[:60])
        raise RuntimeError("Please adjust LEFT_EE_KEYWORDS / RIGHT_EE_KEYWORDS")

    left_fid = model.getFrameId(left_frame)
    right_fid = model.getFrameId(right_frame)
    print("[INFO] Left EE frame:", left_frame, "id=", left_fid)
    print("[INFO] Right EE frame:", right_frame, "id=", right_fid)

    left_joint_chain = chain_joint_ids_from_frame(model, left_frame)
    right_joint_chain = chain_joint_ids_from_frame(model, right_frame)
    left_active_q = joint_ids_to_q_indices(model, left_joint_chain)
    right_active_q = joint_ids_to_q_indices(model, right_joint_chain)
    print("[INFO] Left chain joints:", len(left_joint_chain), "active q dims:", len(left_active_q))
    print("[INFO] Right chain joints:", len(right_joint_chain), "active q dims:", len(right_active_q))

    left_oMf_ref = compute_frame_pose(model, data, q_ref, left_fid)
    right_oMf_ref = compute_frame_pose(model, data, q_ref, right_fid)
    LEFT_R_d = left_oMf_ref.rotation.copy()
    RIGHT_R_d = right_oMf_ref.rotation.copy()

    # bbox
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

    left_grid = build_voxel_grid(left_bbox, VOXEL)
    right_grid = build_voxel_grid(right_bbox, VOXEL)
    print("[INFO] LEFT grid points:", len(left_grid))
    print("[INFO] RIGHT grid points:", len(right_grid))

    # Meshcat
    viz = None
    if USE_MESHCAT:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(q_ref)

    # 并行 worker 配置
    _PAR_CFG = {
        "seed": 0,
        "urdf_path": URDF_PATH,
        "package_dirs": package_dirs,
        "root_joint": ROOT_JOINT,
        "left_frame": left_frame,
        "right_frame": right_frame,
    }

    # ---------------- LEFT solve ----------------
    print(f"[RUN] Solving IK for LEFT arm | parallel={USE_PARALLEL} workers={N_WORKERS} ...")
    t0 = time.time()
    left_reachable = _parallel_solve("left", left_grid)
    print(f"[DONE] LEFT reachable points: {len(left_reachable)} | {time.time()-t0:.1f}s")

    # ---------------- RIGHT solve ----------------
    print(f"[RUN] Solving IK for RIGHT arm | parallel={USE_PARALLEL} workers={N_WORKERS} ...")
    t0 = time.time()
    right_reachable = _parallel_solve("right", right_grid)
    print(f"[DONE] RIGHT reachable points: {len(right_reachable)} | {time.time()-t0:.1f}s")

    # surface
    left_surface = extract_surface_points(left_reachable, left_bbox, VOXEL, connectivity=6)
    right_surface = extract_surface_points(right_reachable, right_bbox, VOXEL, connectivity=6)
    print(f"[SURFACE] LEFT surface points: {len(left_surface)}")
    print(f"[SURFACE] RIGHT surface points: {len(right_surface)}")

    # save
    save_points(left_reachable,
                csv_path=os.path.join(OUT_DIR, "left_reachable.csv"),
                npy_path=os.path.join(OUT_DIR, "left_reachable.npy"))
    save_points(right_reachable,
                csv_path=os.path.join(OUT_DIR, "right_reachable.csv"),
                npy_path=os.path.join(OUT_DIR, "right_reachable.npy"))
    save_points(left_surface,
                csv_path=os.path.join(OUT_DIR, "left_surface.csv"),
                npy_path=os.path.join(OUT_DIR, "left_surface.npy"))
    save_points(right_surface,
                csv_path=os.path.join(OUT_DIR, "right_surface.csv"),
                npy_path=os.path.join(OUT_DIR, "right_surface.npy"))
    print("[OUT] saved to:", OUT_DIR)

    # visualize surface
    if USE_MESHCAT and viz is not None:
        viewer = viz.viewer
        pt_size = max(0.004, 0.25 * VOXEL)
        if len(left_surface) > 0:
            pts = left_surface.T
            colors = np.zeros((3, pts.shape[1])); colors[0, :] = 1.0
            viewer["reachability_left_surface"].set_object(
                g.PointCloud(position=pts, color=colors, size=pt_size)
            )
        if len(right_surface) > 0:
            pts = right_surface.T
            colors = np.zeros((3, pts.shape[1])); colors[2, :] = 1.0
            viewer["reachability_right_surface"].set_object(
                g.PointCloud(position=pts, color=colors, size=pt_size)
            )
        viz.display(q_ref)
        print("[VIS] Meshcat updated.")


if __name__ == "__main__":
    main()
