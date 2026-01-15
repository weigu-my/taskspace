#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
Dual-arm pose-conditioned reachability via IK (Pinocchio)
===========================================================

目标：
- 在“某一个给定末端姿态”（orientation固定）条件下，计算左右手各自的可达空间。
- 方法：任务空间体素采样位置 p，对每个 (p, R_d) 做逆运动学 IK。
- IK 成功 => 该 p 可达；IK 失败 => 不可达（或该点对当前多初值策略未收敛）。

你将得到：
- left_reachable.csv / left_reachable.npy  : 左手可达点集
- right_reachable.csv / right_reachable.npy: 右手可达点集
- 可选：Meshcat 显示机器人 + 左/右可达点云

为什么这更符合 mentor 的“给定姿态下可达空间”：
- 你原来的 FK+蒙特卡洛，是对所有姿态混合后的 position-workspace。
- 这里是固定 R_d（或部分姿态约束），得到“姿态切片”的 workspace。

依赖：
pip install pin meshcat numpy pandas

注意：
- URDF 中 mesh 路径常用 package://，需要 package_dirs 才能加载完整 visual。
- 如果自动找不到左右末端 frame，会打印候选列表，需改关键词或手动指定 frame 名。
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


# ===========================================================
# 1) 用户可配置区：你通常只需要改这里
# ===========================================================

# URDF 路径（你 notebook 里用的那个）
URDF_PATH = "/anyverse/sub_modules/isaacros/src/wheeled_humanoid/robot_model/urdf/wheel_robot.urdf"

# Root joint 选择：
# - 固定底座：ROOT_JOINT=None（一般是 humanoid 固定在地面）
# - 浮动底座：ROOT_JOINT=pin.JointModelFreeFlyer()
ROOT_JOINT = None

# 自动寻找末端 frame 的关键词（按你模型命名习惯改）
LEFT_EE_KEYWORDS  = ["left_gripper", "left_hand", "l_gripper", "l_hand", "left_tip", "left_wrist"]
RIGHT_EE_KEYWORDS = ["right_gripper", "right_hand", "r_gripper", "r_hand", "right_tip", "right_wrist"]

# IK 迭代参数（LM/阻尼最小二乘）
IK_MAX_ITERS = 80          # 每次 IK 最多迭代次数
IK_DAMPING  = 1e-3         # 阻尼系数 λ（避免奇异）
IK_STEP     = 0.6          # 步长 α（<1 更稳定）

# 收敛阈值：位置误差(m) + 姿态误差(rad)
EPS_POS = 2e-3             # 2mm
EPS_ROT = 3e-2             # 0.03rad ~ 1.7deg（更严格可设小，比如 0.01）

# 每个目标点最多尝试多少个初始值（multi-start）
MULTI_START = 6

# 体素分辨率：越小越精细但越慢
VOXEL = 0.05  # 5cm

# 自动估计采样包围盒：先用 FK 随机采样粗估 reachable 区域范围，便于确定扫点范围
BBOX_EST_SAMPLES = 2500
BBOX_MARGIN = 0.05

# 如果你已经知道想扫的空间范围，可手动写死（覆盖自动估计）
LEFT_BBOX_MANUAL  = None
RIGHT_BBOX_MANUAL = None
# 示例：
# LEFT_BBOX_MANUAL  = dict(x=(-0.6, 0.6), y=(-0.8, 0.2), z=(0.2, 1.2))
# RIGHT_BBOX_MANUAL = dict(x=(-0.6, 0.6), y=(-0.2, 0.8), z=(0.2, 1.2))

# 输出目录
OUT_DIR = "./ik_reachability_out"


# ===========================================================
# 2) 小工具函数
# ===========================================================

def guess_package_dirs(urdf_path: str):
    """
    用来给 Pinocchio 解析 URDF 里的 package://xxx/meshes/... 路径。
    原理：把 URDF 所在目录及其上层目录加入 package_dirs 作为搜索根。
    这不是完美方案，但通常能跑通；若仍缺 mesh，建议手工补充正确的 package 根目录。
    """
    dirs = []
    if not urdf_path:
        return dirs
    urdf_dir = os.path.dirname(urdf_path)
    dirs.append(urdf_dir)

    # 添加几层上级目录作为候选搜索根
    cur = urdf_dir
    for _ in range(6):
        cur = os.path.dirname(cur)
        if cur and cur not in dirs:
            dirs.append(cur)

    # 你的环境里常见的额外根目录
    dirs.append("/anyverse")

    # 去重 + 过滤不存在路径
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    return dirs


def find_frame_by_keywords(model: pin.Model, keywords):
    """
    在 model.frames 里按关键词匹配 frame 名称，找到就返回第一个匹配的 name。
    """
    kw = [k.lower() for k in keywords]
    for f in model.frames:
        name = f.name.lower()
        if any(k in name for k in kw):
            return f.name
    return None


def list_frame_candidates(model: pin.Model, contains_any):
    """
    打印候选 frame 列表时用：列出包含某些关键词的 frame 名，帮助你手动选末端。
    """
    keys = [k.lower() for k in contains_any]
    cands = []
    for f in model.frames:
        n = f.name.lower()
        if any(k in n for k in keys):
            cands.append(f.name)
    return cands


def chain_joint_ids_from_frame(model: pin.Model, frame_name: str):
    """
    从末端 frame 往上追溯 parent joint，直到 universe(0)，得到该 frame 对应的关节链条。
    输出：joint id 列表（不包含 universe=0）
    """
    if not model.existFrame(frame_name):
        raise ValueError(f"Frame '{frame_name}' not found.")

    fid = model.getFrameId(frame_name)
    jid = model.frames[fid].parent  # frame 所属的 parent joint
    chain = []
    while jid > 0:
        chain.append(jid)
        jid = model.parents[jid]
    chain.reverse()  # 从基座到末端
    return chain


def joint_ids_to_q_indices(model: pin.Model, joint_ids):
    """
    Pinocchio 里关节配置 q 是一个向量（长度 nq），每个 joint 在 q 里占用 [idx_q, idx_q+nq)。
    这里把 joint id 链条转换成“在 q 向量里需要更新的下标集合 active_q_idx”。
    """
    idx = []
    for jid in joint_ids:
        j = model.joints[jid]
        start = j.idx_q
        nq = j.nq
        idx.extend(range(start, start + nq))
    return sorted(set(idx))


def clamp_to_limits(model: pin.Model, q: np.ndarray):
    """
    把 q 限制在关节上下限内（如果该关节有限制）。
    对没有上下限（inf）的维度不做 clamp。
    """
    q2 = q.copy()
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    finite = np.isfinite(lo) & np.isfinite(hi)
    q2[finite] = np.minimum(np.maximum(q2[finite], lo[finite]), hi[finite])
    return q2


def rand_in_limits(model: pin.Model, q_ref: np.ndarray, active_q_idx):
    """
    生成一个随机初始姿态 q：
    - active_q_idx 对应的关节维度随机采样
    - 其他维度保持 q_ref（固定全身姿态，只让该臂动）
    """
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
    """
    SE(3) 上的“局部误差”：
      err = log( oMf^{-1} * oMd ) in se(3) -> R^6
    这里 err6 = [v(3), w(3)]，前三维是平移误差，后三维是旋转误差（轴角意义）。
    """
    return pin.log6(oMf.actInv(oMd)).vector


def compute_frame_pose(model, data, q, frame_id):
    """
    给定 q，计算 frame 在世界坐标系下的位姿 oMf。
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def compute_frame_jacobian_local(model, data, q, frame_id):
    """
    计算 frame 的 6xN 雅可比（LOCAL 坐标系下）。
    注意：我们用 LOCAL Jacobian 对应上面的局部误差 log(oMf^{-1} oMd)，搭配更一致。
    """
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J6 = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL)
    return J6


def ik_solve_pose_for_arm(
    model, data,
    frame_id,              # 要对齐的末端 frame
    q_init,                # 初始 q
    oMd: pin.SE3,          # 目标位姿（世界系）
    active_q_idx,          # 仅更新该臂对应的 q 下标
    q_ref,                 # 其余维度固定为 q_ref（对应“全身某姿态下只动该臂”）
    max_iters=IK_MAX_ITERS,
    damping=IK_DAMPING,
    step=IK_STEP,
    eps_pos=EPS_POS,
    eps_rot=EPS_ROT
):
    """
    数值 IK（阻尼最小二乘 / LM）：
      dq = J^T (J J^T + λ^2 I)^{-1} * err
    然后只更新 active_q_idx 对应的维度，其它维度严格固定为 q_ref。

    返回：
      success: 是否收敛
      q_sol:   求得的 q（全维度，含固定维度）
      err6:    最终误差（6维）
    """
    q = clamp_to_limits(model, q_init)
    I6 = np.eye(6)

    for _ in range(max_iters):
        # 当前末端位姿
        oMf = compute_frame_pose(model, data, q, frame_id)

        # 局部误差：log(oMf^{-1} oMd)
        err6 = se3_error_local(oMf, oMd)

        # 分别计算位置误差和旋转误差，用于收敛判据
        pos_err = np.linalg.norm(err6[:3])
        rot_err = np.linalg.norm(err6[3:])

        if pos_err < eps_pos and rot_err < eps_rot:
            return True, q, err6

        # 末端雅可比（LOCAL）
        J6 = compute_frame_jacobian_local(model, data, q, frame_id)

        # 只保留该臂关节对应列：6 x k
        J = J6[:, active_q_idx]

        # LM/阻尼最小二乘：解出 dq_active
        A = J @ J.T + (damping ** 2) * I6
        dq_active = J.T @ np.linalg.solve(A, err6)

        # 更新 q：只更新 active 维度
        q_new = q.copy()
        q_new[active_q_idx] = q[active_q_idx] + step * dq_active

        # 非 active 维度严格固定在 q_ref（实现“给定全身姿态，只动该臂”）
        non_active = np.ones(model.nq, dtype=bool)
        non_active[active_q_idx] = False
        q_new[non_active] = q_ref[non_active]

        q = clamp_to_limits(model, q_new)

    # 迭代结束仍未收敛：返回失败
    oMf = compute_frame_pose(model, data, q, frame_id)
    err6 = se3_error_local(oMf, oMd)
    return False, q, err6


def build_voxel_grid(bbox, voxel):
    """
    根据 bbox 生成体素网格中心点集合。
    bbox: {"x":(xmin,xmax), "y":(...), "z":(...)}
    返回：N x 3 的点集
    """
    xs = np.arange(bbox["x"][0], bbox["x"][1] + 1e-9, voxel)
    ys = np.arange(bbox["y"][0], bbox["y"][1] + 1e-9, voxel)
    zs = np.arange(bbox["z"][0], bbox["z"][1] + 1e-9, voxel)
    pts = np.array(np.meshgrid(xs, ys, zs, indexing="xy"), dtype=float)
    return pts.reshape(3, -1).T


def estimate_bbox_via_random_fk(model, data, frame_id, q_ref, active_q_idx,
                                n_samples=2000, margin=0.05):
    """
    用随机 FK 粗估该臂末端能到达的位置范围：
    - 随机采样该臂关节（其它关节固定在 q_ref）
    - 计算末端位置点云
    - 取 min/max 得到 bbox，再扩 margin

    这一步仅用于确定“扫点范围”，最终可达性判定仍由 IK 决定。
    """
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
    """
    保存结果点云到 npy 和 csv，便于后续画图/统计。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.save(npy_path, points)
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(csv_path, index=False)


def is_collision_free(model: pin.Model,
                      data: pin.Data,
                      collision_model: pin.GeometryModel,
                      collision_data: pin.GeometryData,
                      q: np.ndarray,
                      stop_at_first_collision: bool = True) -> bool:
    """检查配置 q 是否碰撞。

    说明：
    - 这里仅做“模型自碰撞/模型内部碰撞对”的检测。
    - 若你还想检测与环境(桌面/地面/物体)的碰撞，需要把环境几何体加入 collision_model。

    返回：True 表示无碰撞；False 表示发生碰撞。
    """
    # 更新几何体位姿前，必须先更新关节位姿
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)

    # Pinocchio 的 computeCollisions 返回值在不同版本可能略有差异：
    # - 有的版本返回 bool (是否发生碰撞)
    # - 有的版本返回 None，需要从 collision_data 里读结果
    res = pin.computeCollisions(collision_model, collision_data, stop_at_first_collision)

    if isinstance(res, (bool, np.bool_)):
        return (not bool(res))

    # 兜底：遍历 collisionResults
    for cr in collision_data.collisionResults:
        if cr.isCollision():
            return False
    return True


def voxel_index_from_point(p: np.ndarray, bbox: dict, voxel: float) -> tuple:
    """把一个体素中心点 p 映射到离散体素坐标 (ix,iy,iz)。

    由于 p 是由 build_voxel_grid() 生成的规则网格点，round 足够稳定。
    """
    xmin, ymin, zmin = bbox["x"][0], bbox["y"][0], bbox["z"][0]
    ix = int(np.round((p[0] - xmin) / voxel))
    iy = int(np.round((p[1] - ymin) / voxel))
    iz = int(np.round((p[2] - zmin) / voxel))
    return (ix, iy, iz)


def point_from_voxel_index(idx: tuple, bbox: dict, voxel: float) -> np.ndarray:
    """把离散体素坐标 (ix,iy,iz) 映射回体素中心点坐标。"""
    xmin, ymin, zmin = bbox["x"][0], bbox["y"][0], bbox["z"][0]
    ix, iy, iz = idx
    return np.array([xmin + ix * voxel, ymin + iy * voxel, zmin + iz * voxel], dtype=float)


def extract_surface_points(reachable_points: np.ndarray, bbox: dict, voxel: float,
                           connectivity: int = 6) -> np.ndarray:
    """从可达点云中提取“表面点”(surface voxels)。

    思路：
    - 把 reachable_points 量化到体素索引集合 occ
    - 若一个体素在 6邻域(或26邻域)上存在至少一个“不在 occ 的邻居”，则判为边界/表面体素。

    参数：
    - connectivity: 6 或 26
    """
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
        # 默认 6 邻域：更“薄”的表面、更清晰
        neighbor_offsets = [(1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]

    surface = []
    for idx in occ:
        ix, iy, iz = idx
        is_surface = False
        for dx, dy, dz in neighbor_offsets:
            nidx = (ix + dx, iy + dy, iz + dz)
            if nidx not in occ:
                is_surface = True
                break
        if is_surface:
            surface.append(point_from_voxel_index(idx, bbox, voxel))

    return np.array(surface, dtype=float)

import multiprocessing
from functools import partial

# 定义全局变量 (让子进程共享，避免 pickling 开销)
_global_model = None
_global_data = None
_global_col_model = None
_global_col_data = None
_global_fid = None
_global_active_idx = None
_global_q_ref = None
_global_R_d = None

def init_worker(urdf_path, package_dirs, root_joint, 
                frame_name, active_idx, q_ref, R_d, use_collision):
    """子进程初始化：智能剔除相邻连杆碰撞对"""
    # 声明全局变量
    global _global_model, _global_data, _global_col_model, _global_col_data
    global _global_fid, _global_active_idx, _global_q_ref, _global_R_d

    # 1. 加载运动学模型
    if root_joint is None:
        model = pin.buildModelFromUrdf(urdf_path)
    else:
        model = pin.buildModelFromUrdf(urdf_path, root_joint)
    
    # 2. 加载碰撞几何体 (修复 Warning: 使用关键字参数传 package_dirs)
    col_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=package_dirs)
    
    _global_model = model
    _global_data = model.createData()
    
    if use_collision:
        col_model.addAllCollisionPairs()
        
        # =========================================================
        # 核心修复逻辑：移除 父Link 与 子Link 之间的碰撞对
        # =========================================================
        pairs_to_remove = []
        
        for i, pair in enumerate(col_model.collisionPairs):
            # 获取两个几何体的 ID
            gid1 = pair.first
            gid2 = pair.second
            
            # 获取它们挂载的关节 ID
            jid1 = col_model.geometryObjects[gid1].parentJoint
            jid2 = col_model.geometryObjects[gid2].parentJoint
            
            # 检查亲缘关系: 同一关节、或者是父子关节，统统移除
            if (jid1 == jid2) or (model.parents[jid2] == jid1) or (model.parents[jid1] == jid2):
                pairs_to_remove.append(i)
        
        # 倒序移除（防止索引错位）
        for i in sorted(pairs_to_remove, reverse=True):
            col_model.removeCollisionPair(col_model.collisionPairs[i])
            
        # 打印一下移除了多少，让你放心 (这行只会打印在子进程日志里，可能看不见，但没关系)
        # print(f"[Worker] Removed {len(pairs_to_remove)} adjacent collision pairs.")
        
        _global_col_model = col_model
        _global_col_data = col_model.createData()
    else:
        _global_col_model = None
        _global_col_data = None

    _global_fid = model.getFrameId(frame_name)
    _global_active_idx = active_idx
    _global_q_ref = q_ref
    _global_R_d = R_d

def process_voxel(p):
    """单个体素的处理函数"""
    # 引用全局变量
    model = _global_model
    data = _global_data
    col_model = _global_col_model
    col_data = _global_col_data
    fid = _global_fid
    active_idx = _global_active_idx
    q_ref = _global_q_ref
    R_d = _global_R_d
    
    # 构造目标
    oMd = pin.SE3(R_d, p)
    
    # 策略：先尝试 q_ref (最快)，失败再尝试随机
    # 减少 Multi-start 次数以换取速度，并行化后基数大，丢失几个点无所谓
    inits = [q_ref] 
    # 再加 2 个随机初值
    for _ in range(2):
        inits.append(rand_in_limits(model, q_ref, active_idx))
        
    for q_init in inits:
        ok, q_sol, _ = ik_solve_pose_for_arm(
            model, data, fid, q_init, oMd, active_idx, q_ref, 
            max_iters=40 # 稍微降低迭代上限
        )
        
        if ok:
            # IK 成功，检查碰撞
            if col_model is not None:
                if is_collision_free(model, data, col_model, col_data, q_sol):
                    return p # 成功
                # else: 发生碰撞，继续尝试下一个初值
            else:
                return p # 无碰撞检测，直接成功
                
    return None # 不可达

# ===========================================================
# 3) 主流程
# ===========================================================

def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    # 给 pinocchio 解析 package:// mesh 用
    package_dirs = guess_package_dirs(URDF_PATH)
    print("[INFO] URDF:", URDF_PATH)
    print("[INFO] package_dirs:")
    for d in package_dirs:
        print("   -", d)

    # 构建模型（含 collision & visual）
    if ROOT_JOINT is None:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs)
    else:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, package_dirs, ROOT_JOINT)

    data = model.createData()

    # -----------------------------
    # 碰撞检测准备
    # -----------------------------
    # 关键点：很多 URDF 默认不会配置 collision pairs。
    # 如果你没有对应 SRDF 来“剔除相邻连杆等不应检测的碰撞对”，
    # 最保守的做法是 addAllCollisionPairs()，保证至少能检测到自碰撞。
    # 代价：碰撞对数量会比较多，计算更慢；若后续要提速，再引入 SRDF 过滤即可。
    collision_data = None
    try:
        if hasattr(collision_model, "addAllCollisionPairs"):
            collision_model.addAllCollisionPairs()
        collision_data = collision_model.createData()
        print(f"[INFO] Collision geom count: {collision_model.ngeoms} | collision pairs: {len(collision_model.collisionPairs)}")
    except Exception as e:
        print("[WARN] Collision model/data init failed -> collision checking disabled.")
        print("       Reason:", repr(e))
        collision_data = None

    # 参考全身姿态 q_ref：neutral 通常对应 URDF 默认零位姿（或你定义的基准姿态）
    # 这就是“某一个给定姿态”的实现载体：把 q_ref 换成 mentor 给的姿态即可。
    q_ref = pin.neutral(model)

    # 自动找左右末端 frame
    left_frame = find_frame_by_keywords(model, LEFT_EE_KEYWORDS)
    right_frame = find_frame_by_keywords(model, RIGHT_EE_KEYWORDS)

    # 找不到就输出候选，让你手动改关键词
    if left_frame is None or right_frame is None:
        print("[WARN] Cannot auto-find both EE frames.")
        print("  Left candidates:", list_frame_candidates(model, ["left", "l_", "hand", "gripper", "wrist", "tip"])[:60])
        print("  Right candidates:", list_frame_candidates(model, ["right", "r_", "hand", "gripper", "wrist", "tip"])[:60])
        raise RuntimeError("Please adjust LEFT_EE_KEYWORDS / RIGHT_EE_KEYWORDS to match your model frame names.")

    left_fid = model.getFrameId(left_frame)
    right_fid = model.getFrameId(right_frame)
    print("[INFO] Left EE frame:", left_frame, "id=", left_fid)
    print("[INFO] Right EE frame:", right_frame, "id=", right_fid)

    # 计算左右臂的关节链，并提取 q 中的 active 下标
    left_joint_chain = chain_joint_ids_from_frame(model, left_frame)
    right_joint_chain = chain_joint_ids_from_frame(model, right_frame)
    left_active_q = joint_ids_to_q_indices(model, left_joint_chain)
    right_active_q = joint_ids_to_q_indices(model, right_joint_chain)

    print("[INFO] Left chain joints:", len(left_joint_chain), "active q dims:", len(left_active_q))
    print("[INFO] Right chain joints:", len(right_joint_chain), "active q dims:", len(right_active_q))

    # ===========================================================
    # 关键：定义“给定姿态” R_d
    # ===========================================================
    # 默认取 q_ref 下末端当前旋转，等价于“在该参考姿态下固定末端方向”
    left_oMf_ref = compute_frame_pose(model, data, q_ref, left_fid)
    right_oMf_ref = compute_frame_pose(model, data, q_ref, right_fid)
    LEFT_R_d = left_oMf_ref.rotation.copy()
    RIGHT_R_d = right_oMf_ref.rotation.copy()

    # 如果你要显式指定“手掌朝下”等：
    # 例如让末端 z 轴对齐世界 -z（这只是示意，具体看你的末端坐标系定义）
    # z_d = np.array([0, 0, -1.0])
    # 你需要构造一个完整的 R_d（给一个 x 轴或用 Gram-Schmidt 构造正交基），此处略。

    # ===========================================================
    # 建立扫点范围 bbox + 体素网格
    # ===========================================================
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

    # ===========================================================
    # Meshcat 可视化：显示机器人模型
    # ===========================================================
    viz = None
    if USE_MESHCAT:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(q_ref)

    num_cores = multiprocessing.cpu_count()
    print(f"[INFO] CPU Cores: {num_cores}. Starting parallel processing...")

    # --- 左臂并行计算 ---
    print(f"[RUN] Left Arm: {len(left_grid)} voxels...")
    t0 = time.time()
    
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(URDF_PATH, package_dirs, ROOT_JOINT, 
                  left_frame, left_active_q, q_ref, LEFT_R_d, True) # True开启碰撞
    ) as pool:
        # 使用 imap_unordered 稍微快一点，chunksize 设大减少通信
        results = pool.map(process_voxel, left_grid, chunksize=100)
    
    # 过滤 None
    left_reachable = np.array([p for p in results if p is not None])
    print(f"  [DONE] Left: {len(left_reachable)} points found. Time: {time.time()-t0:.1f}s")
    
    # --- 右臂并行计算 ---
    print(f"[RUN] Right Arm: {len(right_grid)} voxels...")
    t0 = time.time()
    
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(URDF_PATH, package_dirs, ROOT_JOINT, 
                  right_frame, right_active_q, q_ref, RIGHT_R_d, True)
    ) as pool:
        results = pool.map(process_voxel, right_grid, chunksize=100)
        
    right_reachable = np.array([p for p in results if p is not None])
    print(f"  [DONE] Right: {len(right_reachable)} points found. Time: {time.time()-t0:.1f}s")

    # ==========================================================
    # 从体素可达点中提取“表面/边界点”（更干净、更像工作空间外壳）
    # ==========================================================
    # 说明：这里用 6 邻域提取表面，得到的是“薄壳”。
    # 若你希望表面更厚/更保守，可把 connectivity 改成 26。
    left_surface = extract_surface_points(left_reachable, left_bbox, VOXEL, connectivity=6)
    right_surface = extract_surface_points(right_reachable, right_bbox, VOXEL, connectivity=6)
    print(f"[SURFACE] LEFT surface points: {len(left_surface)}")
    print(f"[SURFACE] RIGHT surface points: {len(right_surface)}")

    # ===========================================================
    # 保存结果
    # ===========================================================
    save_points(left_reachable,
                csv_path=os.path.join(OUT_DIR, "left_reachable.csv"),
                npy_path=os.path.join(OUT_DIR, "left_reachable.npy"))
    save_points(right_reachable,
                csv_path=os.path.join(OUT_DIR, "right_reachable.csv"),
                npy_path=os.path.join(OUT_DIR, "right_reachable.npy"))

    # 额外保存表面点（推荐你后续只用 surface 点做可视化/mesh 化）
    save_points(left_surface,
                csv_path=os.path.join(OUT_DIR, "left_surface.csv"),
                npy_path=os.path.join(OUT_DIR, "left_surface.npy"))
    save_points(right_surface,
                csv_path=os.path.join(OUT_DIR, "right_surface.csv"),
                npy_path=os.path.join(OUT_DIR, "right_surface.npy"))

    print("[OUT] saved to:", OUT_DIR)

    # ===========================================================
    # Meshcat 画点云：默认画“表面点”(更干净)
    # ===========================================================
    if USE_MESHCAT and viz is not None:
        viewer = viz.viewer

        # 点大小按体素尺度自适应（避免“点太小看不清/点太大糊成一团”）
        pt_size = max(0.004, 0.25 * VOXEL)

        # ------- 左臂表面 -------
        if len(left_surface) > 0:
            pts = left_surface.T  # 3 x N
            colors = np.zeros((3, pts.shape[1]))
            colors[0, :] = 1.0  # red
            viewer["reachability_left_surface"].set_object(
                g.PointCloud(position=pts, color=colors, size=pt_size)
            )

        # ------- 右臂表面 -------
        if len(right_surface) > 0:
            pts = right_surface.T
            colors = np.zeros((3, pts.shape[1]))
            colors[2, :] = 1.0  # blue
            viewer["reachability_right_surface"].set_object(
                g.PointCloud(position=pts, color=colors, size=pt_size)
            )

        # 如需对比“全部可达点”和“表面点”，把下面两段取消注释即可。
        # 注意：全点云很密，显示会更慢、也更容易出现采样条纹。
        #
        # if len(left_reachable) > 0:
        #     pts = left_reachable.T
        #     colors = np.zeros((3, pts.shape[1])); colors[0, :] = 0.7
        #     viewer["reachability_left_all"].set_object(
        #         g.PointCloud(position=pts, color=colors, size=0.15 * pt_size)
        #     )
        # if len(right_reachable) > 0:
        #     pts = right_reachable.T
        #     colors = np.zeros((3, pts.shape[1])); colors[2, :] = 0.7
        #     viewer["reachability_right_all"].set_object(
        #         g.PointCloud(position=pts, color=colors, size=0.15 * pt_size)
        #     )

        viz.display(q_ref)
        print("[VIS] Meshcat updated. You can rotate/zoom in the viewer window.")


if __name__ == "__main__":
    main()
