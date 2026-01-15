#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
Dual-arm pose-conditioned reachability via IK (cuRobo 版本)
===========================================================

目标（逻辑与 `ik-workspace.py` 完全一致）：
- 在“某一个给定末端姿态”（orientation 固定）条件下，计算左右手各自的可达空间；
- 不再手写 LM-IK，而是把 IK 求解部分换成 cuRobo 的 IK / motion generation 接口。

说明：
- 这个脚本侧重把“任务空间扫点 + IK 判定”这套逻辑平移到 cuRobo；
- 具体的 cuRobo API（例如 MotionGenConfig 的参数、机器人配置 YAML 路径、planning group 名字等）
  会随版本略有差异，这里给的是**结构完全对应**、API 细节需要你根据本地 cuRobo 示例微调的版本；
- 你可以并行保留 `ik-workspace.py`（Pinocchio 版本）和本文件（cuRobo 版本），方便对比和调试。
"""

import os
import time
import numpy as np
import pandas as pd

import torch

# === cuRobo 相关 import（根据你本地 cuRobo 版本，必要时微调路径） ===
try:
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from curobo.geom.types import Pose
    # 如果你有更合适的 IK 接口（如专门的 IKSolver），也可以在下面的 ik_solve_with_curobo 中替换
except ImportError as e:
    raise ImportError(
        "导入 cuRobo 失败，请先正确安装 cuRobo，并根据本地版本调整 import 路径。\n"
        "参考官方文档: https://curobo.org"
    ) from e


# ===========================================================
# 1) 用户可配置区：你通常只需要改这里（cuRobo 版本）
# ===========================================================

# cuRobo 的 robot/world 配置（通常是 YAML），请按你自己的工程路径修改
ROBOT_CFG_PATH = "/anyverse/curobo_cfg/robot_cfg.yaml"
WORLD_CFG_PATH = "/anyverse/curobo_cfg/world_cfg.yaml"

# 左 / 右臂在 cuRobo 配置里的 planning group 名字（需要与 YAML 一致）
LEFT_GROUP_NAME = "left_arm"
RIGHT_GROUP_NAME = "right_arm"

# 左 / 右末端 link 名（若你的 cuRobo RobotConfig 里已经指定了末端，可不强依赖这里）
LEFT_EE_LINK_NAME = "left_gripper"
RIGHT_EE_LINK_NAME = "right_gripper"

# 体素分辨率：越小越精细但越慢
VOXEL = 0.05  # 5cm

# 手动扫点空间范围（建议先粗略设一个 bbox，跑通后再精调）
# 注意：这里不再通过随机 FK 自动估计 bbox，而是直接人工给定，
#       逻辑与原脚本相同，只是 bbox 来源不同。
LEFT_BBOX = dict(x=(-0.6, 0.6), y=(-0.8, 0.2), z=(0.2, 1.2))
RIGHT_BBOX = dict(x=(-0.6, 0.6), y=(-0.2, 0.8), z=(0.2, 1.2))

# IK 收敛阈值（用在 cuRobo 求解结果的误差检查上）
EPS_POS = 2e-3   # 2mm
EPS_ROT = 3e-2   # 约 1.7deg

# 每个目标点最多尝试多少个初始值（multi-start），与原脚本 MULTI_START 对齐
MULTI_START = 6

# 输出目录
OUT_DIR = "./ik_reachability_out_curobo"

# 固定末端姿态的定义：
# - 这里直接用“世界系下的一个固定四元数”作为 R_d；
# - 通常你可以用 cuRobo / Isaac Sim 中某个标称姿态的末端姿态，抑或简单设为 identity。
FIXED_ORIENTATION_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # [w, x, y, z]


# ===========================================================
# 2) 小工具函数：与原脚本保持同样的体素 / 存储逻辑
# ===========================================================

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


def save_points(points, csv_path, npy_path):
    """
    保存结果点云到 npy 和 csv，便于后续画图/统计。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.save(npy_path, points)
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(csv_path, index=False)


def quat_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def make_pose_from_pos_quat(p: np.ndarray, q: np.ndarray) -> Pose:
    """
    把位置 + 四元数封装成 cuRobo 的 Pose 类型（假定 Pose 支持 from_list / 构造函数）。
    如果你的 cuRobo 版本 API 略不同，请在这里做一次适配即可，主逻辑无需改动。
    """
    q = quat_norm(q)
    # 典型 cuRobo Pose 定义是 [x, y, z, qw, qx, qy, qz]
    pose_vec = np.concatenate([p, q[[1, 2, 3, 0]]])  # 这里示例把 [w,x,y,z] 变成 [x,y,z,qw,qx,qy,qz]
    # 根据你的 cuRobo 版本，可能是 Pose.from_list / Pose.from_batch / Pose(...)
    return Pose.from_list(pose_vec.tolist())  # 如 API 不同，在此处统一改


# ===========================================================
# 3) cuRobo IK 封装
# ===========================================================

class CuroboIKWrapper:
    """
    把 cuRobo 的 MotionGen/IK 调用封装成一个简单接口：
        solve(group_name, target_pose, q_init) -> (success, q_sol, pos_err, rot_err)

    这样主循环的逻辑就可以一模一样地套用。
    """

    def __init__(self, robot_cfg_path, world_cfg_path):
        # 根据 cuRobo 官方示例初始化 MotionGen
        # 注意：不同版本的 MotionGenConfig / MotionGen 构造参数可能略有不同，
        #       若报错，请参考你本地的 examples 进行微调，但对本脚本逻辑影响不大。
        print("[INFO] 初始化 cuRobo MotionGen ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg_path,
            world_cfg_path,
        )
        self.motion_gen = MotionGen(motion_gen_cfg, self.device)
        # 预热，避免第一次调用时开销太大
        self.motion_gen.warmup()

    def get_neutral_q(self, group_name: str) -> np.ndarray:
        """
        获取该 planning group 的一个“中立姿态” joint 配置，用作 q_ref。
        这里假设 cuRobo 提供某种方式得到 neutral / default state；
        若你本地 API 不同，可以直接在此处返回你想要的 q_ref（np.ndarray）。
        """
        # 下面是示意写法：从 cfg 里取默认 joint pos
        # （不同版本字段名可能不一样，如果这里报错，可以简单改成读取你的 YAML 或手写一个 q_ref）
        cfg = self.motion_gen.robot_cfg  # 典型命名，必要时根据实际属性调整
        if hasattr(cfg, "get_neutral_state"):
            q = cfg.get_neutral_state(group_name)
            return q.cpu().numpy().astype(float).ravel()
        else:
            raise NotImplementedError(
                "请在 CuroboIKWrapper.get_neutral_q 中实现获取该 group 的参考姿态 q_ref，"
                "可以从 robot_cfg 里读，也可以手动写死一个 np.ndarray。"
            )

    def solve(
        self,
        group_name: str,
        target_pose: Pose,
        q_init_list: list[np.ndarray],
    ):
        """
        使用 cuRobo 对单个末端目标 pose 做 IK / 运动生成：
        - q_init_list: 多个初始值（与原脚本 MULTI_START 对应）
        返回：
          success, q_sol(best), pos_err, rot_err
        """
        best_q = None
        best_pos_err = None
        best_rot_err = None
        success_any = False

        # 这里用最朴素的模式：依次用不同初始值调用 cuRobo 生成一条到达 target_pose 的轨迹，
        # 如果规划成功，就取末端点做一次 FK 校验误差。

        for q0_np in q_init_list:
            q0 = torch.tensor(q0_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 典型 cuRobo 调用：plan_single(start_state, target_pose, group_name=...)
            # 注意：不同版本函数名/参数可能略有不同，如果你这里报错，请打开 cuRobo 自带的
            #       Python 示例，找到对应的调用方式，然后在本函数里做一次适配即可。
            try:
                result = self.motion_gen.plan_single(
                    start_state=q0,
                    goal_pose=target_pose,
                    group_name=group_name,
                )
            except TypeError:
                # 若参数名不同，可以在这里加一些备用写法或提示
                raise TypeError(
                    "MotionGen.plan_single 的参数签名与本脚本不符，请根据你本地 cuRobo 版本修改 "
                    "CuroboIKWrapper.solve 中的调用方式。"
                )

            if not result.success:
                continue

            # 从结果中取最终关节配置和末端位姿（字段名需根据实际 cuRobo 版本调整）
            q_traj = result.q  # (T, nq)
            ee_traj = result.ee_poses  # (T, 7) or (T, 4,4) 等
            q_final = q_traj[-1].detach().cpu().numpy().astype(float).ravel()

            # 末端位姿误差：用 cuRobo 返回的最终末端 pose 与 target_pose 比较
            # 这里假设 ee_traj[-1] 也是 [x,y,z,qw,qx,qy,qz] 形式的向量；
            # 如果返回的是 SE3 矩阵，请在此处做一次转换再算误差。
            ee_final = ee_traj[-1].detach().cpu().numpy().astype(float).ravel()
            p_final = ee_final[:3]
            q_final_ee = ee_final[3:]

            # 目标 pose 解包
            target_vec = np.array(target_pose.data, dtype=float).ravel()
            p_target = target_vec[:3]
            q_target = target_vec[3:]

            pos_err = np.linalg.norm(p_final - p_target)
            rot_err = np.linalg.norm(q_final_ee - q_target)

            if pos_err < EPS_POS and rot_err < EPS_ROT:
                # 一旦达到收敛阈值，直接返回
                return True, q_final, pos_err, rot_err

            # 否则选一个误差最小的解作为备选
            if (best_pos_err is None) or (pos_err + rot_err < best_pos_err + best_rot_err):
                success_any = True
                best_q = q_final
                best_pos_err = pos_err
                best_rot_err = rot_err

        if success_any:
            return True, best_q, best_pos_err, best_rot_err
        else:
            return False, None, None, None


# ===========================================================
# 4) 主流程：逻辑与 `ik-workspace.py` 对齐，只是把 IK 引擎换成 cuRobo
# ===========================================================


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 初始化 cuRobo IK wrapper
    ik = CuroboIKWrapper(ROBOT_CFG_PATH, WORLD_CFG_PATH)

    # 参考姿态 q_ref（“给定全身姿态”），用 cuRobo 的 neutral 姿态表示
    # 如果你有更精确的“mentor 指定姿态”，可以在 get_neutral_q 中或这里手动替换。
    q_ref_left = ik.get_neutral_q(LEFT_GROUP_NAME)
    q_ref_right = ik.get_neutral_q(RIGHT_GROUP_NAME)

    # 扫点体素网格
    left_grid = build_voxel_grid(LEFT_BBOX, VOXEL)
    right_grid = build_voxel_grid(RIGHT_BBOX, VOXEL)
    print("[INFO] LEFT grid points:", len(left_grid))
    print("[INFO] RIGHT grid points:", len(right_grid))

    # 固定末端姿态（orientation）
    quat_d = FIXED_ORIENTATION_QUAT

    # =======================================================
    # 左臂 IK 扫描（与原来 Pinocchio 版本的结构对应）
    # =======================================================
    left_reachable = []
    q_prev = q_ref_left.copy()

    print("[RUN] Solving IK for LEFT arm with cuRobo ...")
    t0 = time.time()
    for i, p in enumerate(left_grid):
        target_pose = make_pose_from_pos_quat(p, quat_d)

        # multi-start 初始值：上一解 + 参考姿态 + 随机扰动
        inits = [q_prev, q_ref_left]
        for _ in range(max(0, MULTI_START - len(inits))):
            noise = np.random.uniform(-0.1, 0.1, size=q_ref_left.shape)
            inits.append(q_ref_left + noise)

        success, q_sol, pos_err, rot_err = ik.solve(
            LEFT_GROUP_NAME,
            target_pose,
            q_init_list=inits,
        )
        if success:
            left_reachable.append(p)
            if q_sol is not None:
                q_prev = q_sol

        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            print(f"  LEFT {i+1}/{len(left_grid)} | reachable={len(left_reachable)} | {dt:.1f}s")

    left_reachable = np.array(left_reachable, dtype=float)
    print(f"[DONE] LEFT reachable points: {len(left_reachable)}")

    # =======================================================
    # 右臂 IK 扫描（同理）
    # =======================================================
    right_reachable = []
    q_prev = q_ref_right.copy()

    print("[RUN] Solving IK for RIGHT arm with cuRobo ...")
    t0 = time.time()
    for i, p in enumerate(right_grid):
        target_pose = make_pose_from_pos_quat(p, quat_d)

        inits = [q_prev, q_ref_right]
        for _ in range(max(0, MULTI_START - len(inits))):
            noise = np.random.uniform(-0.1, 0.1, size=q_ref_right.shape)
            inits.append(q_ref_right + noise)

        success, q_sol, pos_err, rot_err = ik.solve(
            RIGHT_GROUP_NAME,
            target_pose,
            q_init_list=inits,
        )
        if success:
            right_reachable.append(p)
            if q_sol is not None:
                q_prev = q_sol

        if (i + 1) % 2000 == 0:
            dt = time.time() - t0
            print(f"  RIGHT {i+1}/{len(right_grid)} | reachable={len(right_reachable)} | {dt:.1f}s")

    right_reachable = np.array(right_reachable, dtype=float)
    print(f"[DONE] RIGHT reachable points: {len(right_reachable)}")

    # =======================================================
    # 保存结果（与原脚本命名类似，但放在不同目录中）
    # =======================================================
    save_points(
        left_reachable,
        csv_path=os.path.join(OUT_DIR, "left_reachable.csv"),
        npy_path=os.path.join(OUT_DIR, "left_reachable.npy"),
    )
    save_points(
        right_reachable,
        csv_path=os.path.join(OUT_DIR, "right_reachable.csv"),
        npy_path=os.path.join(OUT_DIR, "right_reachable.npy"),
    )

    print("[OUT] saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

