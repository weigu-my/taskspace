#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
cuRobo 机械臂可达性分析（GPU + 多种指标）
===========================================================

功能:
1. 使用 cuRobo 的 XRDF 配置（直接使用 XRDF）
2. 体素化空间 + GPU IK 批量求解
3. 多 seeds IK 统计解的数量
4. 计算 dexterity（可达姿态数）
5. 计算 Yoshikawa manipulability
6. 导出可视化数据
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import importlib.util
import inspect
import numpy as np
import torch

if importlib.util.find_spec("curobo") is None:
    CUROBO_AVAILABLE = False
else:
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    CUROBO_AVAILABLE = True


@dataclass
class ArmConfig:
    name: str
    xrdf_path: str
    num_seeds: int = 32
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05


@dataclass
class ReachabilityConfig:
    arm: ArmConfig
    voxel_size: float = 0.05
    bbox: Optional[dict] = None
    num_orientations: int = 12
    batch_size: int = 1024
    output_dir: str = "./reachability_output"
    random_seed: int = 42
    solution_uniqueness_tol: float = 1e-3


@dataclass
class ReachabilityResult:
    reachable_points: np.ndarray
    dexterity: np.ndarray
    solution_count: np.ndarray
    manipulability: np.ndarray
    orientations_tested: int
    voxel_size: float
    bbox: dict
    computation_time_s: float


def sample_uniform_quaternions(n: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, (n, 3))
    else:
        u = np.random.uniform(0.0, 1.0, (n, 3))

    sqrt1_u0 = np.sqrt(1 - u[:, 0])
    sqrt_u0 = np.sqrt(u[:, 0])
    quats = np.zeros((n, 4))
    quats[:, 0] = sqrt1_u0 * np.sin(2 * np.pi * u[:, 1])
    quats[:, 1] = sqrt1_u0 * np.cos(2 * np.pi * u[:, 1])
    quats[:, 2] = sqrt_u0 * np.sin(2 * np.pi * u[:, 2])
    quats[:, 3] = sqrt_u0 * np.cos(2 * np.pi * u[:, 2])
    return quats


def build_voxel_grid(bbox: dict, voxel_size: float) -> np.ndarray:
    xs = np.arange(bbox["x"][0], bbox["x"][1] + 1e-9, voxel_size)
    ys = np.arange(bbox["y"][0], bbox["y"][1] + 1e-9, voxel_size)
    zs = np.arange(bbox["z"][0], bbox["z"][1] + 1e-9, voxel_size)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def unique_rows_tol(rows: np.ndarray, tol: float) -> np.ndarray:
    if len(rows) == 0:
        return rows
    rounded = np.round(rows / tol) * tol
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return rows[np.sort(idx)]


class CuroboMultiSeedIK:
    def __init__(self, arm: ArmConfig, device: torch.device):
        if not CUROBO_AVAILABLE:
            raise RuntimeError("未安装 cuRobo")

        self.arm = arm
        self.device = device
        self.tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

        robot_config = self._load_robot_config(arm)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            world_cfg=None,
            rotation_threshold=arm.rotation_threshold,
            position_threshold=arm.position_threshold,
            num_seeds=arm.num_seeds,
            tensor_args=self.tensor_args,
            use_cuda_graph=False,
        )

        self.ik_solver = IKSolver(ik_config)
        self.robot_model = self.ik_solver.robot_config.kinematics
        self.joint_names = self.robot_model.joint_names
        self.supports_seed_q = "seed_q" in inspect.signature(self.ik_solver.solve_batch).parameters
        self.joint_limits = self.robot_model.get_joint_limits().position

        self._warmup()

    def _load_robot_config(self, arm: ArmConfig):
        if arm.xrdf_path:
            if hasattr(RobotConfig, "from_xrdf"):
                return RobotConfig.from_xrdf(arm.xrdf_path, self.tensor_args)
            raise RuntimeError("当前 cuRobo 版本不支持 XRDF 加载")

        raise RuntimeError("未提供 XRDF")

    def _warmup(self):
        dummy_pos = torch.zeros((1, 3), device=self.device)
        dummy_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        goal = Pose(position=dummy_pos, quaternion=dummy_quat)
        _ = self.ik_solver.solve_batch(goal)

    def solve(self, positions: np.ndarray, orientations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.float32)
        quat_tensor = torch.tensor(orientations, device=self.device, dtype=torch.float32)
        goal = Pose(position=pos_tensor, quaternion=quat_tensor)

        if self.supports_seed_q and self.arm.num_seeds > 1:
            seed_q = self._sample_seeds(self.arm.num_seeds)
            seed_q_tensor = torch.tensor(seed_q, device=self.device, dtype=torch.float32)
            result = self.ik_solver.solve_batch(goal, seed_q=seed_q_tensor)
        else:
            result = self.ik_solver.solve_batch(goal)
        success = result.success.detach().cpu().numpy()
        solutions = None

        if hasattr(result, "solution") and result.solution is not None:
            solutions = result.solution.detach().cpu().numpy()

        return success, solutions

    def _sample_seeds(self, num_seeds: int) -> np.ndarray:
        lower = self.joint_limits[0].detach().cpu().numpy()
        upper = self.joint_limits[1].detach().cpu().numpy()
        return np.random.uniform(lower, upper, size=(num_seeds, len(lower)))

    def compute_manipulability(self, q_solutions: np.ndarray) -> np.ndarray:
        if q_solutions.size == 0:
            return np.array([])

        q_tensor = torch.tensor(q_solutions, device=self.device, dtype=torch.float32)
        state = JointState.from_position(q_tensor, self.joint_names)
        jacobian = self.robot_model.get_jacobian(state)
        jac = jacobian.detach().cpu()
        jj_t = jac @ jac.transpose(-1, -2)
        det = torch.linalg.det(jj_t)
        det = torch.clamp(det, min=0.0)
        manipulability = torch.sqrt(det).numpy()
        return manipulability


class ReachabilityAnalyzer:
    def __init__(self, config: ReachabilityConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.arm = config.arm
        if not self.arm.xrdf_path:
            raise RuntimeError("需要提供 XRDF 路径进行可达性分析")

        self.solver = CuroboMultiSeedIK(self.arm, self.device)

    def _estimate_bbox(self, num_samples: int = 5000) -> dict:
        lower, upper = self.solver.robot_model.get_joint_limits().position
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

        q_samples = np.random.uniform(lower, upper, size=(num_samples, len(lower)))
        q_tensor = torch.tensor(q_samples, device=self.device, dtype=torch.float32)
        state = JointState.from_position(q_tensor, self.solver.joint_names)
        fk_result = self.solver.robot_model.get_state(state)
        ee_pos = fk_result.ee_position.detach().cpu().numpy()

        margin = 0.05
        return {
            "x": (float(ee_pos[:, 0].min() - margin), float(ee_pos[:, 0].max() + margin)),
            "y": (float(ee_pos[:, 1].min() - margin), float(ee_pos[:, 1].max() + margin)),
            "z": (float(ee_pos[:, 2].min() - margin), float(ee_pos[:, 2].max() + margin)),
        }

    def analyze(self) -> ReachabilityResult:
        if self.config.bbox is None:
            bbox = self._estimate_bbox()
        else:
            bbox = self.config.bbox

        grid_points = build_voxel_grid(bbox, self.config.voxel_size)
        num_points = len(grid_points)

        orientations = sample_uniform_quaternions(
            self.config.num_orientations,
            seed=self.config.random_seed,
        )

        dexterity = np.zeros(num_points, dtype=np.int32)
        solution_count = np.zeros(num_points, dtype=np.int32)
        manipulability = np.zeros(num_points, dtype=np.float32)

        batch_size = self.config.batch_size
        start_time = time.time()

        for ori_idx, quat in enumerate(orientations):
            quat_batch = np.tile(quat, (num_points, 1))
            for start in range(0, num_points, batch_size):
                end = min(start + batch_size, num_points)
                batch_pos = grid_points[start:end]
                batch_quat = quat_batch[start:end]

                success, solutions = self.solver.solve(batch_pos, batch_quat)
                if success.ndim == 1:
                    success_mask = success.astype(bool)
                    dexterity[start:end] += success_mask.astype(np.int32)

                    if solutions is not None and solutions.ndim == 2:
                        solution_count[start:end] += success_mask.astype(np.int32)
                        success_indices = np.where(success_mask)[0]
                        if len(success_indices) > 0:
                            manip = self.solver.compute_manipulability(solutions[success_indices])
                            for offset, value in zip(success_indices, manip):
                                idx = start + offset
                                manipulability[idx] = max(manipulability[idx], float(value))
                else:
                    success_mask = success.astype(bool)
                    dexterity[start:end] += success_mask.any(axis=1).astype(np.int32)

                    if solutions is not None:
                        for idx in range(success_mask.shape[0]):
                            valid = solutions[idx][success_mask[idx]]
                            unique = unique_rows_tol(valid, self.config.solution_uniqueness_tol)
                            solution_count[start + idx] += len(unique)
                            if len(unique) > 0:
                                manip = self.solver.compute_manipulability(unique)
                                if len(manip) > 0:
                                    manipulability[start + idx] = max(
                                        manipulability[start + idx],
                                        float(np.max(manip)),
                                    )

            elapsed = time.time() - start_time
            print(
                f"[INFO] 姿态 {ori_idx + 1}/{len(orientations)} 完成，"
                f"耗时 {elapsed:.1f}s"
            )

        total_time = time.time() - start_time

        reachable_mask = dexterity > 0
        result = ReachabilityResult(
            reachable_points=grid_points[reachable_mask],
            dexterity=dexterity[reachable_mask],
            solution_count=solution_count[reachable_mask],
            manipulability=manipulability[reachable_mask],
            orientations_tested=len(orientations),
            voxel_size=self.config.voxel_size,
            bbox=bbox,
            computation_time_s=total_time,
        )

        self._save_results(result)
        return result

    def _save_results(self, result: ReachabilityResult):
        prefix = self.output_dir / self.arm.name

        data = np.column_stack(
            [
                result.reachable_points,
                result.dexterity,
                result.solution_count,
                result.manipulability,
            ]
        )
        np.save(f"{prefix}_reachable_metrics.npy", data)

        header = "x,y,z,dexterity,solution_count,manipulability"
        np.savetxt(
            f"{prefix}_reachable_metrics.csv",
            data,
            delimiter=",",
            header=header,
            comments="",
        )

        stats = {
            "arm_name": self.arm.name,
            "total_reachable": int(result.reachable_points.shape[0]),
            "orientations_tested": result.orientations_tested,
            "voxel_size": result.voxel_size,
            "bbox": result.bbox,
            "dexterity_mean": float(result.dexterity.mean()) if len(result.dexterity) else 0.0,
            "solution_count_mean": float(result.solution_count.mean()) if len(result.solution_count) else 0.0,
            "manipulability_mean": float(result.manipulability.mean()) if len(result.manipulability) else 0.0,
            "computation_time_s": result.computation_time_s,
        }
        with open(f"{prefix}_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] 结果保存到: {prefix}_reachable_metrics.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="cuRobo 可达性分析")
    parser.add_argument("--xrdf", required=True, help="XRDF 路径")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素大小")
    parser.add_argument("--num-orientations", type=int, default=12, help="姿态采样数")
    parser.add_argument("--num-seeds", type=int, default=32, help="IK seeds 数量")
    parser.add_argument("--batch-size", type=int, default=1024, help="IK batch size")
    parser.add_argument("--output-dir", default="./reachability_output", help="输出目录")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="工作空间范围",
    )

    args = parser.parse_args()

    bbox = None
    if args.bbox:
        bbox = {
            "x": (args.bbox[0], args.bbox[1]),
            "y": (args.bbox[2], args.bbox[3]),
            "z": (args.bbox[4], args.bbox[5]),
        }

    arm = ArmConfig(
        name="robot_arm",
        xrdf_path=args.xrdf,
        num_seeds=args.num_seeds,
    )

    config = ReachabilityConfig(
        arm=arm,
        voxel_size=args.voxel_size,
        bbox=bbox,
        num_orientations=args.num_orientations,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    analyzer = ReachabilityAnalyzer(config)
    analyzer.analyze()


if __name__ == "__main__":
    main()
