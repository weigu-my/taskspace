"""
动态可达性过滤器

根据对侧臂当前关节姿态，实时从预计算的最大可达空间中排除碰撞区域。
每次过滤都从最大可达空间重新开始，确保姿态变化时可达空间能正确恢复。

核心思路：
- 预计算阶段：只考虑固定身体（躯干/底盘）的碰撞，得到"最大可达空间"
- 动态阶段：用 pinocchio FK 计算对侧臂在当前姿态下的碰撞球位置，
            从最大可达空间中减去碰撞区域
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple


class DynamicReachabilityFilter:
    """
    动态可达性过滤器

    根据对侧臂的碰撞球在当前姿态下的位置，
    从预计算的最大可达空间中过滤掉碰撞点。
    """

    def __init__(
        self,
        urdf_path: str,
        arm_name: str,
        base_link: str,
        max_reachable_mask: np.ndarray,
        grid_points: np.ndarray,
        dexterity: np.ndarray,
        manipulability: np.ndarray,
        other_arm_spheres_local: Dict[str, List[dict]],
    ):
        """
        参数:
            urdf_path: URDF 文件路径
            arm_name: 当前臂名称 ('left'/'right')
            base_link: 当前臂的 base_link 名称
            max_reachable_mask: 预计算的最大可达掩码 [N] (不含对侧臂碰撞)
            grid_points: 所有网格点 [N, 3] (base_link 坐标系)
            dexterity: 灵活度 [N]
            manipulability: 可操作度 [N]
            other_arm_spheres_local: 对侧臂各 link 的碰撞球 (link 局部坐标系)
                {link_name: [{'center': [x,y,z], 'radius': r}, ...]}
        """
        self.arm_name = arm_name
        self.base_link = base_link
        self.max_mask = max_reachable_mask.copy()
        self.grid_points = grid_points
        self.dexterity = dexterity
        self.manipulability = manipulability
        self.other_arm_spheres_local = other_arm_spheres_local

        # 预计算可达索引（不变）
        self._max_reachable_idx = np.where(self.max_mask)[0]
        self._max_reachable_points = self.grid_points[self._max_reachable_idx]

        # pinocchio FK 模型
        import pinocchio as pin
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        self._pin = pin

        # 缓存 frame_id 映射
        self.frame_map = {}
        for i, f in enumerate(self.pin_model.frames):
            self.frame_map[f.name] = i

        # 验证 base_link 存在
        if base_link not in self.frame_map:
            raise ValueError(f"base_link '{base_link}' 不在 URDF frame 中")

        # 缓存对侧臂有效 link
        self._valid_other_links = [
            ln for ln in other_arm_spheres_local if ln in self.frame_map
        ]

        n_spheres = sum(len(other_arm_spheres_local[ln]) for ln in self._valid_other_links)
        print(f"[动态过滤] {arm_name} 臂: 对侧臂 {len(self._valid_other_links)} 个 link, "
              f"{n_spheres} 个碰撞球")

        # 当前过滤后的掩码（初始 = 最大可达空间）
        self.current_mask = self.max_mask.copy()

    def filter(self, joint_positions: Dict[str, float]) -> np.ndarray:
        """
        根据当前全机器人关节姿态过滤可达点

        参数:
            joint_positions: {关节名: 角度} 全机器人的关节状态

        返回:
            filtered_mask: [N] 过滤后的可达掩码
        """
        pin = self._pin

        # 1. 构建 pinocchio 关节向量
        q = pin.neutral(self.pin_model)
        for i in range(1, self.pin_model.njoints):
            jname = self.pin_model.names[i]
            if jname in joint_positions:
                idx_q = self.pin_model.joints[i].idx_q
                nq = self.pin_model.joints[i].nq
                if nq == 1:
                    q[idx_q] = joint_positions[jname]

        # 2. FK
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # 3. 收集对侧臂碰撞球在当前臂 base_link 坐标系中的位置
        base_fid = self.frame_map[self.base_link]
        centers = []
        radii = []

        for link_name in self._valid_other_links:
            fid = self.frame_map[link_name]
            # link 在当前臂 base_link 中的位姿
            link_in_base = self.pin_data.oMf[base_fid].actInv(
                self.pin_data.oMf[fid]
            )
            R = link_in_base.rotation
            t = link_in_base.translation

            for s in self.other_arm_spheres_local[link_name]:
                c_local = np.array(s['center'])
                c_base = R @ c_local + t
                centers.append(c_base)
                radii.append(s['radius'])

        if not centers:
            # 没有对侧臂碰撞球，返回最大可达空间
            self.current_mask = self.max_mask.copy()
            return self.current_mask

        centers = np.array(centers)  # [M, 3]
        radii = np.array(radii)      # [M]

        # 4. 几何过滤：从最大可达空间中排除碰撞点
        # 只检查最大可达空间中的点（节省计算）
        diff = self._max_reachable_points[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # [K, M]
        inside = np.any(dists < radii[np.newaxis, :], axis=1)  # [K]

        # 5. 从最大可达掩码的副本开始（每次重新开始！）
        self.current_mask = self.max_mask.copy()
        # 将碰撞点标记为不可达
        collision_indices = self._max_reachable_idx[inside]
        self.current_mask[collision_indices] = False

        return self.current_mask

    @property
    def n_filtered_out(self) -> int:
        """当前被对侧臂碰撞过滤掉的点数"""
        return int(np.sum(self.max_mask)) - int(np.sum(self.current_mask))

    def save_config(self, output_dir: str):
        """
        保存动态过滤配置到磁盘

        参数:
            output_dir: 输出目录
        """
        config = {
            'arm_name': self.arm_name,
            'base_link': self.base_link,
            'other_arm_spheres_local': {
                ln: [{'center': s['center'] if isinstance(s['center'], list) else s['center'].tolist(),
                      'radius': float(s['radius'])} for s in spheres]
                for ln, spheres in self.other_arm_spheres_local.items()
            }
        }
        path = os.path.join(output_dir, 'dynamic_filter_config.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[动态过滤] 已保存配置: {path}")

    @staticmethod
    def load_config(output_dir: str) -> Optional[dict]:
        """从磁盘加载动态过滤配置"""
        path = os.path.join(output_dir, 'dynamic_filter_config.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
