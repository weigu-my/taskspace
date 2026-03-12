"""
多种子IK求解器 - 使用cuRobo实现GPU加速

本模块提供基于多种子策略的IK求解功能，用于为冗余机械臂寻找多个IK解。
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from .config import IKConfig, ReachabilityConfig
from .urdf_parser import RobotConfig, CuroboConfigGenerator


@dataclass
class IKResult:
    """单个目标位姿的IK求解结果"""
    target_position: np.ndarray       # [3] 目标位置
    target_orientation: np.ndarray    # [4] 目标姿态四元数 (w, x, y, z)
    success: bool                     # 是否找到至少一个解
    num_solutions: int                # 找到的有效IK解数量
    solutions: np.ndarray             # [num_seeds, n_joints] 关节配置
    solution_mask: np.ndarray         # [num_seeds] 有效解的布尔掩码
    position_errors: np.ndarray       # [num_seeds] 位置误差
    rotation_errors: np.ndarray       # [num_seeds] 旋转误差


@dataclass
class BatchIKResult:
    """批量IK求解结果"""
    success_mask: np.ndarray          # [batch_size] 每个位姿是否有解
    num_solutions: np.ndarray         # [batch_size] 每个位姿的解数量
    best_solutions: np.ndarray        # [batch_size, n_joints] 最优关节配置
    all_solutions: np.ndarray         # [batch_size, num_seeds, n_joints] 所有解
    solution_masks: np.ndarray        # [batch_size, num_seeds] 有效性掩码
    solve_time: float                 # 总求解时间


class MultiSeedIKSolver:
    """
    多种子IK求解器 - 使用cuRobo进行GPU加速计算

    该求解器从多个初始种子运行IK，以找到多个解，
    这对于分析冗余机械臂的可达性至关重要。
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        ik_config: IKConfig,
        device: str = "cuda:0",
        world_config: Optional[Any] = None,
        other_arm_links: Optional[set] = None,
        dynamic_mode: bool = False
    ):
        """
        初始化多种子IK求解器

        参数:
            robot_config: 从URDF解析器获取的机器人配置
            ik_config: IK求解器配置
            device: 计算设备
            world_config: cuRobo WorldConfig对象，用于环境碰撞检测
                         可以包含机器人躯体、另一臂等障碍物
            other_arm_links: 对侧臂的 link 名称集合（动态模式下从静态障碍物中排除）
            dynamic_mode: 是否启用动态可达性模式
        """
        self.robot_config = robot_config
        self.ik_config = ik_config
        self.device = device
        self.world_config = world_config  # 环境碰撞配置
        self._other_arm_links = other_arm_links or set()
        self._dynamic_mode = dynamic_mode
        self.ik_solver = None
        self.tensor_args = None
        self._current_batch_size = None  # 跟踪当前批次大小
        self._robot_cfg = None  # 缓存cuRobo机器人配置

        # 用于坐标变换的 Pinocchio 模型
        self.pin_model = None
        self.pin_data = None
        self.base_link_frame_id = None
        self._base_transform = None  # 世界坐标系到 base_link 的变换

        self._initialize_solver()
        self._initialize_transforms()

    def _initialize_solver(self):
        """初始化cuRobo IK求解器"""
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import RobotConfig as CuroboRobotConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

            # 设置张量参数
            self.tensor_args = TensorDeviceType(
                device=torch.device(self.device),
                dtype=torch.float32
            )

            # 生成cuRobo配置
            config_generator = CuroboConfigGenerator(self.robot_config)
            curobo_config = config_generator.generate()

            # 尝试从 mesh 生成碰撞球（更精确），否则回退到简单球
            if self.ik_config.self_collision_check:
                mesh_collision = self._generate_mesh_collision_spheres()
                if mesh_collision:
                    # 用 mesh 碰撞球替换简单碰撞球
                    if 'robot_cfg' not in curobo_config:
                        curobo_config['robot_cfg'] = {}
                    curobo_config['robot_cfg']['collision'] = mesh_collision

            # 为cuRobo创建机器人配置
            robot_cfg = self._create_curobo_robot_config(curobo_config)
            self._robot_cfg = robot_cfg  # 缓存配置用于后续更新

            # 构建环境碰撞模型（包含躯干/底盘等非链 link 的碰撞球）
            world_model = self.world_config
            if (self.ik_config.self_collision_check
                    and hasattr(self, '_body_collision_spheres')
                    and self._body_collision_spheres):
                world_model = self._build_body_world_config(world_model)

            # 创建IK求解器配置
            # 当有环境碰撞球时使用 PRIMITIVE 碰撞检查器
            from curobo.geom.sdf.world import CollisionCheckerType
            checker_type = None  # 默认让 cuRobo 选择
            if world_model is not None:
                checker_type = CollisionCheckerType.PRIMITIVE

            ik_solver_config = IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=world_model,
                tensor_args=self.tensor_args,
                num_seeds=self.ik_config.num_seeds,
                position_threshold=self.ik_config.position_threshold,
                rotation_threshold=self.ik_config.rotation_threshold,
                use_cuda_graph=False,
                self_collision_check=self.ik_config.self_collision_check,
                self_collision_opt=self.ik_config.self_collision_check,
                collision_checker_type=checker_type,
            )

            self.ik_solver = IKSolver(ik_solver_config)
            self.n_joints = len(self.robot_config.active_joints)

            print(f"[IK求解器] 已初始化，种子数: {self.ik_config.num_seeds}")
            print(f"[IK求解器] 机器人: {self.robot_config.name}, 自由度: {self.n_joints}")
            print(f"[IK求解器] 设备: {self.device}")
            print(f"[IK求解器] 自碰撞检测: {'启用' if self.ik_config.self_collision_check else '禁用'}")
            has_world = world_model is not None
            print(f"[IK求解器] 环境碰撞检测: {'启用' if has_world else '禁用'}")

        except ImportError as e:
            print(f"[警告] cuRobo不可用: {e}")
            print("[警告] 使用模拟求解器")
            self._initialize_dummy_solver()

    def update_world_config(self, world_config: Optional[Any]) -> None:
        """
        动态更新环境碰撞配置

        这会重新创建IK求解器以应用新的碰撞配置。
        用于在固定臂姿态变化后更新碰撞障碍物。

        参数:
            world_config: 新的cuRobo WorldConfig对象
        """
        self.world_config = world_config
        if self.ik_solver is not None:
            # 重新初始化求解器以应用新的world_config
            self._initialize_solver()
            print(f"[IK求解器] 已更新环境碰撞配置")

    def _create_curobo_robot_config(self, config_dict: Dict) -> Any:
        """从字典创建cuRobo机器人配置（含碰撞球用于自碰撞检测）"""
        from curobo.types.robot import RobotConfig as CuroboRobotConfig

        # 获取活动关节
        active_joints = self.robot_config.active_joints
        joint_names = [j.name for j in active_joints]

        # 创建配置字典
        robot_cfg_dict = {
            'kinematics': {
                'urdf_path': self.robot_config.urdf_path,
                'asset_root_path': self.robot_config.urdf_dir,
                'base_link': self.robot_config.base_link,
                'ee_link': self.robot_config.ee_link,
                'cspace': {
                    'joint_names': joint_names,
                    'retract_config': [(j.lower_limit + j.upper_limit) / 2 for j in active_joints],
                    'null_space_weight': [1.0] * len(joint_names),
                    'cspace_distance_weight': [1.0] * len(joint_names),
                },
            },
        }

        # 从 CuroboConfigGenerator 生成的配置中提取碰撞球数据
        # 碰撞球必须放入 kinematics 字典中（CudaRobotGeneratorConfig 的字段）
        # 而不是作为 RobotConfig 的顶层 key
        collision_data = None
        if 'robot_cfg' in config_dict and 'collision' in config_dict['robot_cfg']:
            collision_data = config_dict['robot_cfg']['collision']
        elif 'collision' in config_dict:
            collision_data = config_dict['collision']

        if collision_data:
            kin = robot_cfg_dict['kinematics']
            if 'collision_spheres' in collision_data:
                kin['collision_spheres'] = collision_data['collision_spheres']
                # 碰撞 link 名称列表
                kin['collision_link_names'] = list(collision_data['collision_spheres'].keys())
            # self_collision_buffer 和 self_collision_ignore 必须为 dict（不能为 None）
            # 否则 cuRobo 内部调用 .copy() / .keys() 会报 AttributeError
            kin['self_collision_buffer'] = collision_data.get('self_collision_buffer', {}) or {}
            kin['self_collision_ignore'] = collision_data.get('self_collision_ignore', {}) or {}
            if 'collision_sphere_buffer' in collision_data:
                kin['collision_sphere_buffer'] = collision_data['collision_sphere_buffer']
            print(f"[IK求解器] 已加载碰撞球配置（{len(kin.get('collision_link_names', []))} 个链接），用于自碰撞检测")
        else:
            print(f"[IK求解器] 警告: 未找到碰撞球配置，自碰撞检测可能不生效")

        robot_cfg = CuroboRobotConfig.from_dict(
            robot_cfg_dict,
            self.tensor_args
        )

        return robot_cfg

    def _generate_mesh_collision_spheres(self) -> Optional[Dict]:
        """
        从 URDF mesh 文件生成碰撞球配置

        只为 arm chain 上的 link 生成碰撞球（用于 cuRobo 自碰撞检测）。
        非 chain link（躯干/底盘等）的碰撞球作为环境障碍物处理。
        """
        try:
            from arm_reachability.collision_spheres_generator import (
                generate_collision_spheres_yaml
            )
            # 获取运动链上的 link 名称
            chain_links = set(self.robot_config.chain_links) if self.robot_config.chain_links else set()
            if not chain_links:
                chain_links = {self.robot_config.base_link, self.robot_config.ee_link}

            # 排除以下 link 出环境碰撞:
            # 1) chain link 的所有子孙（夹爪等，随臂运动）
            # 2) chain base 的直接父/祖父 link（如 Trunk_Link4，会与臂基座碰撞球重叠）
            child_map = {}
            parent_map = {}  # child -> parent
            for j in self.robot_config.joints:
                child_map.setdefault(j.parent_link, []).append(j.child_link)
                parent_map[j.child_link] = j.parent_link

            # BFS 找 chain 中所有 link 的后代
            moving_links = set(chain_links)
            stack = list(chain_links)
            while stack:
                current = stack.pop()
                for child in child_map.get(current, []):
                    if child not in moving_links:
                        moving_links.add(child)
                        stack.append(child)

            # 沿 fixed joint 向上追溯，排除与臂基座刚性连接的 link
            # （它们和臂基座永远相对静止，碰撞球会重叠）
            current_link = self.robot_config.base_link
            while True:
                parent = parent_map.get(current_link)
                if not parent:
                    break
                # 找连接 current_link 的 joint
                joint_type = None
                for j in self.robot_config.joints:
                    if j.child_link == current_link:
                        joint_type = j.type
                        break
                if joint_type == 'fixed':
                    moving_links.add(parent)
                    current_link = parent
                else:
                    break  # 遇到非 fixed joint 就停止

            # 动态模式下：对侧臂 link 也排除出静态障碍物，单独保存用于动态过滤
            exclude_from_body = set(moving_links)
            if self._dynamic_mode and self._other_arm_links:
                exclude_from_body |= self._other_arm_links
                print(f"[IK求解器] 动态模式: 对侧臂 {len(self._other_arm_links)} 个 link 排除出静态障碍物")

            print(f"[IK求解器] 从 URDF mesh 生成碰撞球（chain links: {len(chain_links)}，排除 links: {len(exclude_from_body)}）...")

            # 只为 chain links 生成碰撞球（用于自碰撞检测）
            config = generate_collision_spheres_yaml(
                urdf_path=self.robot_config.urdf_path,
                output_path=None,
                max_spheres_per_link=8,
                include_links=list(chain_links),
            )

            # 为非运动 links 生成环境碰撞障碍物
            body_config = generate_collision_spheres_yaml(
                urdf_path=self.robot_config.urdf_path,
                output_path=None,
                max_spheres_per_link=6,
                exclude_links=list(exclude_from_body),
            )

            if body_config and body_config.get('collision_spheres'):
                self._body_collision_spheres = body_config['collision_spheres']
                print(f"[IK求解器] 静态身体碰撞球: {len(self._body_collision_spheres)} 个 link（作为环境障碍物）")

            # 动态模式：单独为对侧臂 link 生成碰撞球（link 局部坐标系，用于动态 FK 变换）
            self._other_arm_collision_spheres = {}
            if self._dynamic_mode and self._other_arm_links:
                other_arm_config = generate_collision_spheres_yaml(
                    urdf_path=self.robot_config.urdf_path,
                    output_path=None,
                    max_spheres_per_link=6,
                    include_links=list(self._other_arm_links),
                )
                if other_arm_config and other_arm_config.get('collision_spheres'):
                    self._other_arm_collision_spheres = other_arm_config['collision_spheres']
                    n_spheres = sum(len(v) for v in self._other_arm_collision_spheres.values())
                    print(f"[IK求解器] 对侧臂碰撞球: {len(self._other_arm_collision_spheres)} 个 link, "
                          f"{n_spheres} 个球（用于动态过滤）")
            else:
                self._body_collision_spheres = {}

            return config
        except Exception as e:
            print(f"[IK求解器] mesh 碰撞球生成失败: {e}，回退到简单碰撞球")
            import traceback
            traceback.print_exc()
            return None

    def _build_body_world_config(self, existing_world_config=None):
        """
        将非链 link 的碰撞球转换为 cuRobo 环境碰撞障碍物。

        躯干/底盘等不在运动链上的 link 不能用 cuRobo 自碰撞检测，
        但可以作为环境中的球形障碍物。这些球的位置是在 base_link 坐标系中
        通过零位 FK 计算得到的。
        """
        if not self._body_collision_spheres:
            return existing_world_config

        import pinocchio as pin

        # 用 pinocchio 计算零位时各 link 相对于 base_link 的位置
        model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
        data = model.createData()
        q0 = pin.neutral(model)
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)

        # 找 base_link frame
        base_frame_id = None
        for i, f in enumerate(model.frames):
            if f.name == self.robot_config.base_link:
                base_frame_id = i
                break

        if base_frame_id is None:
            print(f"[IK求解器] 无法找到 base_link '{self.robot_config.base_link}' 的 frame，跳过环境碰撞")
            return existing_world_config

        # 构建 link_name -> frame_id 映射
        frame_map = {}
        for i, f in enumerate(model.frames):
            frame_map[f.name] = i

        sphere_obstacles = []
        for link_name, spheres in self._body_collision_spheres.items():
            if link_name not in frame_map:
                continue
            frame_id = frame_map[link_name]
            # link 在 base_link 坐标系中的位姿
            link_in_base = data.oMf[base_frame_id].actInv(data.oMf[frame_id])
            R = link_in_base.rotation
            t = link_in_base.translation

            for sphere in spheres:
                # 将球的中心从 link 坐标系转到 base_link 坐标系
                center_local = np.array(sphere['center'])
                center_base = R @ center_local + t
                sphere_obstacles.append({
                    "name": f"body_{link_name}_{len(sphere_obstacles)}",
                    "pose": [
                        float(center_base[0]),
                        float(center_base[1]),
                        float(center_base[2]),
                        1.0, 0.0, 0.0, 0.0  # quaternion (w,x,y,z)
                    ],
                    "radius": float(sphere['radius']),
                })

        if not sphere_obstacles:
            return existing_world_config

        # 保存排除区域：用于在 IK 求解前过滤掉落入机器人身体内的目标点
        # cuRobo 的碰撞检测只检查臂链碰撞球与环境的碰撞，
        # 不会检查目标点(ee)本身是否在障碍物内部
        centers = np.array([s["pose"][:3] for s in sphere_obstacles])
        radii = np.array([s["radius"] for s in sphere_obstacles])
        self._body_exclusion_centers = centers  # [N, 3] base_link 坐标系
        self._body_exclusion_radii = radii      # [N]
        print(f"[IK求解器] 身体排除区域: {len(sphere_obstacles)} 个球体")

        print(f"[IK求解器] 环境碰撞: {len(sphere_obstacles)} 个碰撞体（来自躯干/底盘 link）")

        # cuRobo 的 PRIMITIVE 碰撞检测器只支持 cuboid (OBB)
        # 将球形障碍物转换为立方体: dims = [2r, 2r, 2r]
        cuboid_dict = {}
        for s in sphere_obstacles:
            r = s["radius"]
            side = 2.0 * r
            cuboid_dict[s["name"]] = {
                "pose": s["pose"],
                "dims": [side, side, side],
            }

        world_dict = {"cuboid": cuboid_dict}
        if existing_world_config is not None:
            if isinstance(existing_world_config, dict):
                for k, v in existing_world_config.items():
                    if k in world_dict:
                        world_dict[k].update(v) if isinstance(v, dict) else None
                    else:
                        world_dict[k] = v
            return world_dict

        return world_dict

    def _check_body_exclusion(self, positions_base: np.ndarray) -> np.ndarray:
        """
        检查目标点是否在机器人身体排除区域内。

        cuRobo 的碰撞检测只检查臂链 link 碰撞球与环境障碍物的碰撞，
        不会检查目标点(ee/tcp)本身是否在障碍物内部。
        因此需要额外的几何过滤来排除落入躯干/底盘内的目标点。

        参数:
            positions_base: 目标点在 base_link 坐标系中的位置 [N, 3]

        返回:
            excluded_mask: [N] 布尔数组，True 表示该点在身体内部（应排除）
        """
        if not hasattr(self, '_body_exclusion_centers') or self._body_exclusion_centers is None:
            return np.zeros(len(positions_base), dtype=bool)

        centers = self._body_exclusion_centers  # [M, 3]
        radii = self._body_exclusion_radii      # [M]

        # 计算每个目标点到每个排除球心的距离
        # positions_base: [N, 3], centers: [M, 3]
        # diff: [N, M, 3]
        diff = positions_base[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # [N, M]

        # 如果目标点到任一球心的距离 < 球半径，则该点在身体内部
        inside = dists < radii[np.newaxis, :]  # [N, M]
        excluded = np.any(inside, axis=1)  # [N]

        return excluded

    def _initialize_dummy_solver(self):
        """初始化模拟求解器用于测试（无需cuRobo）"""
        self.ik_solver = None
        self.n_joints = len(self.robot_config.active_joints)
        print("[模拟求解器] 已初始化用于测试")

    def _initialize_transforms(self):
        """初始化坐标变换，用于将世界坐标转换为 base_link 坐标"""
        try:
            import pinocchio as pin

            # 加载 URDF 模型
            self.pin_model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            self.pin_data = self.pin_model.createData()

            # 查找 base_link 帧 ID
            self.base_link_frame_id = None
            for i, frame in enumerate(self.pin_model.frames):
                if frame.name == self.robot_config.base_link:
                    self.base_link_frame_id = i
                    break

            if self.base_link_frame_id is None:
                raise RuntimeError(f"未找到 base_link '{self.robot_config.base_link}' 的帧")

            # 计算零位姿下 base_link 在世界坐标系中的位置
            q_neutral = np.zeros(self.pin_model.nq)
            for i in range(self.pin_model.njoints):
                joint = self.pin_model.joints[i]
                if joint.nq > 0:
                    idx_q = joint.idx_q
                    q_neutral[idx_q:idx_q+joint.nq] = 0.0

            # 前向运动学
            pin.forwardKinematics(self.pin_model, self.pin_data, q_neutral)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            # 获取 base_link 在世界坐标系中的变换
            base_transform = self.pin_data.oMf[self.base_link_frame_id]

            # 存储 base_link 的位置和旋转
            self._base_transform = {
                'translation': base_transform.translation.copy(),  # [3]
                'rotation': base_transform.rotation.copy()         # [3, 3]
            }

            print(f"[IK求解器] base_link '{self.robot_config.base_link}' 在世界坐标系中的位置:")
            print(f"  位置: {self._base_transform['translation']}")
            print(f"[IK求解器] 将 FK 结果转换为相对于 base_link 的坐标")
            return

        except Exception as e:
            print(f"[IK求解器] Pinocchio 变换初始化失败: {e}")

        # 回退方案：使用 URDF 关节树估算 base_link 在世界坐标系中的变换
        fallback = self._compute_base_transform_from_urdf()
        if fallback is not None:
            self._base_transform = fallback
            print(f"[IK求解器] 使用URDF树估算 base_link '{self.robot_config.base_link}' 在世界坐标系中的位置:")
            print(f"  位置: {self._base_transform['translation']}")
            print(f"[IK求解器] 将 FK 结果转换为相对于 base_link 的坐标")
        else:
            print(f"[IK求解器] 将使用世界坐标系（不进行坐标变换）")

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

    def _compute_base_transform_from_urdf(self) -> Optional[dict]:
        """在未安装 Pinocchio 时，使用 URDF 关节树估算 base_link 变换"""
        try:
            link_names = {l.name for l in self.robot_config.links}
            child_links = {j.child_link for j in self.robot_config.joints}
            roots = list(link_names - child_links)
            if not roots:
                roots = [self.robot_config.base_link]

            world_like = {"world", "map", "odom"}
            roots.sort(key=lambda x: (0 if x.lower() in world_like else 1, x))

            by_parent = {}
            for j in self.robot_config.joints:
                by_parent.setdefault(j.parent_link, []).append(j)

            T = {r: np.eye(4, dtype=np.float64) for r in roots}
            queue = list(roots)
            visited = set(roots)

            while queue:
                parent = queue.pop(0)
                for j in by_parent.get(parent, []):
                    child = j.child_link
                    if child in visited:
                        continue

                    T_origin = np.eye(4, dtype=np.float64)
                    T_origin[:3, 3] = np.array(j.origin_xyz, dtype=np.float64)
                    T_origin[:3, :3] = self._rpy_to_matrix(
                        j.origin_rpy[0], j.origin_rpy[1], j.origin_rpy[2]
                    )

                    # 在零位姿下，关节运动为单位变换
                    T_child = T[parent] @ T_origin
                    T[child] = T_child
                    visited.add(child)
                    queue.append(child)

            if self.robot_config.base_link not in T:
                return None

            base_T = T[self.robot_config.base_link]
            return {
                'translation': base_T[:3, 3].copy(),
                'rotation': base_T[:3, :3].copy()
            }
        except Exception as e:
            print(f"[IK求解器] 无法使用URDF计算 base_link 变换: {e}")
            return None

    def get_base_transform(self) -> Optional[dict]:
        """获取 base_link 在世界坐标系中的变换（translation, rotation）"""
        return self._base_transform

    def solve_single(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """
        为单个目标位姿求解IK（使用多种子）

        参数:
            position: 目标位置 [3]（相对于 base_link）
            orientation: 目标姿态四元数 [4] (w, x, y, z)

        返回:
            包含所有找到解的IKResult
        """
        if self.ik_solver is None:
            return self._solve_single_dummy(position, orientation)

        from curobo.types.math import Pose

        # 将目标位置从 base_link 坐标系转换到世界坐标系（cuRobo 需要）
        world_position = self._transform_to_world_frame(position)

        # 创建目标位姿
        goal_pose = Pose(
            position=torch.tensor(
                world_position.reshape(1, 3),
                dtype=torch.float32,
                device=self.device
            ),
            quaternion=torch.tensor(
                orientation.reshape(1, 4),
                dtype=torch.float32,
                device=self.device
            )
        )

        # 求解IK
        result = self.ik_solver.solve_batch(goal_pose)

        # 提取结果
        success = result.success.cpu().numpy()[0]
        solutions = result.solution.cpu().numpy()[0]  # [num_seeds, n_joints]

        # 获取所有种子的位置和旋转误差
        if hasattr(result, 'position_error') and result.position_error is not None:
            position_errors = result.position_error.cpu().numpy().flatten()
        else:
            position_errors = np.zeros(self.ik_config.num_seeds)

        if hasattr(result, 'rotation_error') and result.rotation_error is not None:
            rotation_errors = result.rotation_error.cpu().numpy().flatten()
        else:
            rotation_errors = np.zeros(self.ik_config.num_seeds)

        # 根据误差阈值创建解的有效性掩码
        solution_mask = (
            (position_errors < self.ik_config.position_threshold) &
            (rotation_errors < self.ik_config.rotation_threshold)
        )

        num_solutions = int(np.sum(solution_mask))

        return IKResult(
            target_position=position,
            target_orientation=orientation,
            success=success or num_solutions > 0,
            num_solutions=num_solutions,
            solutions=solutions,
            solution_mask=solution_mask,
            position_errors=position_errors,
            rotation_errors=rotation_errors
        )

    def _solve_single_dummy(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> IKResult:
        """用于测试的模拟求解器"""
        # 模拟随机结果
        success = np.random.random() > 0.3
        num_solutions = np.random.randint(0, self.ik_config.num_seeds // 2) if success else 0

        solutions = np.random.uniform(
            -np.pi, np.pi,
            (self.ik_config.num_seeds, self.n_joints)
        )

        solution_mask = np.zeros(self.ik_config.num_seeds, dtype=bool)
        solution_mask[:num_solutions] = True

        return IKResult(
            target_position=position,
            target_orientation=orientation,
            success=success,
            num_solutions=num_solutions,
            solutions=solutions.astype(np.float32),
            solution_mask=solution_mask,
            position_errors=np.random.uniform(0, 0.01, self.ik_config.num_seeds).astype(np.float32),
            rotation_errors=np.random.uniform(0, 0.1, self.ik_config.num_seeds).astype(np.float32)
        )

    def solve_batch(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        return_all_solutions: bool = True,
        positions_in_world: bool = False
    ) -> BatchIKResult:
        """
        使用GPU加速批量求解IK

        参数:
            positions: 目标位置 [batch_size, 3]
            orientations: 目标姿态四元数 [batch_size, 4] (w, x, y, z)
            return_all_solutions: 如果为True，返回所有种子的解
            positions_in_world: 如果为True，positions已经是world坐标系，不需要转换

        返回:
            包含所有位姿解的BatchIKResult
        """
        start_time = time.time()
        batch_size = len(positions)

        if self.ik_solver is None:
            return self._solve_batch_dummy(positions, orientations, return_all_solutions)

        from curobo.types.math import Pose

        # cuRobo IK solver 工作在 base_link 坐标系中
        # 如果输入是 world 坐标系，需要逆变换到 base_link；否则直接使用
        if positions_in_world:
            base_positions = self._transform_to_base_frame(positions)
        else:
            base_positions = positions

        # 创建目标位姿
        goal_pose = Pose(
            position=torch.tensor(
                np.ascontiguousarray(base_positions),
                dtype=torch.float32,
                device=self.device
            ),
            quaternion=torch.tensor(
                orientations,
                dtype=torch.float32,
                device=self.device
            )
        )

        # 身体排除过滤：排除落入机器人身体内的目标点
        body_excluded = self._check_body_exclusion(base_positions)
        n_excluded = int(np.sum(body_excluded))

        # 对于不在身体内的点，正常求解 IK
        # 对于在身体内的点，直接标记为不可达
        if n_excluded == batch_size:
            # 所有点都在身体内，直接返回全部不可达
            solve_time = time.time() - start_time
            return BatchIKResult(
                success_mask=np.zeros(batch_size, dtype=bool),
                num_solutions=np.zeros(batch_size, dtype=np.int32),
                best_solutions=np.zeros((batch_size, len(self.robot_config.active_joints)), dtype=np.float32),
                all_solutions=np.zeros((batch_size, self.ik_config.num_seeds, len(self.robot_config.active_joints)), dtype=np.float32) if return_all_solutions else None,
                solution_masks=np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool),
                solve_time=solve_time
            )

        # 批量求解IK
        result = self.ik_solver.solve_batch(goal_pose)

        # 提取结果
        success_mask = result.success.cpu().numpy()
        solutions = result.solution.cpu().numpy()  # [batch_size, num_seeds, n_joints]

        # 将身体内的点标记为不可达
        if n_excluded > 0:
            success_mask[body_excluded] = False

        # 每个位姿的最优解（取第一个成功的种子）
        best_solutions = solutions[:, 0, :]  # 取第一个种子作为最优解

        # 统计每个位姿的解数量
        num_solutions = np.zeros(batch_size, dtype=np.int32)

        # 解的有效性掩码
        solution_masks = np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool)

        # 对于成功的位置，统计所有有效种子
        if hasattr(result, 'position_error') and result.position_error is not None:
            position_errors = result.position_error.cpu().numpy()
            rotation_errors = result.rotation_error.cpu().numpy() if hasattr(result, 'rotation_error') else np.zeros_like(position_errors)

            for i in range(batch_size):
                if body_excluded[i]:
                    continue  # 身体内的点已标记不可达
                valid = (
                    (position_errors[i] < self.ik_config.position_threshold) &
                    (rotation_errors[i] < self.ik_config.rotation_threshold)
                )
                solution_masks[i] = valid
                num_solutions[i] = np.sum(valid)
        else:
            # 使用成功掩码估计解数量
            num_solutions = success_mask.astype(np.int32)
            solution_masks[:, 0] = success_mask

        solve_time = time.time() - start_time

        return BatchIKResult(
            success_mask=success_mask,
            num_solutions=num_solutions,
            best_solutions=best_solutions.astype(np.float32),
            all_solutions=solutions.astype(np.float32) if return_all_solutions else None,
            solution_masks=solution_masks,
            solve_time=solve_time
        )

    def _solve_batch_dummy(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        return_all_solutions: bool
    ) -> BatchIKResult:
        """用于测试的模拟批量求解器"""
        start_time = time.time()
        batch_size = len(positions)

        # 根据与原点的距离模拟结果
        distances = np.linalg.norm(positions, axis=1)
        success_prob = np.clip(1.0 - distances / 2.0, 0.1, 0.9)
        success_mask = np.random.random(batch_size) < success_prob

        num_solutions = np.where(
            success_mask,
            np.random.randint(1, self.ik_config.num_seeds // 2, batch_size),
            0
        ).astype(np.int32)

        best_solutions = np.random.uniform(
            -np.pi, np.pi,
            (batch_size, self.n_joints)
        ).astype(np.float32)

        if return_all_solutions:
            all_solutions = np.random.uniform(
                -np.pi, np.pi,
                (batch_size, self.ik_config.num_seeds, self.n_joints)
            ).astype(np.float32)
        else:
            all_solutions = None

        solution_masks = np.zeros((batch_size, self.ik_config.num_seeds), dtype=bool)
        for i in range(batch_size):
            solution_masks[i, :num_solutions[i]] = True

        solve_time = time.time() - start_time

        return BatchIKResult(
            success_mask=success_mask,
            num_solutions=num_solutions,
            best_solutions=best_solutions,
            all_solutions=all_solutions,
            solution_masks=solution_masks,
            solve_time=solve_time
        )

    def solve_batch_multi_orientation(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        batch_size: int = 1024,
        positions_in_world: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        为多个位置和多个姿态求解IK

        这是可达性分析的主要方法，测试每个位置的多个姿态。

        参数:
            positions: 网格位置 [n_positions, 3]
            orientations: 要测试的姿态 [n_orientations, 4]
            batch_size: GPU批处理大小
            positions_in_world: 如果为 True，positions 已经是 world 坐标系

        返回:
            元组包含:
                - reachable_mask: [n_positions] 任意姿态成功则为True
                - dexterity: [n_positions] 可达姿态数量
                - num_solutions: [n_positions, n_orientations] 每个位姿的解数量
        """
        n_positions = len(positions)
        n_orientations = len(orientations)

        print(f"[IK求解器] 测试 {n_positions} 个位置 × {n_orientations} 个姿态")
        print(f"[IK求解器] 总位姿数: {n_positions * n_orientations}")

        # 预检查：统计有多少位置在身体排除区域内
        if hasattr(self, '_body_exclusion_centers') and self._body_exclusion_centers is not None:
            if positions_in_world:
                check_positions = self._transform_to_base_frame(positions)
            else:
                check_positions = positions
            excluded = self._check_body_exclusion(check_positions)
            n_excluded = int(np.sum(excluded))
            print(f"[IK求解器] 身体排除过滤: {n_excluded} / {n_positions} 个位置在机器人身体内（将被排除）")

        # 创建所有位置-姿态组合
        # positions: [n_positions, 3] -> [n_positions, n_orientations, 3]
        # orientations: [n_orientations, 4] -> [n_positions, n_orientations, 4]
        pos_expanded = np.tile(positions[:, np.newaxis, :], (1, n_orientations, 1))
        ori_expanded = np.tile(orientations[np.newaxis, :, :], (n_positions, 1, 1))

        # 展平用于批处理
        all_positions = pos_expanded.reshape(-1, 3)
        all_orientations = ori_expanded.reshape(-1, 4)
        total_poses = len(all_positions)

        # 分批处理
        all_success = []
        all_num_solutions = []

        num_batches = (total_poses + batch_size - 1) // batch_size
        print(f"[IK求解器] 处理 {num_batches} 个批次...")

        for i in range(0, total_poses, batch_size):
            end_idx = min(i + batch_size, total_poses)
            batch_positions = all_positions[i:end_idx]
            batch_orientations = all_orientations[i:end_idx]

            result = self.solve_batch(
                batch_positions,
                batch_orientations,
                return_all_solutions=False,
                positions_in_world=positions_in_world
            )

            all_success.append(result.success_mask)
            all_num_solutions.append(result.num_solutions)

            # 显示进度
            progress = (i + batch_size) / total_poses * 100
            print(f"\r[IK求解器] 进度: {min(progress, 100):.1f}%", end='', flush=True)

        print()  # 换行

        # 合并结果
        success = np.concatenate(all_success)
        num_solutions = np.concatenate(all_num_solutions)

        # 重塑为 [n_positions, n_orientations]
        success_grid = success.reshape(n_positions, n_orientations)
        num_solutions_grid = num_solutions.reshape(n_positions, n_orientations)

        # 计算每个位置的指标
        reachable_mask = np.any(success_grid, axis=1)
        dexterity = np.sum(success_grid, axis=1)

        print(f"[IK求解器] 可达位置: {np.sum(reachable_mask)} / {n_positions}")
        print(f"[IK求解器] 最大灵活度: {np.max(dexterity)}")

        return reachable_mask, dexterity, num_solutions_grid

    def _transform_to_base_frame(self, world_positions: np.ndarray) -> np.ndarray:
        """
        将世界坐标系中的位置转换为 base_link 坐标系

        参数:
            world_positions: 世界坐标系中的位置 [batch_size, 3] 或 [3]

        返回:
            base_positions: base_link 坐标系中的位置 [batch_size, 3] 或 [3]
        """
        if self._base_transform is None:
            # 没有变换信息，直接返回原始位置
            return world_positions

        # 提取 base_link 的位置和旋转
        base_translation = self._base_transform['translation']
        base_rotation = self._base_transform['rotation']

        # 将位置从世界坐标系转换到 base_link 坐标系
        # p_base = R^T * (p_world - t_base)
        if world_positions.ndim == 1:
            # 单个位置 [3]
            relative_pos = world_positions - base_translation
            base_positions = base_rotation.T @ relative_pos
        else:
            # 批量位置 [batch_size, 3]
            relative_pos = world_positions - base_translation[np.newaxis, :]
            # 使用 einsum 保持数组连续性
            base_positions = np.einsum('ji,nj->ni', base_rotation, relative_pos)
            # 确保数组在内存中是连续的
            base_positions = np.ascontiguousarray(base_positions)

        return base_positions

    def _transform_to_base_frame(self, world_positions: np.ndarray) -> np.ndarray:
        """
        将世界坐标系中的位置转换为 base_link 坐标系

        参数:
            world_positions: 世界坐标系中的位置 [batch_size, 3] 或 [3]

        返回:
            base_positions: base_link 坐标系中的位置 [batch_size, 3] 或 [3]
        """
        if self._base_transform is None:
            return world_positions

        base_translation = self._base_transform['translation']
        base_rotation = self._base_transform['rotation']

        # p_base = R^T * (p_world - t_base)
        R_inv = base_rotation.T
        if world_positions.ndim == 1:
            base_positions = R_inv @ (world_positions - base_translation)
        else:
            shifted = world_positions - base_translation[np.newaxis, :]
            base_positions = np.einsum('ij,nj->ni', R_inv, shifted)
            base_positions = np.ascontiguousarray(base_positions)

        return base_positions

    def _transform_to_world_frame(self, base_positions: np.ndarray) -> np.ndarray:
        """
        将 base_link 坐标系中的位置转换为世界坐标系

        参数:
            base_positions: base_link 坐标系中的位置 [batch_size, 3] 或 [3]

        返回:
            world_positions: 世界坐标系中的位置 [batch_size, 3] 或 [3]
        """
        if self._base_transform is None:
            # 没有变换信息，直接返回原始位置
            return base_positions

        # 提取 base_link 的位置和旋转
        base_translation = self._base_transform['translation']
        base_rotation = self._base_transform['rotation']

        # 将位置从 base_link 坐标系转换到世界坐标系
        # p_world = R * p_base + t_base
        if base_positions.ndim == 1:
            # 单个位置 [3]
            world_positions = base_rotation @ base_positions + base_translation
        else:
            # 批量位置 [batch_size, 3]
            # 使用 einsum 保持数组连续性，避免 cuRobo 的张量视图问题
            world_positions = np.einsum('ij,nj->ni', base_rotation, base_positions) + base_translation[np.newaxis, :]
            # 确保数组在内存中是连续的
            world_positions = np.ascontiguousarray(world_positions)

        return world_positions

    def get_forward_kinematics(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算给定关节位置的正运动学

        参数:
            joint_positions: 关节位置 [batch_size, n_joints]

        返回:
            元组包含:
                - positions: 末端执行器位置（相对于 base_link）[batch_size, 3]
                - orientations: 末端执行器姿态 [batch_size, 4]
        """
        if self.ik_solver is None:
            # 模拟FK
            batch_size = len(joint_positions)
            positions = np.random.uniform(-1, 1, (batch_size, 3)).astype(np.float32)
            orientations = np.zeros((batch_size, 4), dtype=np.float32)
            orientations[:, 0] = 1.0  # w = 1 表示单位四元数
            return positions, orientations

        # 使用cuRobo运动学
        q = torch.tensor(
            joint_positions,
            dtype=torch.float32,
            device=self.device
        )

        state = self.ik_solver.fk(q)

        # cuRobo FK 输出已经在 base_link 坐标系中
        positions = state.ee_position.cpu().numpy()
        orientations = state.ee_quaternion.cpu().numpy()  # (w, x, y, z)

        return positions, orientations

    def sample_workspace(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用随机FK采样机器人的工作空间

        参数:
            n_samples: 要采样的随机关节配置数量

        返回:
            元组包含:
                - positions: 采样的末端执行器位置（相对于 base_link）[n_samples, 3]
                - joint_configs: 对应的关节配置 [n_samples, n_joints]
        """
        # 在限位范围内生成随机关节配置
        active_joints = self.robot_config.active_joints
        joint_configs = np.zeros((n_samples, self.n_joints), dtype=np.float32)

        for i, joint in enumerate(active_joints):
            joint_configs[:, i] = np.random.uniform(
                joint.lower_limit,
                joint.upper_limit,
                n_samples
            )

        # 计算FK
        positions, _ = self.get_forward_kinematics(joint_configs)

        return positions, joint_configs

    def estimate_workspace_bounds(
        self,
        n_samples: int = 10000,
        padding: float = 0.5
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        使用FK采样估计工作空间边界（相对于 base_link）

        该方法通过随机采样关节配置并计算 FK，估计机械臂的工作空间边界。
        返回的边界是相对于 base_link 坐标系的，即 base_link 位于原点 (0, 0, 0)。

        重要：为了显示完整的可达空间边界（球形/环形），我们需要将采样范围
        扩展到 FK 可达范围之外。默认 padding=0.5 (50%) 可以较好地显示边界。

        参数:
            n_samples: 用于估计的样本数量
            padding: 添加到边界的填充比例（默认 0.5 = 50%，确保能看到可达边界）

        返回:
            (x_range, y_range, z_range) 元组，每个为 (min, max) 元组
            边界相对于 base_link 坐标系
        """
        print(f"[IK求解器] 使用 {n_samples} 个FK样本估计工作空间边界...")
        print(f"[IK求解器] 边界将相对于 base_link '{self.robot_config.base_link}' 计算")

        positions, _ = self.sample_workspace(n_samples)

        x_min, y_min, z_min = positions.min(axis=0)
        x_max, y_max, z_max = positions.max(axis=0)

        # ============================
        # 边界策略说明
        # FK 采样得到的是实际可达位置，为了显示完整的可达空间边界（类球形），
        # 需要将采样范围扩展到可达范围之外。
        #
        # - x/y：以 base_link 为中心对称
        # - z：同样对称，覆盖上下空间
        # - padding=0.5：扩展 50%，确保能看到可达边界
        # ============================
        x_abs = float(max(abs(x_min), abs(x_max)))
        y_abs = float(max(abs(y_min), abs(y_max)))
        z_abs = float(max(abs(z_min), abs(z_max)))

        # 添加填充（扩展范围以显示可达边界）
        x_half = x_abs * (1.0 + padding)
        y_half = y_abs * (1.0 + padding)
        z_half = z_abs * (1.0 + padding)

        # 给一个最小范围，避免采样不足导致范围过小
        min_half = 0.5  # meters
        x_half = max(x_half, min_half)
        y_half = max(y_half, min_half)
        z_half = max(z_half, min_half)

        x_range = (-x_half, x_half)
        y_range = (-y_half, y_half)
        z_range = (-z_half, z_half)

        print(f"[IK求解器] FK 可达范围:")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
        print(f"[IK求解器] 扩展后采样边界 (padding={padding*100:.0f}%):")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range

    def estimate_workspace_bounds_world(
        self,
        n_samples: int = 10000,
        padding: float = 0.5
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        在 world 坐标系下估计工作空间边界

        与 estimate_workspace_bounds 不同，此方法直接在 world 坐标系下计算边界，
        考虑了 base_link 相对于 world 的旋转。这对于双臂机器人特别重要，
        因为每条臂的 base_link 可能有不同的旋转角度。

        参数:
            n_samples: 用于估计的样本数量
            padding: 添加到边界的填充比例（默认 0.5 = 50%）

        返回:
            (x_range, y_range, z_range) 元组，每个为 (min, max) 元组
            边界在 world 坐标系下
        """
        print(f"[IK求解器] 使用 {n_samples} 个FK样本估计工作空间边界（world 坐标系）...")
        print(f"[IK求解器] base_link: '{self.robot_config.base_link}'")

        if self.ik_solver is None:
            print("[IK求解器] 求解器未初始化，回退到 base_link 坐标系")
            return self.estimate_workspace_bounds(n_samples, padding)

        # 在关节限位范围内生成随机关节配置
        active_joints = self.robot_config.active_joints
        joint_configs = np.zeros((n_samples, self.n_joints), dtype=np.float32)

        for i, joint in enumerate(active_joints):
            joint_configs[:, i] = np.random.uniform(
                joint.lower_limit,
                joint.upper_limit,
                n_samples
            )

        # 计算 FK（cuRobo FK 输出在 base_link 坐标系中）
        q = torch.tensor(joint_configs, dtype=torch.float32, device=self.device)
        state = self.ik_solver.fk(q)
        base_positions = state.ee_position.cpu().numpy()  # base_link 坐标系

        # 转换到 world 坐标系
        world_positions = self._transform_to_world_frame(base_positions)

        # 计算 world 坐标系下的边界
        x_min, y_min, z_min = world_positions.min(axis=0)
        x_max, y_max, z_max = world_positions.max(axis=0)

        # 计算范围并添加 padding
        x_range_val = x_max - x_min
        y_range_val = y_max - y_min
        z_range_val = z_max - z_min

        x_pad = x_range_val * padding / 2
        y_pad = y_range_val * padding / 2
        z_pad = z_range_val * padding / 2

        # 最小 padding
        min_pad = 0.25  # meters
        x_pad = max(x_pad, min_pad)
        y_pad = max(y_pad, min_pad)
        z_pad = max(z_pad, min_pad)

        x_range = (float(x_min - x_pad), float(x_max + x_pad))
        y_range = (float(y_min - y_pad), float(y_max + y_pad))
        z_range = (float(z_min - z_pad), float(z_max + z_pad))

        print(f"[IK求解器] FK 可达范围 (world 坐标系):")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
        print(f"[IK求解器] 扩展后采样边界 (padding={padding*100:.0f}%):")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")

        return x_range, y_range, z_range
