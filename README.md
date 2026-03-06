# 机械臂可达性分析框架

通用机械臂可达性分析框架，基于 cuRobo 实现 GPU 加速的逆运动学（IK）求解，支持灵活度和可操作度分析。

## 功能特性

- **通用 URDF 支持**：支持任意 URDF 格式的机械臂，自动检测臂链（左/右臂）
- **GPU 加速 IK 求解**：使用 cuRobo 实现高效的批量 IK 计算
- **多种子 IK 求解**：通过多种子策略找到冗余机械臂的多个 IK 解
- **碰撞检测**：
  - **自碰撞检测**：基于 URDF mesh 自动生成碰撞球，cuRobo 内置自碰撞检测
  - **身体碰撞排除**：自动从 URDF mesh 生成躯干/底盘碰撞球，过滤掉落入机器人身体内的目标点
  - 碰撞球通过 K-means 聚类拟合 STL mesh 表面生成
- **体素空间采样**：基于体素网格的工作空间采样，支持自动边界检测
- **灵活度分析 (Dexterity)**：计算每个位置可达的姿态数量
- **可操作度分析 (Manipulability)**：计算 Yoshikawa 可操作度指标
- **RViz2 点云可视化**：
  - 点的 **颜色** 表示灵活度（dexterity 色阶）
  - 点的 **透明度** 表示可操作度（alpha）
  - 支持双臂同时可视化
- **交互式可视化（Plotly/Matplotlib）**：
  - 点的 **大小** 表示可操作度
  - 点的 **颜色** 表示灵活度
  - 机器人 mesh 模型渲染

## 安装依赖

```bash
# 核心依赖
pip install torch numpy pandas pyyaml

# cuRobo (GPU 加速 IK) - 作为 git submodule 包含在 curobo/ 目录
# 安装方式见 curobo/README.md

# URDF 解析
pip install yourdfpy urdf-parser-py

# 碰撞球生成
pip install trimesh

# 运动学
conda install -c conda-forge pinocchio

# 可视化
pip install plotly matplotlib

# ROS 2（RViz 可视化需要）
# 需要已安装 ROS 2 Humble 或更高版本
```

## 快速开始

### 命令行使用

```bash
# 基本用法：分析单臂
python main.py --urdf /path/to/robot.urdf

# 双臂分析 + RViz 可视化（推荐用法）
./launch_reachability_rviz.sh robot_model/urdf/wheel_robot.urdf \
    --arm both --joint-state-gui

# 指定参数
python main.py --urdf robot.urdf \
    --resolution 0.05 \
    --num-seeds 32 \
    --orientation-mode multi_fixed \
    --points-frame world \
    --output ./results

# 启动 RViz2 点云可视化
python main.py --urdf robot.urdf \
    --arm both \
    --rviz \
    --rviz-frame world \
    --rviz-color-mode dexterity
```

### Python API

```python
from arm_reachability import ReachabilityAnalyzer

# 快速分析
analyzer = ReachabilityAnalyzer.from_urdf(
    urdf_path="robot.urdf",
    resolution=0.05,
    num_seeds=32
)
result = analyzer.analyze()
analyzer.save_results(result)
```

### RViz2 可视化

```bash
# 一键脚本：分析 + RViz 可视化
./launch_reachability_rviz.sh robot_model/urdf/wheel_robot.urdf \
  --arm both \
  --points-frame world \
  --resolution 0.05 \
  --rviz-color-mode dexterity

# 跳过分析，直接发布已有结果
./launch_reachability_rviz.sh robot_model/urdf/wheel_robot.urdf \
  --arm both \
  --skip-analysis \
  --rviz-color-mode dexterity
```

`--rviz-color-mode` 可选值：`dexterity`（按灵活度上色，默认）、`arm`（臂固定颜色）、`tint`（灵活度 + 臂偏色）

## 项目结构

```
arm_reachability/               # 核心模块
├── __init__.py                 # 包初始化和公共接口导出
├── config.py                   # 配置类定义（体素、IK、姿态、多臂配置）
├── urdf_parser.py              # URDF 解析、臂链检测、cuRobo 配置生成
├── ik_solver.py                # 多种子 GPU 加速 IK 求解器 + 身体碰撞排除
├── collision_spheres_generator.py  # 从 URDF mesh(STL) 生成碰撞球
├── reachability.py             # 可达性分析核心逻辑（单臂/多臂）
├── manipulability.py           # Yoshikawa 可操作度计算（Pinocchio 雅可比）
├── visualization.py            # 可视化模块（Plotly/Matplotlib）
├── ros2_visualization.py       # ROS 2 RViz 点云发布
├── rviz_publisher_node.py      # RViz 发布节点（独立运行）
├── utils.py                    # 工具函数（四元数、颜色映射等）
├── dynamic_reachability.py     # 动态可达性分析（考虑指定姿态下的碰撞）
├── examples/                   # 示例脚本
│   └── dynamic_reachability_example.py
└── launch/                     # ROS 2 launch 文件
    └── dynamic_reachability_launch.py

main.py                         # 命令行主入口
launch_reachability_rviz.sh     # 一键分析 + RViz 启动脚本
robot_model/urdf/               # 机器人 URDF 模型
    ├── wheel_robot.urdf
    └── wheel_robot/            # mesh 文件目录
curobo/                         # cuRobo GPU IK 引擎（git submodule）
```

## 核心流程

```
URDF 文件
    │
    ▼
URDFParser.detect_arm_chains()     ← 自动检测左/右臂链
    │
    ▼
collision_spheres_generator        ← 从 STL mesh 生成碰撞球
    │                                 ├── 臂链 link → cuRobo 自碰撞球
    │                                 └── 身体 link → 排除区域球体
    ▼
MultiSeedIKSolver                  ← cuRobo GPU IK 求解
    │  ├── 身体排除过滤：目标点在身体碰撞球内 → 直接排除
    │  ├── cuRobo 自碰撞检测：臂链 link 间碰撞
    │  └── cuRobo 环境碰撞：臂链 link 与身体 cuboid 碰撞
    ▼
ReachabilityAnalyzer.analyze()     ← 体素网格 × 多姿态 IK 求解
    │
    ▼
结果输出
    ├── .npy / .csv / .json        ← 数据文件
    ├── Plotly HTML                 ← 交互式 3D 可视化
    └── RViz PointCloud2           ← ROS 2 点云发布
```

### 碰撞检测机制

碰撞检测分三层：

1. **身体排除过滤**（`_check_body_exclusion`）：从 URDF mesh 为非运动链 link（躯干、底盘、轮子等）生成碰撞球，检查 IK 目标点是否落入这些球体内部。如果目标点在球内，直接标记为不可达。这是必要的，因为 cuRobo 只检查臂链碰撞球与障碍物的碰撞，不检查目标点本身。

2. **cuRobo 环境碰撞**（PRIMITIVE checker）：非运动链 link 的碰撞球转换为 cuboid（cuRobo PRIMITIVE 只支持 OBB），作为环境障碍物。IK 优化时臂链碰撞球会避开这些障碍物。

3. **cuRobo 自碰撞**：臂链 link 之间的碰撞检测，使用 `self_collision_ignore` 排除相邻 link 对。

## 配置参数

### 体素配置 (VoxelConfig)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| x_range | X轴范围 [min, max] (米) | (-1.0, 1.0) |
| y_range | Y轴范围 [min, max] (米) | (-1.0, 1.0) |
| z_range | Z轴范围 [min, max] (米) | (0.0, 1.5) |
| resolution | 体素分辨率 (米) | 0.05 |
| auto_detect | 自动检测工作空间边界 | True |

### IK 配置 (IKConfig)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| num_seeds | IK 求解种子数 | 32 |
| position_threshold | 位置误差阈值 (米) | 0.005 |
| rotation_threshold | 旋转误差阈值 (弧度) | 0.05 |
| batch_size | GPU 批处理大小 | 1024 |
| self_collision_check | 启用自碰撞检测 | True |

### 姿态配置 (OrientationConfig)

| 模式 | 说明 |
|------|------|
| fixed | 单一固定姿态（默认朝下） |
| multi_fixed | 多个预定义姿态（6个主轴方向） |
| spherical | 球面均匀采样（斐波那契格点） |
| random | 随机姿态（均匀分布于 SO(3)） |

## 输出文件

| 文件 | 说明 |
|------|------|
| `reachability_grid_points.npy` | 所有体素中心点坐标 [N, 3] |
| `reachability_reachable_mask.npy` | 可达性布尔掩码 [N] |
| `reachability_reachable_points.npy` | 可达点坐标 [M, 3] |
| `reachability_dexterity.npy` | 每个点的灵活度 [N] |
| `reachability_manipulability.npy` | 每个点的可操作度 [N] |
| `reachability_best_solutions.npy` | 每个点的最优 IK 解 [N, n_joints] |
| `reachability_data.csv` | CSV 格式的完整数据 |
| `reachability_stats.json` | 统计信息（可达率、平均值等） |

## 指标说明

### 灵活度 (Dexterity)

每个位置可达的姿态数量。值越高表示该位置的姿态选择越多。

- **可视化**: 颜色（蓝 → 青 → 绿 → 黄 → 红 = 低 → 高）

### 可操作度 (Manipulability)

Yoshikawa 可操作度指标 `w = sqrt(det(J * J^T))`，衡量离奇异点的距离。

- **可视化**: Plotly 中用点大小表示，RViz 中用透明度表示

## 当前已知的机器人配置

- **wheel_robot**：双臂轮式机器人
  - 左臂：`AR5_5_07L_base → AR5_5_07L_tcp`（7 DOF）
  - 右臂：`AR5_5_07R_base → AR5_5_07R_tcp`（7 DOF）
  - 自动检测：`URDFParser.detect_arm_chains()` 或 `parse_arm('left'/'right')`

## 常见问题

### Q: 如何提高分析速度？

增大 `resolution`（如 0.1）、减小 `num_seeds`（如 16）、增大 `batch_size`、使用 `fixed` 姿态模式。

### Q: 可达空间中有点穿透躯干？

确保 `self_collision_check=True`（默认开启）。身体排除过滤会自动从 URDF mesh 生成碰撞球，过滤掉落入身体内的目标点。如果仍有穿透，可能是 mesh 碰撞球覆盖不够密——可增加 `max_spheres_per_link` 参数。

### Q: 可操作度显示不正确？

安装 Pinocchio 可获得精确的解析雅可比。确保 URDF 中关节限位正确。

## 许可证

MIT License
