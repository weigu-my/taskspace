# 机械臂可达性分析框架

通用机械臂可达性分析框架，基于 cuRobo 实现 GPU 加速的逆运动学（IK）求解，支持灵活度和可操作度分析。

## 功能特性

- **通用 URDF 支持**：支持任意 URDF 格式的机械臂，自动检测基座链接和末端执行器
- **GPU 加速 IK 求解**：使用 cuRobo 实现高效的批量 IK 计算
- **多种子 IK 求解**：通过多种子策略找到冗余机械臂的多个 IK 解
- **体素空间采样**：基于体素网格的工作空间采样，支持自动边界检测
- **灵活度分析 (Dexterity)**：计算每个位置可达的姿态数量
- **可操作度分析 (Manipulability)**：计算 Yoshikawa 可操作度指标
- **交互式可视化**：
  - 点的 **大小** 表示可操作度
  - 点的 **颜色** 表示灵活度
  - 机器人 mesh 模型渲染

## 安装依赖

```bash
# 核心依赖
pip install torch numpy pandas pyyaml

# cuRobo (GPU 加速 IK)
pip install curobo

# URDF 解析
pip install yourdfpy urdf-parser-py

# 可视化
pip install plotly trimesh matplotlib

# 可选: Pinocchio (用于精确的雅可比计算)
conda install -c conda-forge pinocchio
```

## 快速开始

### 命令行使用

```bash
# 基本用法
python main.py --urdf /path/to/robot.urdf

# 指定参数
python main.py --urdf robot.urdf \
    --resolution 0.05 \
    --num-seeds 32 \
    --orientation-mode multi_fixed \
    --output ./results

# 指定末端执行器和基座链接
python main.py --urdf robot.urdf \
    --ee-link tool0 \
    --base-link base_link

# 使用配置文件
python main.py --urdf robot.urdf --config config.yaml
```

### Python API 使用

```python
from arm_reachability import ReachabilityAnalyzer

# 方法1: 快速分析
analyzer = ReachabilityAnalyzer.from_urdf(
    urdf_path="robot.urdf",
    resolution=0.05,
    num_seeds=32
)
result = analyzer.analyze()
analyzer.save_results(result)

# 方法2: 自定义配置
from arm_reachability import ReachabilityConfig, VoxelConfig, IKConfig

config = ReachabilityConfig(
    urdf_path="robot.urdf",
    output_dir="./output"
)
config.voxel.resolution = 0.04
config.ik.num_seeds = 64

analyzer = ReachabilityAnalyzer(config)
result = analyzer.analyze()
```

### 可视化

```python
from arm_reachability import ReachabilityVisualizer

visualizer = ReachabilityVisualizer(robot_config, result)

# 交互式 HTML 可视化
visualizer.visualize_plotly(
    show_robot=True,
    save_html="visualization.html"
)

# 生成汇总图
visualizer.create_summary_plot(save_path="summary.png")
```

## 项目结构

```
arm_reachability/
├── __init__.py           # 包初始化和公共接口导出
├── config.py             # 配置类定义（体素、IK、姿态配置）
├── urdf_parser.py        # URDF 解析和 cuRobo 配置生成
├── ik_solver.py          # 多种子 GPU 加速 IK 求解器
├── reachability.py       # 可达性分析核心逻辑
├── manipulability.py     # Yoshikawa 可操作度计算
├── visualization.py      # 可视化模块（Plotly/Matplotlib）
└── utils.py              # 工具函数（四元数、颜色映射等）

main.py                   # 命令行入口
example.py                # 使用示例
```

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
| collision_check | 启用碰撞检测 | True |
| self_collision_check | 启用自碰撞检测 | True |

### 姿态配置 (OrientationConfig)

| 模式 | 说明 |
|------|------|
| fixed | 单一固定姿态（默认朝下） |
| multi_fixed | 多个预定义姿态（6个主轴方向） |
| spherical | 球面均匀采样（斐波那契格点） |
| random | 随机姿态（均匀分布于 SO(3)） |

## 输出文件

分析完成后生成以下文件：

| 文件 | 说明 |
|------|------|
| `reachability_grid_points.npy` | 所有体素中心点坐标 [N, 3] |
| `reachability_reachable_mask.npy` | 可达性布尔掩码 [N] |
| `reachability_reachable_points.npy` | 可达点坐标 [M, 3] |
| `reachability_dexterity.npy` | 每个点的灵活度 [N] |
| `reachability_manipulability.npy` | 每个点的可操作度 [N] |
| `reachability_data.csv` | CSV 格式的完整数据 |
| `reachability_stats.json` | 统计信息（可达率、平均值等） |
| `reachability_visualization.html` | 交互式 3D 可视化 |
| `reachability_summary.png` | 多视图汇总图 |

## 指标说明

### 灵活度 (Dexterity)

灵活度表示在某个位置有多少种末端姿态可以到达。值越高表示该位置的姿态选择越多，机器人在该处的灵活性越好。

- **取值范围**: 0 到 姿态采样数
- **可视化**: 点的颜色（蓝色 → 青色 → 绿色 → 黄色 → 红色，表示低 → 高）

### 可操作度 (Manipulability)

使用 Yoshikawa 可操作度指标：

```
w = sqrt(det(J * J^T))
```

其中 J 是雅可比矩阵。该指标衡量机器人在该配置下各方向运动的能力：
- 值越大表示离奇异点越远
- 在奇异点处值为 0
- 反映末端执行器在各方向上的运动能力均衡性

- **取值范围**: 0 到 1（归一化后）
- **可视化**: 点的大小（小 → 大 表示 低 → 高）

### IK 解的数量

对于冗余机械臂（自由度 > 6），同一个末端位姿可能有多个 IK 解。通过多种子策略可以找到更多解，用于评估冗余度利用程度。

## 核心模块说明

### URDF 解析器 (urdf_parser.py)

- 自动解析 URDF 文件结构
- 自动检测基座链接（优先识别 `base_link`、`base` 等命名）
- 自动检测末端执行器（优先识别 `tool`、`ee`、`gripper` 等命名）
- 生成 cuRobo 兼容的配置文件

### IK 求解器 (ik_solver.py)

- 基于 cuRobo 的 GPU 加速批量 IK 求解
- 支持多种子策略，提高解的覆盖率
- 动态检测实际关节数，适配不同机械臂
- 自动处理碰撞检测和自碰撞检测

### 可操作度计算器 (manipulability.py)

- 支持 Pinocchio 计算解析雅可比（精确）
- 回退到数值雅可比近似（兼容性）
- 计算条件数用于各向同性分析
- GPU 加速的批量计算支持

### 可视化模块 (visualization.py)

- Plotly 交互式 3D 可视化
- Matplotlib 静态图像生成
- 机器人网格模型加载和渲染
- 多视图汇总图（3D、俯视、正视、侧视）

## 使用示例

### 示例 1: 基本分析

```python
from arm_reachability import ReachabilityAnalyzer

# 创建分析器并执行分析
analyzer = ReachabilityAnalyzer.from_urdf("robot.urdf")
result = analyzer.analyze()

# 打印统计信息
print(f"可达点数: {result.reachable_count}/{result.total_points}")
print(f"可达率: {result.reachability_ratio * 100:.1f}%")
print(f"平均灵活度: {result.dexterity[result.reachable_mask].mean():.2f}")
```

### 示例 2: 自定义工作空间

```python
from arm_reachability import ReachabilityConfig

config = ReachabilityConfig(urdf_path="robot.urdf")

# 设置自定义工作空间范围
config.voxel.x_range = (-0.5, 0.5)
config.voxel.y_range = (-0.5, 0.5)
config.voxel.z_range = (0.2, 1.0)
config.voxel.resolution = 0.03  # 更细的分辨率

analyzer = ReachabilityAnalyzer(config)
result = analyzer.analyze()
```

### 示例 3: 球面姿态采样

```python
from arm_reachability import ReachabilityConfig

config = ReachabilityConfig(urdf_path="robot.urdf")

# 使用球面姿态采样
config.orientation.mode = "spherical"
config.orientation.num_samples = 50  # 50 个均匀分布的姿态

analyzer = ReachabilityAnalyzer(config)
result = analyzer.analyze()
```

## 常见问题

### Q: 为什么某些网格无法加载？

A: 检查 URDF 中的网格路径是否正确。如果使用 `package://` 格式，确保：
1. 包目录结构正确
2. meshes 文件夹与 URDF 在正确的相对位置

### Q: 如何提高分析速度？

A: 可以尝试：
1. 增大 `resolution`（如 0.1）减少采样点
2. 减小 IK `num_seeds`（如 16）
3. 增大 `batch_size`（需要更多 GPU 显存）
4. 使用 `fixed` 姿态模式而非 `multi_fixed`

### Q: 可操作度显示不正确怎么办？

A: 可操作度计算依赖雅可比矩阵：
1. 安装 Pinocchio 可获得精确的解析雅可比
2. 确保 URDF 中关节限位正确
3. 检查末端执行器链接是否正确设置

## 许可证

MIT License
