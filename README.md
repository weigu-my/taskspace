# ARM Reachability Analysis Framework

通用机械臂可达性分析框架，基于 cuRobo 实现 GPU 加速的 IK 求解。

## 功能特性

- **通用 URDF 支持**：支持任意 URDF 格式的机械臂
- **GPU 加速 IK 求解**：使用 cuRobo 实现高效的批量 IK 计算
- **多种子 IK 求解**：通过多种子策略找到冗余机械臂的多个 IK 解
- **体素空间采样**：基于体素网格的工作空间采样
- **灵活度分析 (Dexterity)**：计算每个位置可达的姿态数量
- **可操作度分析 (Manipulability)**：计算 Yoshikawa 可操作度指标
- **交互式可视化**：
  - 点的 **大小** 表示可操作度 (manipulability)
  - 点的 **颜色** 表示灵活度 (dexterity)
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

# 可选: Pinocchio (用于精确的 Jacobian 计算)
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
├── __init__.py           # 包初始化
├── config.py             # 配置类定义
├── urdf_parser.py        # URDF 解析和 cuRobo 配置生成
├── ik_solver.py          # 多种子 GPU 加速 IK 求解器
├── reachability.py       # 可达性分析核心逻辑
├── manipulability.py     # Yoshikawa 可操作度计算
├── visualization.py      # 可视化模块
└── utils.py              # 工具函数

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
| fixed | 单一固定姿态 |
| multi_fixed | 多个预定义姿态 (默认6个方向) |
| spherical | 球面均匀采样 |
| random | 随机姿态 |

## 输出文件

分析完成后生成以下文件：

| 文件 | 说明 |
|------|------|
| `reachability_grid_points.npy` | 所有体素中心点坐标 |
| `reachability_reachable_mask.npy` | 可达性布尔掩码 |
| `reachability_reachable_points.npy` | 可达点坐标 |
| `reachability_dexterity.npy` | 每个点的灵活度 |
| `reachability_manipulability.npy` | 每个点的可操作度 |
| `reachability_data.csv` | CSV 格式的完整数据 |
| `reachability_stats.json` | 统计信息 |
| `reachability_visualization.html` | 交互式可视化 |
| `reachability_summary.png` | 汇总图 |

## 指标说明

### 灵活度 (Dexterity)

灵活度表示在某个位置有多少种末端姿态可以到达。值越高表示该位置的姿态选择越多。

- 取值范围: 0 到 姿态采样数
- 可视化: 点的颜色 (蓝色→红色 表示 低→高)

### 可操作度 (Manipulability)

使用 Yoshikawa 可操作度指标：

```
w = sqrt(det(J * J^T))
```

其中 J 是雅可比矩阵。该指标衡量机器人在该配置下各方向运动的能力，值越大表示离奇异点越远。

- 取值范围: 0 到 1 (归一化后)
- 可视化: 点的大小 (小→大 表示 低→高)

### IK 解的数量

对于冗余机械臂 (DOF > 6)，同一个末端位姿可能有多个 IK 解。通过多种子策略可以找到更多解，用于评估冗余度利用程度。

## 示例

运行示例脚本：

```bash
# 运行所有示例
python example.py --urdf robot.urdf

# 运行特定示例
python example.py --urdf robot.urdf --example 1  # 基本分析
python example.py --urdf robot.urdf --example 2  # 自定义配置
python example.py --urdf robot.urdf --example 3  # 球面姿态采样
python example.py --urdf robot.urdf --example 4  # 可视化
python example.py --urdf robot.urdf --example 5  # URDF 解析
python example.py --urdf robot.urdf --example 6  # IK 求解器
```

## 许可证

MIT License
