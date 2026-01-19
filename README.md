# taskspace

## 机械臂可达性分析通用框架（cuRobo + GPU）

本项目提供一套从 **XRDF → GPU IK → 可达性分析 → 可视化** 的完整流程，满足以下需求：

1. 直接使用 XRDF 进行可达性分析。
2. 使用体素划分 + GPU IK 批量求解进行可达性分析。
3. 支持多 seeds IK，统计每个体素点的解的数量。
4. 计算 dexterity 与 Yoshikawa manipulability。
5. 可视化机器人模型 + 可达点云（点大小=manipulability，颜色=dexterity）。

---

## 1. 可达性分析（GPU IK + 多种指标）

```bash
python reachability_analysis.py \
  --xrdf /path/to/robot.xrdf \
  --voxel-size 0.05 \
  --num-orientations 16 \
  --num-seeds 64 \
  --batch-size 1024 \
  --output-dir ./reachability_output
```

输出文件（示例）：

- `reachability_output/robot_arm_reachable_metrics.csv`
- `reachability_output/robot_arm_reachable_metrics.npy`
- `reachability_output/robot_arm_stats.json`

CSV 字段：

```
x,y,z,dexterity,solution_count,manipulability
```

---

## 2. 可视化（URDF + 点云）

```bash
python visualize_with_robot.py \
  --urdf /path/to/robot.urdf \
  --metrics ./reachability_output/robot_arm_reachable_metrics.csv \
  --output ./reachability_output/reachability.html
```

- 点大小 = manipulability
- 点颜色 = dexterity

打开 `reachability.html` 即可查看。
