#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可达性可视化 + 机械臂模型显示
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# cuRobo 导入
try:
    import torch
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    CUROBO_AVAILABLE = True
except ImportError:
    CUROBO_AVAILABLE = False
    print("[WARN] cuRobo 未安装")


def get_robot_link_positions(robot_config_file: str, joint_config: np.ndarray = None):
    """
    获取机械臂各链接的位置（用于绘制骨架）

    Args:
        robot_config_file: cuRobo 机器人配置文件
        joint_config: 关节配置，如果为 None 则使用零位

    Returns:
        link_positions: 各链接位置列表
    """
    if not CUROBO_AVAILABLE:
        # 返回模拟的 Franka 机械臂位置
        return np.array([
            [0, 0, 0],
            [0, 0, 0.333],
            [0, 0, 0.333],
            [0.0825, 0, 0.649],
            [0, 0, 0.649],
            [0.0825, 0, 1.033],
            [0, 0, 1.033],
            [0.088, 0, 1.033],
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

    # 加载配置
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_config_file,
        None,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )

    ik_solver = IKSolver(ik_config)
    kinematics = ik_solver.robot_config.kinematics

    # 获取关节数量
    n_joints = len(kinematics.joint_names)

    if joint_config is None:
        # 使用零位配置
        joint_config = np.zeros(n_joints)

    # 转换为 tensor
    q = torch.tensor(joint_config, device=device, dtype=torch.float32).unsqueeze(0)

    # 获取所有链接的位姿
    link_positions = []

    # 基座位置
    link_positions.append([0, 0, 0])

    # 通过 FK 获取末端位置
    state = JointState.from_position(q, kinematics.joint_names)

    # 尝试获取所有链接位置
    try:
        # 获取运动学链中的位置
        kin_state = kinematics.get_state(state)

        # 末端位置
        ee_pos = kin_state.ee_position.cpu().numpy().flatten()
        link_positions.append(ee_pos.tolist())

    except Exception as e:
        print(f"[WARN] 获取链接位置失败: {e}")

    return np.array(link_positions)


def create_robot_skeleton(robot_config_file: str = "franka.yml", joint_config: np.ndarray = None):
    """
    创建机械臂骨架的 3D 线段数据

    Returns:
        traces: Plotly trace 列表
    """
    # Franka Panda 的典型链接位置（零位配置）
    # 这些是 DH 参数计算出的近似位置
    if joint_config is None:
        joint_config = np.zeros(7)

    # Franka Panda 零位时的链接位置 (近似值)
    link_positions = np.array([
        [0, 0, 0],           # base
        [0, 0, 0.333],       # link1
        [0, 0, 0.333],       # link2
        [0, -0.316, 0.333],  # link3
        [0.0825, -0.316, 0.333],  # link4
        [0.0825, 0, 0.333],  # link5
        [0.0825 + 0.384, 0, 0.333],  # link6
        [0.0825 + 0.384, 0, 0.333 - 0.107],  # link7 (flange)
        [0.0825 + 0.384 + 0.1, 0, 0.333 - 0.107],  # end effector
    ])

    traces = []

    # 绘制连杆
    traces.append(go.Scatter3d(
        x=link_positions[:, 0],
        y=link_positions[:, 1],
        z=link_positions[:, 2],
        mode='lines+markers',
        line=dict(color='blue', width=8),
        marker=dict(size=6, color='darkblue'),
        name='Robot Arm',
    ))

    # 绘制关节球
    traces.append(go.Scatter3d(
        x=link_positions[:, 0],
        y=link_positions[:, 1],
        z=link_positions[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=['red'] + ['orange'] * (len(link_positions) - 2) + ['green'],
            symbol='circle',
        ),
        name='Joints',
        showlegend=False,
    ))

    # 添加基座
    base_size = 0.15
    theta = np.linspace(0, 2*np.pi, 20)
    base_x = base_size * np.cos(theta)
    base_y = base_size * np.sin(theta)
    base_z = np.zeros_like(theta)

    traces.append(go.Scatter3d(
        x=base_x,
        y=base_y,
        z=base_z,
        mode='lines',
        line=dict(color='gray', width=5),
        name='Base',
        showlegend=False,
    ))

    return traces


def visualize_reachability_with_robot(
    reachable_points_path: str,
    output_html: str = "reachability_with_robot.html",
    robot_config: str = "franka.yml",
    title: str = "Arm Reachability with Robot Model",
):
    """
    可视化可达空间和机械臂模型
    """
    # 加载可达点
    points = np.load(reachable_points_path)
    print(f"[INFO] 加载可达点: {len(points)}")

    traces = []

    # 添加可达点云
    # 采样以提高性能
    if len(points) > 8000:
        indices = np.random.choice(len(points), 8000, replace=False)
        points_sample = points[indices]
    else:
        points_sample = points

    traces.append(go.Scatter3d(
        x=points_sample[:, 0],
        y=points_sample[:, 1],
        z=points_sample[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points_sample[:, 2],  # 按高度着色
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Z (m)'),
        ),
        name=f'Reachable ({len(points)} pts)',
    ))

    # 添加机械臂骨架
    robot_traces = create_robot_skeleton(robot_config)
    traces.extend(robot_traces)

    # 添加坐标轴
    axis_length = 0.3
    for axis, color, name in [
        ([axis_length, 0, 0], 'red', 'X'),
        ([0, axis_length, 0], 'green', 'Y'),
        ([0, 0, axis_length], 'blue', 'Z'),
    ]:
        traces.append(go.Scatter3d(
            x=[0, axis[0]],
            y=[0, axis[1]],
            z=[0, axis[2]],
            mode='lines+text',
            line=dict(color=color, width=6),
            text=['', name],
            textposition='top center',
            name=f'{name} axis',
            showlegend=False,
        ))

    # 创建图形
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # 保存
    fig.write_html(output_html)
    print(f"[SAVE] 已保存到: {output_html}")
    print("用浏览器打开查看（支持鼠标旋转、缩放、平移）")

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="可达性可视化 + 机械臂模型")
    parser.add_argument("data_path", type=str, nargs='?',
                        default="./reachability_output/left_arm_reachable.npy",
                        help="可达点云文件")
    parser.add_argument("-o", "--output", type=str, default="reachability_with_robot.html",
                        help="输出 HTML 文件")
    parser.add_argument("-t", "--title", type=str, default="Franka Panda Reachability",
                        help="图表标题")

    args = parser.parse_args()

    visualize_reachability_with_robot(
        args.data_path,
        output_html=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
