#!/usr/bin/env python3
"""
机械臂可达性分析工具

基于cuRobo的GPU加速多种子IK求解，支持灵活度和可操作度分析的通用框架。

用法:
    python main.py --urdf <urdf文件路径> [选项]

功能特性:
    - 支持任意URDF格式的机械臂
    - 使用cuRobo进行GPU加速IK求解
    - 多种子IK用于寻找多个解
    - 基于体素的工作空间采样
    - 灵活度计算（每个位置可达的姿态数量）
    - Yoshikawa可操作度指标
    - 交互式3D可视化（含机器人模型）

输出:
    - 点云：大小=可操作度，颜色=灵活度
    - CSV数据文件
    - JSON统计信息
    - 交互式HTML可视化
"""

import argparse
import os
import sys
from typing import Optional

# 添加包路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_reachability.config import (
    ReachabilityConfig,
    VoxelConfig,
    IKConfig,
    OrientationConfig,
    OrientationMode
)
from arm_reachability.urdf_parser import URDFParser
from arm_reachability.reachability import ReachabilityAnalyzer, ReachabilityResult
from arm_reachability.visualization import ReachabilityVisualizer, visualize_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="机械臂可达性分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 必需参数
    parser.add_argument(
        '--urdf', '-u',
        type=str,
        required=True,
        help='URDF文件路径'
    )

    # 机器人配置
    parser.add_argument(
        '--ee-link', '-e',
        type=str,
        default='',
        help='末端执行器链接名（不指定则自动检测）'
    )
    parser.add_argument(
        '--base-link', '-b',
        type=str,
        default='',
        help='基座链接名（不指定则自动检测）'
    )

    # 体素网格配置
    parser.add_argument(
        '--resolution', '-r',
        type=float,
        default=0.05,
        help='体素分辨率，单位米（默认: 0.05）'
    )
    parser.add_argument(
        '--x-range',
        type=float,
        nargs=2,
        default=None,
        help='X轴范围 [最小值 最大值]，单位米'
    )
    parser.add_argument(
        '--y-range',
        type=float,
        nargs=2,
        default=None,
        help='Y轴范围 [最小值 最大值]，单位米'
    )
    parser.add_argument(
        '--z-range',
        type=float,
        nargs=2,
        default=None,
        help='Z轴范围 [最小值 最大值]，单位米'
    )
    parser.add_argument(
        '--auto-bounds',
        action='store_true',
        default=True,
        help='使用FK采样自动检测工作空间边界（默认: True）'
    )
    parser.add_argument(
        '--no-auto-bounds',
        action='store_false',
        dest='auto_bounds',
        help='禁用自动检测工作空间边界'
    )

    # IK配置
    parser.add_argument(
        '--num-seeds', '-s',
        type=int,
        default=32,
        help='多种子IK的种子数量（默认: 32）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='GPU批处理大小（默认: 1024）'
    )
    parser.add_argument(
        '--pos-threshold',
        type=float,
        default=0.005,
        help='位置误差阈值，单位米（默认: 0.005）'
    )
    parser.add_argument(
        '--rot-threshold',
        type=float,
        default=0.05,
        help='旋转误差阈值，单位弧度（默认: 0.05）'
    )

    # 姿态配置
    parser.add_argument(
        '--orientation-mode',
        type=str,
        choices=['fixed', 'multi_fixed', 'spherical', 'random'],
        default='multi_fixed',
        help='姿态采样模式（默认: multi_fixed）'
    )
    parser.add_argument(
        '--num-orientations',
        type=int,
        default=6,
        help='球面/随机模式的姿态数量（默认: 6）'
    )

    # 计算选项
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备（默认: cuda:0）'
    )
    parser.add_argument(
        '--no-manipulability',
        action='store_true',
        help='跳过可操作度计算'
    )

    # 输出选项
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./reachability_output',
        help='输出目录（默认: ./reachability_output）'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='reachability',
        help='输出文件前缀（默认: reachability）'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='跳过交互式可视化'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML配置文件路径'
    )

    return parser.parse_args()


def create_config_from_args(args) -> ReachabilityConfig:
    """从命令行参数创建配置"""
    # 如果提供了YAML配置，先加载
    if args.config:
        config = ReachabilityConfig.from_yaml(args.config)
    else:
        config = ReachabilityConfig()

    # 用命令行参数覆盖
    config.urdf_path = args.urdf
    config.ee_link = args.ee_link
    config.base_link = args.base_link
    config.output_dir = args.output
    config.device = args.device
    config.compute_manipulability = not args.no_manipulability

    # 体素配置
    config.voxel.resolution = args.resolution
    config.voxel.auto_detect = args.auto_bounds

    if args.x_range:
        config.voxel.x_range = tuple(args.x_range)
        config.voxel.auto_detect = False
    if args.y_range:
        config.voxel.y_range = tuple(args.y_range)
        config.voxel.auto_detect = False
    if args.z_range:
        config.voxel.z_range = tuple(args.z_range)
        config.voxel.auto_detect = False

    # IK配置
    config.ik.num_seeds = args.num_seeds
    config.ik.batch_size = args.batch_size
    config.ik.position_threshold = args.pos_threshold
    config.ik.rotation_threshold = args.rot_threshold

    # 姿态配置
    config.orientation.mode = OrientationMode(args.orientation_mode)
    if args.orientation_mode == 'spherical':
        config.orientation.num_spherical_samples = args.num_orientations
    elif args.orientation_mode == 'random':
        config.orientation.num_random_samples = args.num_orientations

    return config


def main():
    """主入口函数"""
    args = parse_args()

    print("=" * 60)
    print("机械臂可达性分析工具")
    print("=" * 60)

    # 验证URDF路径
    if not os.path.exists(args.urdf):
        print(f"错误: URDF文件不存在: {args.urdf}")
        sys.exit(1)

    # 创建配置
    config = create_config_from_args(args)

    # 打印配置摘要
    print(f"\n配置:")
    print(f"  URDF: {config.urdf_path}")
    print(f"  末端链接: {config.ee_link or '自动检测'}")
    print(f"  基座链接: {config.base_link or '自动检测'}")
    print(f"  分辨率: {config.voxel.resolution}m")
    print(f"  IK种子数: {config.ik.num_seeds}")
    print(f"  姿态模式: {config.orientation.mode.value}")
    print(f"  设备: {config.device}")
    print(f"  输出目录: {config.output_dir}")

    # 创建分析器并运行
    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()

    # 保存结果
    analyzer.save_results(result, prefix=args.prefix)

    # 保存配置
    config_path = os.path.join(config.output_dir, f"{args.prefix}_config.yaml")
    config.to_yaml(config_path)
    print(f"  已保存: {args.prefix}_config.yaml")

    # 可视化
    if not args.no_visualization:
        print("\n" + "=" * 60)
        print("创建可视化")
        print("=" * 60)

        visualizer = ReachabilityVisualizer(analyzer.robot_config, result)

        # 保存HTML
        html_path = os.path.join(config.output_dir, f"{args.prefix}_visualization.html")
        visualizer.visualize_plotly(
            show_robot=True,
            save_html=html_path
        )

        # 保存汇总图
        summary_path = os.path.join(config.output_dir, f"{args.prefix}_summary.png")
        try:
            visualizer.create_summary_plot(save_path=summary_path)
        except Exception as e:
            print(f"[警告] 无法创建汇总图: {e}")

        # 显示交互式可视化
        print("\n启动交互式可视化...")
        try:
            visualizer.visualize_plotly(show_robot=True)
        except Exception as e:
            print(f"[警告] 无法显示交互式可视化: {e}")
            print(f"请在浏览器中打开 {html_path}")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"\n结果已保存到: {config.output_dir}/")
    print(f"\n汇总:")
    print(f"  总体素数: {result.total_points}")
    print(f"  可达点数: {result.reachable_count} ({result.reachability_ratio*100:.1f}%)")
    print(f"  最大灵活度: {result.max_dexterity}")
    print(f"  平均灵活度: {result.mean_dexterity:.2f}")
    print(f"  最大可操作度: {result.max_manipulability:.4f}")
    print(f"  平均可操作度: {result.mean_manipulability:.4f}")
    print(f"  计算时间: {result.computation_time:.2f}秒")


def quick_analyze(
    urdf_path: str,
    ee_link: str = "",
    resolution: float = 0.05,
    num_seeds: int = 32,
    output_dir: str = "./reachability_output",
    visualize: bool = True
) -> ReachabilityResult:
    """
    快速分析函数（编程接口）

    参数:
        urdf_path: URDF文件路径
        ee_link: 末端执行器链接名
        resolution: 体素分辨率
        num_seeds: IK种子数量
        output_dir: 输出目录
        visualize: 是否显示可视化

    返回:
        ReachabilityResult

    示例:
        >>> from main import quick_analyze
        >>> result = quick_analyze("robot.urdf", resolution=0.05)
    """
    config = ReachabilityConfig(
        urdf_path=urdf_path,
        ee_link=ee_link,
        output_dir=output_dir
    )
    config.voxel.resolution = resolution
    config.voxel.auto_detect = True
    config.ik.num_seeds = num_seeds

    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()
    analyzer.save_results(result)

    if visualize:
        visualizer = ReachabilityVisualizer(analyzer.robot_config, result)
        visualizer.visualize_plotly(show_robot=True)

    return result


if __name__ == '__main__':
    main()
