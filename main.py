#!/usr/bin/env python3
"""
机械臂可达性分析工具

基于cuRobo的GPU加速多种子IK求解，支持灵活度和可操作度分析的通用框架。
支持单臂和多臂（双臂）同时分析。

用法:
    # 单臂分析（自动检测）
    python main.py --urdf <urdf文件路径>

    # 双臂同时分析
    python main.py --urdf <urdf文件路径> --arm both

    # 仅分析左臂
    python main.py --urdf <urdf文件路径> --arm left

    # 检测URDF中的机械臂
    python main.py --urdf <urdf文件路径> --detect-arms

功能特性:
    - 支持任意URDF格式的机械臂
    - 自动检测多臂机器人的各个机械臂链
    - 支持单臂/双臂/所有臂同时分析
    - 使用cuRobo进行GPU加速IK求解
    - 多种子IK用于寻找多个解
    - 基于体素的工作空间采样
    - 灵活度计算（每个位置可达的姿态数量）
    - Yoshikawa可操作度指标
    - 交互式3D可视化（含机器人模型）
    - RViz2点云可视化

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
    OrientationMode,
    ArmMode,
    MultiArmConfig
)
from arm_reachability.urdf_parser import URDFParser, detect_arms
from arm_reachability.reachability import (
    ReachabilityAnalyzer,
    ReachabilityResult,
    MultiArmReachabilityAnalyzer,
    MultiArmReachabilityResult
)
from arm_reachability.visualization import ReachabilityVisualizer, visualize_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="机械臂可达性分析工具 - 支持单臂/双臂分析",
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

    # 机械臂选择
    parser.add_argument(
        '--arm', '-a',
        type=str,
        choices=['auto', 'left', 'right', 'both', 'all', 'single'],
        default='auto',
        help='机械臂分析模式: auto=自动检测单臂, left=左臂, right=右臂, both=双臂, all=所有臂, single=指定臂名（默认: auto）'
    )
    parser.add_argument(
        '--arm-name',
        type=str,
        default='',
        help='指定臂名称（仅当 --arm single 时使用）'
    )
    parser.add_argument(
        '--detect-arms',
        action='store_true',
        help='仅检测并显示URDF中的机械臂信息，不进行分析'
    )

    # 机器人配置
    parser.add_argument(
        '--ee-link', '-e',
        type=str,
        default='',
        help='末端执行器链接名（不指定则自动检测，仅单臂模式有效）'
    )
    parser.add_argument(
        '--base-link', '-b',
        type=str,
        default='',
        help='基座链接名（不指定则自动检测，仅单臂模式有效）'
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
        '--points-frame',
        type=str,
        choices=['base', 'world'],
        default='world',
        help='输出点云坐标系（默认: world）'
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
        '--rviz',
        action='store_true',
        help='完成分析后发布到 RViz2（需要 ROS 2 环境）'
    )
    parser.add_argument(
        '--rviz-topic',
        type=str,
        default='/reachability/points',
        help='RViz 点云 Topic 名称（默认: /reachability/points）'
    )
    parser.add_argument(
        '--rviz-frame',
        type=str,
        default='world',
        help='RViz 固定坐标系（默认: world）'
    )
    parser.add_argument(
        '--rviz-color-mode',
        type=str,
        choices=['dexterity', 'arm', 'tint'],
        default='dexterity',
        help='RViz 点云颜色模式（默认: dexterity）'
    )
    parser.add_argument(
        '--rviz-once',
        action='store_true',
        help='仅发布一次后退出（默认持续发布，需要手动 Ctrl+C 停止）'
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
    config.points_frame = args.points_frame

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

    # 多臂配置
    config.multi_arm.mode = ArmMode(args.arm)
    config.multi_arm.arm_name = args.arm_name

    return config


def detect_arms_info(urdf_path: str):
    """检测并显示URDF中的机械臂信息"""
    print("=" * 60)
    print("检测机械臂")
    print("=" * 60)

    arm_info = detect_arms(urdf_path)

    if not arm_info:
        print("\n未检测到机械臂链。")
        print("可能原因:")
        print("  1. URDF中没有包含末端执行器关键词（tcp, ee, tool等）")
        print("  2. 机械臂链的关节数量少于3个")
        print("\n建议手动指定 --base-link 和 --ee-link 参数")
        return

    print(f"\n检测到 {len(arm_info)} 条机械臂:\n")

    for name, info in arm_info.items():
        print(f"  [{name}]")
        print(f"    基座链接: {info['base_link']}")
        print(f"    末端链接: {info['ee_link']}")
        print(f"    自由度:   {info['num_dof']}")
        print(f"    关节:     {', '.join(info['joints'])}")
        print()

    print("使用示例:")
    if len(arm_info) == 1:
        name = list(arm_info.keys())[0]
        print(f"  python main.py --urdf {urdf_path}")
    else:
        print(f"  # 分析双臂:")
        print(f"  python main.py --urdf {urdf_path} --arm both")
        print(f"  # 仅分析左臂:")
        print(f"  python main.py --urdf {urdf_path} --arm left")
        print(f"  # 分析所有臂:")
        print(f"  python main.py --urdf {urdf_path} --arm all")


def run_single_arm_analysis(config: ReachabilityConfig, args) -> ReachabilityResult:
    """运行单臂分析"""
    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()

    # 保存结果
    analyzer.save_results(result, prefix=args.prefix)

    # 保存配置
    config_path = os.path.join(config.output_dir, f"{args.prefix}_config.yaml")
    config.to_yaml(config_path)
    print(f"  已保存: {args.prefix}_config.yaml")

    # 可视化
    if not args.no_visualization or args.rviz:
        visualizer = ReachabilityVisualizer(analyzer.robot_config, result)

        if not args.no_visualization:
            print("\n" + "=" * 60)
            print("创建可视化")
            print("=" * 60)

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

        if args.rviz:
            run_rviz_visualization(visualizer, args)

    return result


def run_multi_arm_analysis(config: ReachabilityConfig, args) -> MultiArmReachabilityResult:
    """运行多臂分析"""
    analyzer = MultiArmReachabilityAnalyzer(config)
    result = analyzer.analyze()

    # 保存配置
    config_path = os.path.join(config.output_dir, f"{args.prefix}_config.yaml")
    config.to_yaml(config_path)
    print(f"  已保存: {args.prefix}_config.yaml")

    # RViz可视化（多臂）
    if args.rviz and result.results:
        print("\n" + "=" * 60)
        print("发布多臂可达性到 RViz2")
        print("=" * 60)
        try:
            from arm_reachability.ros2_visualization import MultiArmRVizPublisher, RVizPublisherConfig, ArmConfig

            # 创建多臂RViz发布器配置
            rviz_config = RVizPublisherConfig(
                frame_id=args.rviz_frame,
                base_topic=args.rviz_topic.rsplit('/', 1)[0] if '/' in args.rviz_topic else "/reachability",
                color_mode=args.rviz_color_mode
            )

            for arm_name, arm_result in result.results.items():
                arm_output_dir = os.path.join(config.output_dir, arm_name)
                arm_cfg = ArmConfig(
                    name=arm_name,
                    urdf_path=config.urdf_path,
                    data_dir=arm_output_dir,
                    data_prefix=f"reachability_{arm_name}",
                    color_override=arm_result.color,
                    topic_suffix=f"_{arm_name}"
                )
                rviz_config.arms.append(arm_cfg)

            publisher = MultiArmRVizPublisher(rviz_config)
            publisher.run(publish_once=args.rviz_once)

        except ImportError as e:
            print(f"[错误] 未找到 ROS 2 依赖: {e}")
            print("请确保已安装 ROS 2 并 source 环境后重试。")
        except Exception as e:
            print(f"[错误] 发布到 RViz 失败: {e}")

    return result


def run_rviz_visualization(visualizer, args):
    """运行RViz可视化"""
    print("\n" + "=" * 60)
    print("发布到 RViz2")
    print("=" * 60)
    try:
        visualizer.visualize_rviz(
            frame_id=args.rviz_frame,
            topic=args.rviz_topic,
            keep_alive=not args.rviz_once,
            color_mode=args.rviz_color_mode
        )
    except ImportError as e:
        print(f"[错误] 未找到 ROS 2 依赖: {e}")
        print("请确保已安装 ROS 2 并 source 环境后重试。")
    except Exception as e:
        print(f"[错误] 发布到 RViz 失败: {e}")
        print("请检查 ROS 环境是否正确启动。")


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

    # 仅检测机械臂模式
    if args.detect_arms:
        detect_arms_info(args.urdf)
        return

    # 创建配置
    config = create_config_from_args(args)

    # 打印配置摘要
    print(f"\n配置:")
    print(f"  URDF: {config.urdf_path}")
    print(f"  分析模式: {config.multi_arm.mode.value}")
    if config.multi_arm.mode == ArmMode.SINGLE:
        print(f"  指定臂名: {config.multi_arm.arm_name}")
    if config.ee_link:
        print(f"  末端链接: {config.ee_link}")
    if config.base_link:
        print(f"  基座链接: {config.base_link}")
    print(f"  分辨率: {config.voxel.resolution}m")
    print(f"  IK种子数: {config.ik.num_seeds}")
    print(f"  姿态模式: {config.orientation.mode.value}")
    print(f"  设备: {config.device}")
    print(f"  输出目录: {config.output_dir}")

    # 根据模式选择分析器
    is_multi_arm = config.multi_arm.mode in [ArmMode.BOTH, ArmMode.ALL]

    if is_multi_arm:
        # 多臂分析
        result = run_multi_arm_analysis(config, args)

        print("\n" + "=" * 60)
        print("多臂分析完成!")
        print("=" * 60)
        print(f"\n结果已保存到: {config.output_dir}/")
        print(f"\n各臂汇总:")
        for name, arm_result in result.results.items():
            print(f"\n  [{name}]")
            print(f"    总体素数: {arm_result.total_points}")
            print(f"    可达点数: {arm_result.reachable_count} ({arm_result.reachability_ratio*100:.1f}%)")
            print(f"    最大灵活度: {arm_result.max_dexterity}")
            print(f"    平均灵活度: {arm_result.mean_dexterity:.2f}")
            print(f"    平均可操作度: {arm_result.mean_manipulability:.4f}")
            print(f"    计算时间: {arm_result.computation_time:.2f}秒")
        print(f"\n总计算时间: {result.total_computation_time:.2f}秒")

    else:
        # 单臂分析
        result = run_single_arm_analysis(config, args)

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


def quick_analyze_multi_arm(
    urdf_path: str,
    arm_mode: str = "both",
    resolution: float = 0.05,
    num_seeds: int = 32,
    output_dir: str = "./reachability_output"
) -> MultiArmReachabilityResult:
    """
    多臂快速分析函数（编程接口）

    参数:
        urdf_path: URDF文件路径
        arm_mode: 臂模式 ('left', 'right', 'both', 'all')
        resolution: 体素分辨率
        num_seeds: IK种子数量
        output_dir: 输出目录

    返回:
        MultiArmReachabilityResult

    示例:
        >>> from main import quick_analyze_multi_arm
        >>> result = quick_analyze_multi_arm("robot.urdf", arm_mode="both")
    """
    config = ReachabilityConfig(
        urdf_path=urdf_path,
        output_dir=output_dir
    )
    config.voxel.resolution = resolution
    config.voxel.auto_detect = True
    config.ik.num_seeds = num_seeds
    config.multi_arm.mode = ArmMode(arm_mode)

    analyzer = MultiArmReachabilityAnalyzer(config)
    return analyzer.analyze()


if __name__ == '__main__':
    main()
