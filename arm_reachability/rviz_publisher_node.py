#!/usr/bin/env python3
"""
RViz 可达性点云发布节点

支持功能：
- 单臂/双臂可达空间可视化
- 机器人 URDF 模型显示
- 左臂/右臂/双臂选择

使用方式：
1. 单臂（默认）:
   python3 rviz_publisher_node.py --data-dir ./reachability_output

2. 带 URDF 模型:
   python3 rviz_publisher_node.py --data-dir ./reachability_output --urdf /path/to/robot.urdf

3. 双臂可视化:
   python3 rviz_publisher_node.py --arm both \
       --left-urdf /path/to/left.urdf --left-data ./left_output \
       --right-urdf /path/to/right.urdf --right-data ./right_output
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='发布可达性点云到 RViz2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单臂可视化
  python3 rviz_publisher_node.py --data-dir ./reachability_output

  # 单臂 + URDF 模型
  python3 rviz_publisher_node.py --data-dir ./output --urdf ./robot.urdf

  # 左臂可视化
  python3 rviz_publisher_node.py --arm left --left-data ./left_output --left-urdf ./left.urdf

  # 双臂可视化
  python3 rviz_publisher_node.py --arm both \\
      --left-data ./left_output --left-urdf ./left.urdf \\
      --right-data ./right_output --right-urdf ./right.urdf
        """
    )

    # 臂选择
    parser.add_argument('--arm', type=str, default='single',
                        choices=['single', 'left', 'right', 'both'],
                        help='可视化哪个臂: single(单臂), left(左臂), right(右臂), both(双臂)')

    # 单臂模式参数
    parser.add_argument('--data-dir', type=str, default='./reachability_output',
                        help='数据目录路径 (单臂模式)')
    parser.add_argument('--prefix', type=str, default='reachability',
                        help='数据文件前缀 (单臂模式)')
    parser.add_argument('--urdf', type=str, default=None,
                        help='URDF 文件路径 (单臂模式)')

    # 左臂参数
    parser.add_argument('--left-data', type=str, default=None,
                        help='左臂数据目录')
    parser.add_argument('--left-prefix', type=str, default='reachability',
                        help='左臂数据文件前缀')
    parser.add_argument('--left-urdf', type=str, default=None,
                        help='左臂 URDF 文件路径')

    # 右臂参数
    parser.add_argument('--right-data', type=str, default=None,
                        help='右臂数据目录')
    parser.add_argument('--right-prefix', type=str, default='reachability',
                        help='右臂数据文件前缀')
    parser.add_argument('--right-urdf', type=str, default=None,
                        help='右臂 URDF 文件路径')

    # 通用参数
    parser.add_argument('--frame-id', type=str, default='world',
                        help='坐标系名称')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='发布频率 (Hz)，0 表示只发布一次')
    parser.add_argument('--topic', type=str, default='/reachability',
                        help='基础 Topic 名称')
    parser.add_argument('--no-tf', action='store_true',
                        help='禁用 TF 发布（当 robot_state_publisher 已运行时使用）')
    parser.add_argument('--translation-only', action='store_true',
                        help='仅使用平移，不应用 base_transform 旋转（用于调试点云位置）')
    parser.add_argument('--color-mode', type=str, default='dexterity',
                        choices=['dexterity', 'arm', 'tint'],
                        help='颜色模式: dexterity(灵活度色阶), arm(固定臂色), tint(臂色偏色+灵活度)')

    # 解析参数
    args, unknown = parser.parse_known_args()

    # 添加模块路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from arm_reachability.ros2_visualization import (
        RVizPublisherConfig,
        RVizReachabilityPublisher,
        ArmConfig
    )

    print("=" * 60)
    print("RViz2 可达性点云发布器")
    print("=" * 60)
    print(f"臂模式: {args.arm}")
    print(f"坐标系: {args.frame_id}")
    print(f"基础 Topic: {args.topic}")
    print(f"发布频率: {args.rate} Hz")
    print(f"颜色模式: {args.color_mode}")
    if args.translation_only:
        print(f"变换模式: 仅平移（忽略 base_transform 旋转）")
    print("=" * 60)

    # 构建臂配置列表
    arms = []

    if args.arm == 'single':
        # 单臂模式
        arm = ArmConfig(
            name="default",
            urdf_path=args.urdf,
            data_dir=args.data_dir,
            data_prefix=args.prefix,
            topic_suffix=""
        )
        arms.append(arm)
        print(f"单臂配置:")
        print(f"  数据目录: {args.data_dir}")
        print(f"  文件前缀: {args.prefix}")
        print(f"  URDF: {args.urdf or '未指定'}")

    elif args.arm == 'left':
        # 只有左臂
        left_data = args.left_data or args.data_dir
        arm = ArmConfig(
            name="left",
            urdf_path=args.left_urdf or args.urdf,
            data_dir=left_data,
            data_prefix=args.left_prefix,
            topic_suffix="_left",
            color_override=None  # 使用默认颜色映射
        )
        arms.append(arm)
        print(f"左臂配置:")
        print(f"  数据目录: {left_data}")
        print(f"  文件前缀: {args.left_prefix}")
        print(f"  URDF: {arm.urdf_path or '未指定'}")

    elif args.arm == 'right':
        # 只有右臂
        right_data = args.right_data or args.data_dir
        arm = ArmConfig(
            name="right",
            urdf_path=args.right_urdf or args.urdf,
            data_dir=right_data,
            data_prefix=args.right_prefix,
            topic_suffix="_right",
            color_override=None  # 使用默认颜色映射
        )
        arms.append(arm)
        print(f"右臂配置:")
        print(f"  数据目录: {right_data}")
        print(f"  文件前缀: {args.right_prefix}")
        print(f"  URDF: {arm.urdf_path or '未指定'}")

    elif args.arm == 'both':
        # 双臂
        left_data = args.left_data or args.data_dir
        right_data = args.right_data or args.data_dir

        # 使用共享的 URDF（如果单独的臂 URDF 未指定）
        shared_urdf = args.urdf
        left_urdf = args.left_urdf or shared_urdf
        right_urdf = args.right_urdf or shared_urdf

        left_arm = ArmConfig(
            name="left",
            urdf_path=left_urdf,
            data_dir=left_data,
            data_prefix=args.left_prefix,
            topic_suffix="_left",
            color_override=(0.0, 0.5, 1.0)  # 蓝色系表示左臂
        )
        arms.append(left_arm)

        right_arm = ArmConfig(
            name="right",
            urdf_path=right_urdf,
            data_dir=right_data,
            data_prefix=args.right_prefix,
            topic_suffix="_right",
            color_override=(1.0, 0.5, 0.0)  # 橙色系表示右臂
        )
        arms.append(right_arm)

        print(f"左臂配置:")
        print(f"  数据目录: {left_data}")
        print(f"  文件前缀: {args.left_prefix}")
        print(f"  URDF: {left_arm.urdf_path or '未指定'}")
        print(f"  颜色: 蓝色系")
        print(f"右臂配置:")
        print(f"  数据目录: {right_data}")
        print(f"  文件前缀: {args.right_prefix}")
        print(f"  URDF: {right_arm.urdf_path or '未指定'}")
        print(f"  颜色: 橙色系")

    print("=" * 60)

    # 验证数据文件
    for arm in arms:
        if arm.data_dir:
            required_files = [
                f"{arm.data_prefix}_grid_points.npy",
                f"{arm.data_prefix}_reachable_mask.npy",
            ]
            missing = []
            for f in required_files:
                path = os.path.join(arm.data_dir, f)
                if not os.path.exists(path):
                    missing.append(f)

            if missing:
                print(f"[警告] {arm.name} 臂缺少文件: {missing}")
            else:
                print(f"[OK] {arm.name} 臂数据文件验证通过")

    print("=" * 60)

    # 创建配置并启动
    config = RVizPublisherConfig(
        frame_id=args.frame_id,
        publish_rate=args.rate,
        base_topic=args.topic,
        arms=arms,
        publish_tf=not args.no_tf,  # 当 robot_state_publisher 运行时禁用 TF
        use_base_transform_rotation=not args.translation_only,
        color_mode=args.color_mode
    )

    if args.no_tf:
        print("[TF] 禁用 TF 发布（依赖 robot_state_publisher）")

    publisher = RVizReachabilityPublisher(config)

    try:
        print("开始发布 (Ctrl+C 停止)...")
        publisher.spin()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        publisher.destroy()
        print("节点已关闭")


if __name__ == '__main__':
    main()
