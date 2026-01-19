#!/usr/bin/env python3
"""
ARM Reachability Analysis Tool

A comprehensive framework for robotic arm reachability analysis using cuRobo
with GPU-accelerated multi-seed IK solving, dexterity and manipulability metrics.

Usage:
    python main.py --urdf <path_to_urdf> [options]

Features:
    - Supports any URDF robot arm
    - GPU-accelerated IK solving with cuRobo
    - Multi-seed IK for finding multiple solutions
    - Voxel-based workspace sampling
    - Dexterity calculation (reachable orientations per position)
    - Yoshikawa manipulability index
    - Interactive 3D visualization with robot model

Output:
    - Point cloud with SIZE = manipulability, COLOR = dexterity
    - CSV data files
    - JSON statistics
    - Interactive HTML visualization
"""

import argparse
import os
import sys
from typing import Optional

# Add package to path
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ARM Reachability Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--urdf', '-u',
        type=str,
        required=True,
        help='Path to URDF file'
    )

    # Robot configuration
    parser.add_argument(
        '--ee-link', '-e',
        type=str,
        default='',
        help='End-effector link name (auto-detect if not specified)'
    )
    parser.add_argument(
        '--base-link', '-b',
        type=str,
        default='',
        help='Base link name (auto-detect if not specified)'
    )

    # Voxel grid configuration
    parser.add_argument(
        '--resolution', '-r',
        type=float,
        default=0.05,
        help='Voxel grid resolution in meters (default: 0.05)'
    )
    parser.add_argument(
        '--x-range',
        type=float,
        nargs=2,
        default=None,
        help='X-axis range [min max] in meters'
    )
    parser.add_argument(
        '--y-range',
        type=float,
        nargs=2,
        default=None,
        help='Y-axis range [min max] in meters'
    )
    parser.add_argument(
        '--z-range',
        type=float,
        nargs=2,
        default=None,
        help='Z-axis range [min max] in meters'
    )
    parser.add_argument(
        '--auto-bounds',
        action='store_true',
        default=True,
        help='Auto-detect workspace bounds using FK sampling (default: True)'
    )
    parser.add_argument(
        '--no-auto-bounds',
        action='store_false',
        dest='auto_bounds',
        help='Disable auto-detection of workspace bounds'
    )

    # IK configuration
    parser.add_argument(
        '--num-seeds', '-s',
        type=int,
        default=32,
        help='Number of IK seeds for multi-seed solving (default: 32)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for GPU processing (default: 1024)'
    )
    parser.add_argument(
        '--pos-threshold',
        type=float,
        default=0.005,
        help='Position error threshold in meters (default: 0.005)'
    )
    parser.add_argument(
        '--rot-threshold',
        type=float,
        default=0.05,
        help='Rotation error threshold in radians (default: 0.05)'
    )

    # Orientation configuration
    parser.add_argument(
        '--orientation-mode',
        type=str,
        choices=['fixed', 'multi_fixed', 'spherical', 'random'],
        default='multi_fixed',
        help='Orientation sampling mode (default: multi_fixed)'
    )
    parser.add_argument(
        '--num-orientations',
        type=int,
        default=6,
        help='Number of orientations for spherical/random modes (default: 6)'
    )

    # Computation options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Computation device (default: cuda:0)'
    )
    parser.add_argument(
        '--no-manipulability',
        action='store_true',
        help='Skip manipulability computation'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./reachability_output',
        help='Output directory (default: ./reachability_output)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='reachability',
        help='Output file prefix (default: reachability)'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip interactive visualization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    return parser.parse_args()


def create_config_from_args(args) -> ReachabilityConfig:
    """Create configuration from command-line arguments."""
    # Start with YAML config if provided
    if args.config:
        config = ReachabilityConfig.from_yaml(args.config)
    else:
        config = ReachabilityConfig()

    # Override with command-line arguments
    config.urdf_path = args.urdf
    config.ee_link = args.ee_link
    config.base_link = args.base_link
    config.output_dir = args.output
    config.device = args.device
    config.compute_manipulability = not args.no_manipulability

    # Voxel config
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

    # IK config
    config.ik.num_seeds = args.num_seeds
    config.ik.batch_size = args.batch_size
    config.ik.position_threshold = args.pos_threshold
    config.ik.rotation_threshold = args.rot_threshold

    # Orientation config
    config.orientation.mode = OrientationMode(args.orientation_mode)
    if args.orientation_mode == 'spherical':
        config.orientation.num_spherical_samples = args.num_orientations
    elif args.orientation_mode == 'random':
        config.orientation.num_random_samples = args.num_orientations

    return config


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("ARM Reachability Analysis Tool")
    print("=" * 60)

    # Validate URDF path
    if not os.path.exists(args.urdf):
        print(f"Error: URDF file not found: {args.urdf}")
        sys.exit(1)

    # Create configuration
    config = create_config_from_args(args)

    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  URDF: {config.urdf_path}")
    print(f"  EE Link: {config.ee_link or 'auto-detect'}")
    print(f"  Base Link: {config.base_link or 'auto-detect'}")
    print(f"  Resolution: {config.voxel.resolution}m")
    print(f"  IK Seeds: {config.ik.num_seeds}")
    print(f"  Orientation Mode: {config.orientation.mode.value}")
    print(f"  Device: {config.device}")
    print(f"  Output: {config.output_dir}")

    # Create analyzer and run analysis
    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()

    # Save results
    analyzer.save_results(result, prefix=args.prefix)

    # Save configuration
    config_path = os.path.join(config.output_dir, f"{args.prefix}_config.yaml")
    config.to_yaml(config_path)
    print(f"  Saved: {args.prefix}_config.yaml")

    # Visualization
    if not args.no_visualization:
        print("\n" + "=" * 60)
        print("Creating Visualization")
        print("=" * 60)

        visualizer = ReachabilityVisualizer(analyzer.robot_config, result)

        # Save HTML
        html_path = os.path.join(config.output_dir, f"{args.prefix}_visualization.html")
        visualizer.visualize_plotly(
            show_robot=True,
            save_html=html_path
        )

        # Save summary plot
        summary_path = os.path.join(config.output_dir, f"{args.prefix}_summary.png")
        try:
            visualizer.create_summary_plot(save_path=summary_path)
        except Exception as e:
            print(f"[Warning] Could not create summary plot: {e}")

        # Show interactive visualization
        print("\nLaunching interactive visualization...")
        try:
            visualizer.visualize_plotly(show_robot=True)
        except Exception as e:
            print(f"[Warning] Could not show interactive visualization: {e}")
            print(f"Please open {html_path} in a web browser.")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.output_dir}/")
    print(f"\nSummary:")
    print(f"  Total voxels: {result.total_points}")
    print(f"  Reachable: {result.reachable_count} ({result.reachability_ratio*100:.1f}%)")
    print(f"  Max dexterity: {result.max_dexterity}")
    print(f"  Mean dexterity: {result.mean_dexterity:.2f}")
    print(f"  Max manipulability: {result.max_manipulability:.4f}")
    print(f"  Mean manipulability: {result.mean_manipulability:.4f}")
    print(f"  Computation time: {result.computation_time:.2f}s")


def quick_analyze(
    urdf_path: str,
    ee_link: str = "",
    resolution: float = 0.05,
    num_seeds: int = 32,
    output_dir: str = "./reachability_output",
    visualize: bool = True
) -> ReachabilityResult:
    """
    Quick analysis function for programmatic use.

    Args:
        urdf_path: Path to URDF file
        ee_link: End-effector link name
        resolution: Voxel resolution
        num_seeds: Number of IK seeds
        output_dir: Output directory
        visualize: Whether to show visualization

    Returns:
        ReachabilityResult

    Example:
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
