#!/usr/bin/env python3
"""
Example usage of the ARM Reachability Analysis Framework.

This script demonstrates how to use the framework programmatically.
"""

import os
import sys
import numpy as np

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_reachability import (
    ReachabilityConfig,
    VoxelConfig,
    IKConfig,
    OrientationConfig,
    OrientationMode,
    URDFParser,
    CuroboConfigGenerator,
    MultiSeedIKSolver,
    ReachabilityAnalyzer,
    ManipulabilityCalculator,
    ReachabilityVisualizer
)


def example_basic_analysis(urdf_path: str):
    """
    Example 1: Basic reachability analysis with default settings.

    This is the simplest way to analyze a robot arm.
    """
    print("=" * 60)
    print("Example 1: Basic Reachability Analysis")
    print("=" * 60)

    # Create analyzer with defaults
    analyzer = ReachabilityAnalyzer.from_urdf(
        urdf_path=urdf_path,
        resolution=0.05,  # 5cm voxel resolution
        num_seeds=32,     # 32 IK seeds for finding multiple solutions
        output_dir="./output_basic"
    )

    # Run analysis
    result = analyzer.analyze()

    # Save results
    analyzer.save_results(result)

    # Print summary
    print(f"\nResults:")
    print(f"  Reachable points: {result.reachable_count}/{result.total_points}")
    print(f"  Reachability: {result.reachability_ratio*100:.1f}%")
    print(f"  Max dexterity: {result.max_dexterity}")
    print(f"  Mean manipulability: {result.mean_manipulability:.4f}")

    return result


def example_custom_config(urdf_path: str):
    """
    Example 2: Custom configuration for detailed analysis.

    This shows how to configure all parameters.
    """
    print("=" * 60)
    print("Example 2: Custom Configuration Analysis")
    print("=" * 60)

    # Create custom configuration
    config = ReachabilityConfig(
        urdf_path=urdf_path,
        ee_link="",  # Auto-detect
        base_link="",  # Auto-detect
        output_dir="./output_custom",
        device="cuda:0",
        compute_manipulability=True,
        verbose=True
    )

    # Configure voxel grid
    config.voxel = VoxelConfig(
        x_range=(-0.8, 0.8),
        y_range=(-0.8, 0.8),
        z_range=(0.0, 1.2),
        resolution=0.04,  # 4cm resolution
        auto_detect=False  # Use specified bounds
    )

    # Configure IK solver
    config.ik = IKConfig(
        num_seeds=64,  # More seeds for better coverage
        position_threshold=0.003,  # 3mm accuracy
        rotation_threshold=0.03,   # ~1.7 degree accuracy
        batch_size=2048,  # Larger batches for GPU
        collision_check=True,
        self_collision_check=True
    )

    # Configure orientations
    config.orientation = OrientationConfig(
        mode=OrientationMode.MULTI_FIXED,
        multi_orientations_euler=[
            [0, 0, 0],       # Z-down
            [180, 0, 0],     # Z-up
            [90, 0, 0],      # X-forward
            [-90, 0, 0],     # X-backward
            [0, 90, 0],      # Y-left
            [0, -90, 0],     # Y-right
            [45, 0, 0],      # Diagonal
            [-45, 0, 0],     # Diagonal
        ]
    )

    # Create analyzer and run
    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()
    analyzer.save_results(result, prefix="custom_analysis")

    return result


def example_spherical_orientations(urdf_path: str):
    """
    Example 3: Analysis with spherical orientation sampling.

    This tests reachability with evenly distributed orientations
    on a sphere, useful for omnidirectional tasks.
    """
    print("=" * 60)
    print("Example 3: Spherical Orientation Sampling")
    print("=" * 60)

    config = ReachabilityConfig(
        urdf_path=urdf_path,
        output_dir="./output_spherical"
    )

    # Use spherical orientation sampling
    config.orientation = OrientationConfig(
        mode=OrientationMode.SPHERICAL,
        num_spherical_samples=26  # Covers all directions
    )

    config.voxel.resolution = 0.06  # Coarser for faster computation
    config.ik.num_seeds = 32

    analyzer = ReachabilityAnalyzer(config)
    result = analyzer.analyze()
    analyzer.save_results(result)

    # The dexterity will now represent how many of the 26 orientations
    # are reachable at each position
    print(f"\nSpherical analysis results:")
    print(f"  Max dexterity: {result.max_dexterity}/26 orientations")

    return result


def example_visualization(urdf_path: str, result=None):
    """
    Example 4: Visualization options.

    Shows different ways to visualize results.
    """
    print("=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)

    # If no result provided, do a quick analysis
    if result is None:
        analyzer = ReachabilityAnalyzer.from_urdf(
            urdf_path=urdf_path,
            resolution=0.06,
            output_dir="./output_viz"
        )
        result = analyzer.analyze()
        robot_config = analyzer.robot_config
    else:
        parser = URDFParser(urdf_path)
        robot_config = parser.parse()

    # Create visualizer
    visualizer = ReachabilityVisualizer(robot_config, result)

    # Option 1: Interactive Plotly visualization (HTML)
    print("\n1. Creating interactive HTML visualization...")
    visualizer.visualize_plotly(
        show_robot=True,
        show_points=True,
        point_size_range=(2.0, 15.0),
        robot_opacity=0.5,
        save_html="./output_viz/interactive.html"
    )

    # Option 2: Summary plot with multiple views
    print("2. Creating summary plot...")
    visualizer.create_summary_plot(save_path="./output_viz/summary.png")

    # Option 3: Matplotlib static visualization
    print("3. Creating matplotlib visualization...")
    try:
        visualizer.visualize_matplotlib(
            point_size_range=(5.0, 100.0),
            save_path="./output_viz/matplotlib.png"
        )
    except Exception as e:
        print(f"  Matplotlib visualization skipped: {e}")

    print("\nVisualization files saved to ./output_viz/")


def example_urdf_parsing(urdf_path: str):
    """
    Example 5: URDF parsing and cuRobo config generation.

    Useful for debugging robot configuration.
    """
    print("=" * 60)
    print("Example 5: URDF Parsing")
    print("=" * 60)

    # Parse URDF
    parser = URDFParser(urdf_path)
    robot_config = parser.parse()

    print(f"\nRobot Information:")
    print(f"  Name: {robot_config.name}")
    print(f"  Base link: {robot_config.base_link}")
    print(f"  EE link: {robot_config.ee_link}")
    print(f"  DOF: {robot_config.num_dof}")

    print(f"\nActive Joints:")
    for joint in robot_config.active_joints:
        print(f"  - {joint.name}: {joint.type}")
        print(f"    Limits: [{joint.lower_limit:.3f}, {joint.upper_limit:.3f}]")

    print(f"\nLinks:")
    for link in robot_config.links:
        mesh = link.visual_mesh or "No mesh"
        print(f"  - {link.name}: {mesh}")

    # Generate cuRobo config
    generator = CuroboConfigGenerator(robot_config)
    curobo_config = generator.generate("./output_viz/curobo_config.yaml")

    print(f"\ncuRobo configuration saved to ./output_viz/curobo_config.yaml")

    return robot_config


def example_ik_solver(urdf_path: str):
    """
    Example 6: Direct IK solver usage.

    Shows how to use the IK solver independently.
    """
    print("=" * 60)
    print("Example 6: Direct IK Solver Usage")
    print("=" * 60)

    # Parse URDF
    parser = URDFParser(urdf_path)
    robot_config = parser.parse()

    # Create IK solver
    ik_config = IKConfig(
        num_seeds=32,
        position_threshold=0.005,
        rotation_threshold=0.05
    )

    solver = MultiSeedIKSolver(
        robot_config=robot_config,
        ik_config=ik_config,
        device="cuda:0"
    )

    # Test single pose IK
    print("\nSolving IK for a single target pose...")
    target_position = np.array([0.5, 0.0, 0.5])
    target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    result = solver.solve_single(target_position, target_orientation)

    print(f"  Target position: {target_position}")
    print(f"  Success: {result.success}")
    print(f"  Number of solutions found: {result.num_solutions}")

    if result.success:
        print(f"  Best solution: {result.solutions[0]}")

    # Test batch IK
    print("\nSolving IK for batch of target poses...")
    batch_positions = np.array([
        [0.4, 0.0, 0.5],
        [0.5, 0.1, 0.4],
        [0.3, -0.2, 0.6],
        [0.6, 0.0, 0.3],
    ])
    batch_orientations = np.tile(target_orientation, (4, 1))

    batch_result = solver.solve_batch(batch_positions, batch_orientations)

    print(f"  Batch size: {len(batch_positions)}")
    print(f"  Successful: {np.sum(batch_result.success_mask)}")
    print(f"  Solutions per pose: {batch_result.num_solutions}")

    # Estimate workspace
    print("\nEstimating workspace bounds...")
    x_range, y_range, z_range = solver.estimate_workspace_bounds(n_samples=5000)

    return solver


def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description="ARM Reachability Examples")
    parser.add_argument(
        '--urdf', '-u',
        type=str,
        required=True,
        help='Path to URDF file'
    )
    parser.add_argument(
        '--example', '-e',
        type=int,
        default=0,
        help='Example number to run (0 for all, 1-6 for specific)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        print(f"Error: URDF file not found: {args.urdf}")
        sys.exit(1)

    # Create output directories
    os.makedirs("./output_basic", exist_ok=True)
    os.makedirs("./output_custom", exist_ok=True)
    os.makedirs("./output_spherical", exist_ok=True)
    os.makedirs("./output_viz", exist_ok=True)

    examples = {
        1: ("Basic Analysis", example_basic_analysis),
        2: ("Custom Configuration", example_custom_config),
        3: ("Spherical Orientations", example_spherical_orientations),
        4: ("Visualization", example_visualization),
        5: ("URDF Parsing", example_urdf_parsing),
        6: ("IK Solver", example_ik_solver),
    }

    if args.example == 0:
        # Run all examples
        for num, (name, func) in examples.items():
            try:
                func(args.urdf)
            except Exception as e:
                print(f"\n[Error in {name}]: {e}")
            print()
    else:
        # Run specific example
        if args.example in examples:
            name, func = examples[args.example]
            func(args.urdf)
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")


if __name__ == '__main__':
    main()
