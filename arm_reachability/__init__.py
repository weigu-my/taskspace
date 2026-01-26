"""
机械臂可达性分析框架

基于 cuRobo 的通用机械臂可达性分析框架，支持 GPU 加速的多种子 IK 求解、
灵活度分析和可操作度计算。
"""

from .config import ReachabilityConfig, VoxelConfig, IKConfig
from .urdf_parser import URDFParser, CuroboConfigGenerator
from .ik_solver import MultiSeedIKSolver
from .reachability import ReachabilityAnalyzer
from .manipulability import ManipulabilityCalculator
from .visualization import ReachabilityVisualizer


def get_rviz_publisher():
    """
    获取 RViz 发布器（需要 ROS 2 环境）

    返回:
        (RVizReachabilityPublisher, RVizPublisherConfig) 元组
    """
    from .ros2_visualization import RVizReachabilityPublisher, RVizPublisherConfig
    return RVizReachabilityPublisher, RVizPublisherConfig


__version__ = "2.0.0"
__all__ = [
    "ReachabilityConfig",
    "VoxelConfig",
    "IKConfig",
    "URDFParser",
    "CuroboConfigGenerator",
    "MultiSeedIKSolver",
    "ReachabilityAnalyzer",
    "ManipulabilityCalculator",
    "ReachabilityVisualizer",
    "get_rviz_publisher",
]
