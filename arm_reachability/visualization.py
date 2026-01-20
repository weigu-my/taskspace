"""
Visualization Module for Reachability Analysis.

This module provides visualization of:
- Robot mesh model from URDF
- Reachability point cloud with:
  - Point SIZE representing manipulability
  - Point COLOR representing dexterity
"""

import os
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from .urdf_parser import RobotConfig, URDFParser
from .reachability import ReachabilityResult
from .utils import colormap_dexterity, size_from_manipulability


@dataclass
class MeshData:
    """Container for mesh data."""
    vertices: np.ndarray
    faces: np.ndarray
    link_name: str
    transform: np.ndarray  # 4x4 transformation matrix
    scale: Optional[np.ndarray] = None  # (3,) mesh scale


class URDFMeshLoader:
    """Load and transform meshes from URDF."""

    def __init__(self, urdf_path: str):
        """
        Initialize mesh loader.

        Args:
            urdf_path: Path to URDF file
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)
        self.package_paths: Dict[str, str] = {}

        # Try to auto-detect package paths
        self._detect_package_paths()

    def _detect_package_paths(self):
        """Auto-detect ROS package paths."""
        # Look for common package locations
        search_dirs = [
            self.urdf_dir,
            os.path.dirname(self.urdf_dir),
            os.path.join(os.path.dirname(self.urdf_dir), 'meshes'),
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Try to find package.xml
                package_xml = os.path.join(search_dir, 'package.xml')
                if os.path.exists(package_xml):
                    try:
                        tree = ET.parse(package_xml)
                        root = tree.getroot()
                        name_elem = root.find('name')
                        if name_elem is not None and name_elem.text:
                            self.package_paths[name_elem.text] = search_dir
                    except:
                        pass

    def resolve_mesh_path(self, mesh_filename: str) -> Optional[str]:
        """
        Resolve mesh filename to actual file path.

        Args:
            mesh_filename: Mesh filename from URDF (may use package://)

        Returns:
            Resolved file path or None if not found
        """
        if not mesh_filename:
            return None

        # Handle package:// URLs
        if mesh_filename.startswith('package://'):
            # Extract package name and relative path
            remainder = mesh_filename[len('package://'):]
            parts = remainder.split('/', 1)

            if len(parts) == 2:
                package_name, relative_path = parts

                # Check registered package paths
                if package_name in self.package_paths:
                    resolved = os.path.join(self.package_paths[package_name], relative_path)
                    if os.path.exists(resolved):
                        return resolved

                # Try common relative paths
                search_paths = [
                    os.path.join(self.urdf_dir, '..', relative_path),
                    os.path.join(self.urdf_dir, '..', '..', package_name, relative_path),
                    os.path.join(self.urdf_dir, relative_path),
                    os.path.join(self.urdf_dir, package_name, relative_path),
                    os.path.join(self.urdf_dir, '..', package_name, relative_path),
                    # Try searching in current directory and subdirectories
                    os.path.join(os.path.dirname(self.urdf_dir), package_name, relative_path),
                    os.path.join(os.path.dirname(os.path.dirname(self.urdf_dir)), package_name, relative_path),
                ]

                for path in search_paths:
                    resolved = os.path.abspath(path)
                    if os.path.exists(resolved):
                        return resolved
                
                # Last resort: try to find meshes directory near URDF
                # Look for meshes directory in parent directories
                current_dir = self.urdf_dir
                for _ in range(3):  # Search up to 3 levels up
                    meshes_dir = os.path.join(current_dir, 'meshes')
                    if os.path.exists(meshes_dir):
                        # Extract just the filename from relative_path
                        mesh_basename = os.path.basename(relative_path)
                        potential_path = os.path.join(meshes_dir, mesh_basename)
                        if os.path.exists(potential_path):
                            return potential_path
                    current_dir = os.path.dirname(current_dir)

            return None

        # Handle file:// URLs
        if mesh_filename.startswith('file://'):
            return mesh_filename[len('file://'):]

        # Handle relative paths
        if not os.path.isabs(mesh_filename):
            resolved = os.path.join(self.urdf_dir, mesh_filename)
            if os.path.exists(resolved):
                return resolved

            # Try parent directories
            for parent in ['..', '../..', '../../..']:
                resolved = os.path.join(self.urdf_dir, parent, mesh_filename)
                resolved = os.path.abspath(resolved)
                if os.path.exists(resolved):
                    return resolved

        return mesh_filename if os.path.exists(mesh_filename) else None

    def load_mesh(self, mesh_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load mesh from file.

        Args:
            mesh_path: Path to mesh file

        Returns:
            Tuple of (vertices, faces) or None if failed
        """
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, force='mesh')

            if isinstance(mesh, trimesh.Scene):
                # Combine all meshes in scene
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return None

            return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

        except Exception as e:
            print(f"[Warning] Failed to load mesh {mesh_path}: {e}")
            return None

    def load_robot_meshes(
        self,
        robot_config: RobotConfig,
        joint_positions: Optional[np.ndarray] = None
    ) -> List[MeshData]:
        """
        Load all robot meshes with transforms.

        Args:
            robot_config: Robot configuration
            joint_positions: Joint positions for FK (default: zero config)

        Returns:
            List of MeshData objects
        """
        meshes = []
        loaded_count = 0
        failed_count = 0

        # Get transforms for each link using FK
        link_transforms = self._compute_link_transforms(robot_config, joint_positions)

        for link in robot_config.links:
            # Try visual mesh first, then collision mesh
            mesh_filename = link.visual_mesh or link.collision_mesh
            mesh_scale = link.visual_scale if link.visual_mesh else link.collision_scale

            if not mesh_filename:
                continue

            mesh_path = self.resolve_mesh_path(mesh_filename)
            if not mesh_path:
                failed_count += 1
                if failed_count <= 5:  # Only print first 5 warnings to avoid spam
                    print(f"[Warning] Could not resolve mesh: {mesh_filename}")
                continue

            result = self.load_mesh(mesh_path)
            if result is None:
                failed_count += 1
                if failed_count <= 5:
                    print(f"[Warning] Failed to load mesh: {mesh_path}")
                continue

            vertices, faces = result
            loaded_count += 1

            # Get transform for this link
            transform = link_transforms.get(link.name, np.eye(4))

            # Apply visual origin if present
            if link.visual_origin is not None:
                visual_transform = self._origin_to_transform(link.visual_origin)
                transform = transform @ visual_transform

            meshes.append(MeshData(
                vertices=vertices,
                faces=faces,
                link_name=link.name,
                transform=transform,
                scale=np.array(mesh_scale, dtype=np.float32) if mesh_scale else None
            ))

        print(f"[MeshLoader] Loaded {loaded_count} meshes, {failed_count} failed")
        return meshes

    def _compute_link_transforms(
        self,
        robot_config: RobotConfig,
        joint_positions: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Compute FK transforms for all links."""
        if joint_positions is None:
            joint_positions = np.zeros(robot_config.num_dof)

        transforms = {}

        # Try using pinocchio for accurate FK
        try:
            import pinocchio as pin

            model = pin.buildModelFromUrdf(robot_config.urdf_path)
            data = model.createData()

            # Build full configuration
            q = np.zeros(model.nq)
            joint_idx = 0
            for i, name in enumerate(model.names[1:]):
                for j, joint in enumerate(robot_config.active_joints):
                    if joint.name == name and joint_idx < len(joint_positions):
                        q[i] = joint_positions[joint_idx]
                        joint_idx += 1
                        break

            # Compute FK
            pin.framesForwardKinematics(model, data, q)

            # Get transforms for each frame
            for i, frame in enumerate(model.frames):
                transforms[frame.name] = data.oMf[i].homogeneous.copy()

            return transforms

        except Exception as e:
            print(f"[Warning] Pinocchio FK failed: {e}")

        # Fallback: compute transforms from joint chain
        transforms[robot_config.base_link] = np.eye(4)

        # Build parent-child relationships
        parent_to_joint = {}
        for joint in robot_config.joints:
            parent_to_joint[joint.child_link] = joint

        # BFS to compute all transforms
        visited = {robot_config.base_link}
        queue = [robot_config.base_link]
        joint_idx = 0

        while queue:
            current = queue.pop(0)

            for joint in robot_config.joints:
                if joint.parent_link == current and joint.child_link not in visited:
                    # Compute joint transform
                    joint_transform = self._joint_to_transform(
                        joint,
                        joint_positions[joint_idx] if joint.type in ['revolute', 'continuous', 'prismatic'] and joint_idx < len(joint_positions) else 0
                    )

                    if joint.type in ['revolute', 'continuous', 'prismatic']:
                        joint_idx += 1

                    # Chain transforms
                    transforms[joint.child_link] = transforms[current] @ joint_transform

                    visited.add(joint.child_link)
                    queue.append(joint.child_link)

        return transforms

    def _joint_to_transform(self, joint, q: float) -> np.ndarray:
        """Convert joint info and position to 4x4 transform."""
        # Origin transform
        T = self._origin_to_transform(joint.origin_xyz + joint.origin_rpy)

        # Joint motion
        if joint.type == 'revolute' or joint.type == 'continuous':
            axis = np.array(joint.axis)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            R = self._axis_angle_to_matrix(axis, q)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T = T @ T_joint

        elif joint.type == 'prismatic':
            axis = np.array(joint.axis)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            T_joint = np.eye(4)
            T_joint[:3, 3] = axis * q
            T = T @ T_joint

        return T

    def _origin_to_transform(self, origin: List[float]) -> np.ndarray:
        """Convert origin (xyz + rpy) to 4x4 transform."""
        T = np.eye(4)

        if len(origin) >= 3:
            T[:3, 3] = origin[:3]

        if len(origin) >= 6:
            roll, pitch, yaw = origin[3:6]
            T[:3, :3] = self._rpy_to_matrix(roll, pitch, yaw)

        return T

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll-pitch-yaw to rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        return R

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to rotation matrix (Rodrigues formula)."""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R


class ReachabilityVisualizer:
    """
    Visualize reachability analysis results.

    Features:
    - Robot mesh model rendering
    - Reachability point cloud visualization
    - Point SIZE = manipulability
    - Point COLOR = dexterity
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        result: ReachabilityResult
    ):
        """
        Initialize visualizer.

        Args:
            robot_config: Robot configuration
            result: Reachability analysis result
        """
        self.robot_config = robot_config
        self.result = result
        self.mesh_loader = URDFMeshLoader(robot_config.urdf_path)

    def visualize_plotly(
        self,
        show_robot: bool = True,
        show_points: bool = True,
        point_size_range: Tuple[float, float] = (2.0, 15.0),
        robot_opacity: float = 0.5,
        robot_color: str = 'lightgray',
        save_html: Optional[str] = None
    ):
        """
        Create interactive 3D visualization using Plotly.

        Args:
            show_robot: Whether to show robot mesh
            show_points: Whether to show reachability points
            point_size_range: (min_size, max_size) for point scaling
            robot_opacity: Opacity of robot mesh
            robot_color: Color of robot mesh
            save_html: Path to save HTML file
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add robot meshes
        if show_robot:
            print(f"[Visualization] Loading robot meshes from {self.robot_config.urdf_path}...")
            meshes = self.mesh_loader.load_robot_meshes(self.robot_config)
            
            if len(meshes) == 0:
                print(f"[Warning] No meshes loaded! Robot model will not be visible in visualization.")
                print(f"[Warning] Check that mesh files exist and paths are correct in URDF.")
            else:
                print(f"[Visualization] Adding {len(meshes)} mesh(es) to visualization...")

            for mesh_data in meshes:
                # Transform vertices
                vertices = mesh_data.vertices
                if mesh_data.scale is not None:
                    vertices = vertices * mesh_data.scale.reshape(1, 3)
                ones = np.ones((len(vertices), 1))
                vertices_h = np.hstack([vertices, ones])
                transformed = (mesh_data.transform @ vertices_h.T).T[:, :3]

                # Add mesh
                fig.add_trace(go.Mesh3d(
                    x=transformed[:, 0],
                    y=transformed[:, 1],
                    z=transformed[:, 2],
                    i=mesh_data.faces[:, 0],
                    j=mesh_data.faces[:, 1],
                    k=mesh_data.faces[:, 2],
                    color=robot_color,
                    opacity=robot_opacity,
                    name=mesh_data.link_name,
                    showlegend=False,
                    flatshading=True
                ))

        # Add reachability points
        if show_points and self.result.reachable_count > 0:
            # Get reachable points and their metrics
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]
            manipulability = self.result.manipulability[reachable_mask]

            # Compute colors (dexterity)
            colors = colormap_dexterity(dexterity)

            # Compute sizes (manipulability)
            sizes = size_from_manipulability(
                manipulability,
                min_size=point_size_range[0],
                max_size=point_size_range[1]
            )

            # Create hover text
            hover_text = [
                f"Position: ({x:.3f}, {y:.3f}, {z:.3f})<br>"
                f"Dexterity: {d}<br>"
                f"Manipulability: {m:.4f}"
                for (x, y, z), d, m in zip(points, dexterity, manipulability)
            ]

            # Convert colors to plotly format
            color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]

            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color_strings,
                    opacity=0.8,
                    line=dict(width=0)
                ),
                text=hover_text,
                hoverinfo='text',
                name='Reachable Points'
            ))

        # Configure layout
        fig.update_layout(
            title=dict(
                text=f"Reachability Analysis: {self.robot_config.name}",
                x=0.5
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Add color bar for dexterity
        if show_points and self.result.reachable_count > 0:
            # Create dummy trace for colorbar
            fig.add_trace(go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(
                    colorscale=[
                        [0, 'rgb(0,0,255)'],      # Blue
                        [0.25, 'rgb(0,255,255)'], # Cyan
                        [0.5, 'rgb(0,255,0)'],    # Green
                        [0.75, 'rgb(255,255,0)'], # Yellow
                        [1, 'rgb(255,0,0)']       # Red
                    ],
                    cmin=0,
                    cmax=int(np.max(self.result.dexterity)),
                    colorbar=dict(
                        title="Dexterity",
                        x=1.02,
                        thickness=20
                    ),
                    showscale=True
                ),
                showlegend=False
            ))

        # Show or save
        if save_html:
            fig.write_html(save_html)
            print(f"Saved visualization to {save_html}")
        else:
            fig.show()

        return fig

    def visualize_matplotlib(
        self,
        show_robot: bool = False,  # Matplotlib 3D doesn't handle meshes well
        point_size_range: Tuple[float, float] = (5.0, 100.0),
        elevation: float = 20,
        azimuth: float = 45,
        save_path: Optional[str] = None
    ):
        """
        Create static 3D visualization using Matplotlib.

        Args:
            show_robot: Whether to show robot skeleton
            point_size_range: (min_size, max_size) for point scaling
            elevation: View elevation angle
            azimuth: View azimuth angle
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Add reachability points
        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]
            manipulability = self.result.manipulability[reachable_mask]

            # Compute colors and sizes
            colors = colormap_dexterity(dexterity)
            sizes = size_from_manipulability(
                manipulability,
                min_size=point_size_range[0],
                max_size=point_size_range[1]
            )

            # Scatter plot
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=sizes,
                alpha=0.6,
                depthshade=True
            )

        # Add robot skeleton if requested
        if show_robot:
            self._draw_robot_skeleton(ax)

        # Configure view
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Reachability: {self.robot_config.name}\n'
                    f'Points: {self.result.reachable_count}/{self.result.total_points} '
                    f'({self.result.reachability_ratio*100:.1f}%)')

        # Add colorbar
        if self.result.reachable_count > 0:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.jet,
                norm=plt.Normalize(0, np.max(self.result.dexterity))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Dexterity')

        # Add legend for size
        if self.result.reachable_count > 0:
            ax.text2D(0.02, 0.98, "Size = Manipulability",
                     transform=ax.transAxes, fontsize=10,
                     verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        return fig, ax

    def _draw_robot_skeleton(self, ax):
        """Draw robot as skeleton (lines between joint origins)."""
        try:
            import pinocchio as pin

            model = pin.buildModelFromUrdf(self.robot_config.urdf_path)
            data = model.createData()

            q = np.zeros(model.nq)
            pin.framesForwardKinematics(model, data, q)

            # Draw links as lines
            for i in range(1, len(model.frames)):
                frame = model.frames[i]
                pos = data.oMf[i].translation

                # Find parent
                parent_id = frame.parent
                if parent_id > 0 and parent_id < len(data.oMf):
                    parent_pos = data.oMf[parent_id].translation

                    ax.plot3D(
                        [parent_pos[0], pos[0]],
                        [parent_pos[1], pos[1]],
                        [parent_pos[2], pos[2]],
                        'k-', linewidth=2, alpha=0.7
                    )

                # Draw joint position
                ax.scatter(pos[0], pos[1], pos[2],
                          c='black', s=30, marker='o')

        except Exception as e:
            print(f"[Warning] Could not draw robot skeleton: {e}")

    def create_summary_plot(self, save_path: Optional[str] = None):
        """
        Create summary visualization with multiple views.

        Args:
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))

        # Main 3D view
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self._plot_points_3d(ax1, elev=30, azim=45)
        ax1.set_title('3D View (Isometric)')

        # Top view (XY)
        ax2 = fig.add_subplot(2, 2, 2)
        self._plot_points_2d(ax2, 'x', 'y', 'Top View (XY)')

        # Front view (XZ)
        ax3 = fig.add_subplot(2, 2, 3)
        self._plot_points_2d(ax3, 'x', 'z', 'Front View (XZ)')

        # Side view (YZ)
        ax4 = fig.add_subplot(2, 2, 4)
        self._plot_points_2d(ax4, 'y', 'z', 'Side View (YZ)')

        plt.suptitle(f'Reachability Analysis: {self.robot_config.name}\n'
                    f'Reachable: {self.result.reachable_count}/{self.result.total_points} '
                    f'({self.result.reachability_ratio*100:.1f}%)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved summary to {save_path}")
        else:
            plt.show()

        return fig

    def _plot_points_3d(self, ax, elev=30, azim=45):
        """Helper to plot 3D points."""
        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]

            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=dexterity, cmap='jet', s=5, alpha=0.5
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def _plot_points_2d(self, ax, x_axis: str, y_axis: str, title: str):
        """Helper to plot 2D projection."""
        axis_map = {'x': 0, 'y': 1, 'z': 2}

        if self.result.reachable_count > 0:
            reachable_mask = self.result.reachable_mask
            points = self.result.grid_points[reachable_mask]
            dexterity = self.result.dexterity[reachable_mask]

            xi = axis_map[x_axis]
            yi = axis_map[y_axis]

            scatter = ax.scatter(
                points[:, xi], points[:, yi],
                c=dexterity, cmap='jet', s=3, alpha=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Dexterity')

        ax.set_xlabel(f'{x_axis.upper()} (m)')
        ax.set_ylabel(f'{y_axis.upper()} (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


def visualize_results(
    urdf_path: str,
    result: ReachabilityResult,
    output_dir: str = "./",
    show: bool = True
):
    """
    Convenience function to visualize results.

    Args:
        urdf_path: Path to URDF file
        result: ReachabilityResult
        output_dir: Directory to save visualizations
        show: Whether to display interactive visualization
    """
    parser = URDFParser(urdf_path)
    robot_config = parser.parse()

    visualizer = ReachabilityVisualizer(robot_config, result)

    # Save HTML visualization
    html_path = os.path.join(output_dir, "reachability_visualization.html")
    visualizer.visualize_plotly(
        show_robot=True,
        save_html=html_path
    )

    # Save summary plot
    summary_path = os.path.join(output_dir, "reachability_summary.png")
    visualizer.create_summary_plot(save_path=summary_path)

    if show:
        visualizer.visualize_plotly(show_robot=True)
