"""
Visualization Module - Plotting and Animation

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module provides visualization tools for simulation results:
- Time history plots (roll/pitch/yaw, rates, control commands)
- 3D rocket orientation animation
- Error plots for analysis

Uses matplotlib for 2D plots and 3D visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional


class Visualizer:
    """Visualization tools for simulation results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.fig = None
        self.axes = None
    
    def plot_simulation_results(
        self,
        time: np.ndarray,
        euler_angles: np.ndarray,
        angular_rates: np.ndarray,
        control_torques: np.ndarray,
        deflections: np.ndarray,
        target_euler: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive simulation results
        
        Args:
            time: Time array [s]
            euler_angles: Euler angles [N x 3] (roll, pitch, yaw) [rad]
            angular_rates: Angular rates [N x 3] (p, q, r) [rad/s]
            control_torques: Control torques [N x 3] [N·m]
            deflections: Elevon deflections [N x 4] [rad]
            target_euler: Target Euler angles [N x 3] (optional)
            save_path: Path to save figure (if None, displays interactively)
        """
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        fig.suptitle('Rocket Stabilization Simulation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Euler Angles
        ax = axes[0]
        ax.plot(time, np.degrees(euler_angles[:, 0]), 'r-', label='Roll', linewidth=2)
        ax.plot(time, np.degrees(euler_angles[:, 1]), 'g-', label='Pitch', linewidth=2)
        ax.plot(time, np.degrees(euler_angles[:, 2]), 'b-', label='Yaw', linewidth=2)
        
        if target_euler is not None:
            ax.plot(time, np.degrees(target_euler[:, 0]), 'r--', alpha=0.5, label='Roll Target')
            ax.plot(time, np.degrees(target_euler[:, 1]), 'g--', alpha=0.5, label='Pitch Target')
            ax.plot(time, np.degrees(target_euler[:, 2]), 'b--', alpha=0.5, label='Yaw Target')
        
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Attitude (Roll, Pitch, Yaw)')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Angular Rates
        ax = axes[1]
        ax.plot(time, angular_rates[:, 0], 'r-', label='p (roll rate)', linewidth=2)
        ax.plot(time, angular_rates[:, 1], 'g-', label='q (pitch rate)', linewidth=2)
        ax.plot(time, angular_rates[:, 2], 'b-', label='r (yaw rate)', linewidth=2)
        ax.set_ylabel('Rate [rad/s]')
        ax.set_title('Angular Rates')
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Control Torques
        ax = axes[2]
        ax.plot(time, control_torques[:, 0], 'r-', label='τx (roll)', linewidth=2)
        ax.plot(time, control_torques[:, 1], 'g-', label='τy (pitch)', linewidth=2)
        ax.plot(time, control_torques[:, 2], 'b-', label='τz (yaw)', linewidth=2)
        ax.set_ylabel('Torque [N·m]')
        ax.set_title('Control Torques')
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Elevon Deflections
        ax = axes[3]
        ax.plot(time, np.degrees(deflections[:, 0]), label='Elevon 1', linewidth=2)
        ax.plot(time, np.degrees(deflections[:, 1]), label='Elevon 2', linewidth=2)
        ax.plot(time, np.degrees(deflections[:, 2]), label='Elevon 3', linewidth=2)
        ax.plot(time, np.degrees(deflections[:, 3]), label='Elevon 4', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Deflection [deg]')
        ax.set_title('Elevon Deflections')
        ax.legend(loc='best', ncol=4)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_errors(
        self,
        time: np.ndarray,
        attitude_errors: np.ndarray,
        rate_errors: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot tracking errors
        
        Args:
            time: Time array [s]
            attitude_errors: Attitude errors [N x 3] [rad]
            rate_errors: Rate errors [N x 3] [rad/s]
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Tracking Errors', fontsize=16, fontweight='bold')
        
        # Attitude errors
        ax = axes[0]
        ax.plot(time, np.degrees(attitude_errors[:, 0]), 'r-', label='Roll Error', linewidth=2)
        ax.plot(time, np.degrees(attitude_errors[:, 1]), 'g-', label='Pitch Error', linewidth=2)
        ax.plot(time, np.degrees(attitude_errors[:, 2]), 'b-', label='Yaw Error', linewidth=2)
        ax.set_ylabel('Error [deg]')
        ax.set_title('Attitude Errors')
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        
        # Rate errors
        ax = axes[1]
        ax.plot(time, rate_errors[:, 0], 'r-', label='p Error', linewidth=2)
        ax.plot(time, rate_errors[:, 1], 'g-', label='q Error', linewidth=2)
        ax.plot(time, rate_errors[:, 2], 'b-', label='r Error', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Error [rad/s]')
        ax.set_title('Rate Errors')
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error plot saved to: {save_path}")
        else:
            plt.show()
    
    def animate_3d_orientation(
        self,
        quaternions: np.ndarray,
        time: np.ndarray,
        save_path: Optional[str] = None,
        interval: int = 50
    ):
        """
        Create 3D animation of rocket orientation
        
        Args:
            quaternions: Quaternion history [N x 4]
            time: Time array [s]
            save_path: Path to save animation (if None, displays interactively)
            interval: Animation interval [ms]
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize rocket body axes
        rocket_length = 2.0
        body_axes = np.array([
            [rocket_length, 0, 0],  # x-axis (longitudinal)
            [0, 0.3, 0],            # y-axis
            [0, 0, 0.3]             # z-axis
        ])
        
        # Lines for rocket body axes
        line_x, = ax.plot([], [], [], 'r-', linewidth=3, label='X (Roll)')
        line_y, = ax.plot([], [], [], 'g-', linewidth=3, label='Y (Pitch)')
        line_z, = ax.plot([], [], [], 'b-', linewidth=3, label='Z (Yaw)')
        
        # Reference frame (NED)
        ax.plot([0, 1], [0, 0], [0, 0], 'k--', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 1], [0, 0], 'k--', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 0], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Labels
        ax.text(1.1, 0, 0, 'North', fontsize=10)
        ax.text(0, 1.1, 0, 'East', fontsize=10)
        ax.text(0, 0, 1.1, 'Down', fontsize=10)
        
        # Set axis properties
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X (North)')
        ax.set_ylabel('Y (East)')
        ax.set_zlabel('Z (Down)')
        ax.set_title('Rocket Orientation (3D)')
        ax.legend()
        
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            """Initialize animation"""
            line_x.set_data([], [])
            line_x.set_3d_properties([])
            line_y.set_data([], [])
            line_y.set_3d_properties([])
            line_z.set_data([], [])
            line_z.set_3d_properties([])
            time_text.set_text('')
            return line_x, line_y, line_z, time_text
        
        def update(frame):
            """Update animation frame"""
            # Get quaternion and rotation matrix
            q = quaternions[frame]
            R = self._quaternion_to_rotation_matrix(q)
            
            # Rotate body axes
            rotated_axes = (R @ body_axes.T).T
            
            # Update lines
            line_x.set_data([0, rotated_axes[0, 0]], [0, rotated_axes[0, 1]])
            line_x.set_3d_properties([0, rotated_axes[0, 2]])
            
            line_y.set_data([0, rotated_axes[1, 0]], [0, rotated_axes[1, 1]])
            line_y.set_3d_properties([0, rotated_axes[1, 2]])
            
            line_z.set_data([0, rotated_axes[2, 0]], [0, rotated_axes[2, 1]])
            line_z.set_3d_properties([0, rotated_axes[2, 2]])
            
            # Update time text
            time_text.set_text(f'Time: {time[frame]:.2f} s')
            
            return line_x, line_y, line_z, time_text
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(quaternions),
            interval=interval,
            blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved to: {save_path}")
        else:
            plt.show()
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        
        return R
