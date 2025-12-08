"""
Demo Scenario: Guidance Tracking

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This scenario demonstrates the guidance system tracking a moving target point.
The rocket adjusts its orientation to point toward the target as it moves
in a circular pattern.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import RocketSimulation, SimulationConfig


def run_guidance_tracking():
    """
    Run guidance tracking scenario
    
    Initial condition: Zero orientation
    Target: Moving point in circular pattern
    Objective: Point rocket toward target
    """
    print("\n" + "="*70)
    print("Guidance Tracking Scenario")
    print("="*70)
    print("\n⚠️  Educational simulation only - not for real aerospace use\n")
    
    # Configuration
    config = SimulationConfig(
        dt=0.01,
        duration=40.0
    )
    
    # Create simulation
    sim = RocketSimulation(config)
    
    # Set initial conditions
    sim.reset()
    
    print("Initial conditions: Zero orientation")
    print("Target: Moving point in circular pattern")
    print("  Radius: 50 m")
    print("  Angular velocity: 0.2 rad/s")
    print(f"\nDuration: {config.duration}s\n")
    
    # Run simulation with guidance updates
    steps = int(config.duration / config.dt)
    
    print(f"Running simulation for {config.duration}s ({steps} steps)...")
    
    for i in range(steps):
        # Update target position (circular motion)
        t = sim.time
        radius = 50.0
        omega = 0.2  # rad/s
        
        target_position = np.array([
            radius * np.cos(omega * t),
            radius * np.sin(omega * t),
            10.0  # Constant altitude
        ])
        
        sim.set_point_tracking(target_position)
        
        # Execute simulation step
        sim.step()
        
        # Progress indicator
        if i % (steps // 10) == 0:
            progress = 100 * i / steps
            print(f"  Progress: {progress:.0f}%")
    
    print("Simulation complete!")
    
    # Safety summary
    if sim.safety:
        sim.safety.print_summary()
    
    # Get history
    history = sim.get_history()
    
    # Display results
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    
    # Final state
    final_euler = history['euler_angles'][-1]
    print(f"\nFinal attitude:")
    print(f"  Roll:  {np.degrees(final_euler[0]):.2f} deg")
    print(f"  Pitch: {np.degrees(final_euler[1]):.2f} deg")
    print(f"  Yaw:   {np.degrees(final_euler[2]):.2f} deg")
    
    # Average tracking error
    # Compare actual orientation with desired (pointing toward target)
    pointing_errors = []
    for i in range(len(history['time'])):
        t = history['time'][i]
        radius = 50.0
        omega = 0.2
        
        # Target position
        target_pos = np.array([
            radius * np.cos(omega * t),
            radius * np.sin(omega * t),
            10.0
        ])
        
        # Desired direction (unit vector toward target)
        desired_dir = target_pos / np.linalg.norm(target_pos)
        
        # Actual direction (rocket's x-axis in NED frame)
        q = history['quaternion'][i]
        R = sim.physics.rotation_matrix(q)
        actual_dir = R @ np.array([1.0, 0.0, 0.0])
        
        # Angular error
        dot_product = np.dot(desired_dir, actual_dir)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        pointing_errors.append(angle_error)
    
    pointing_errors = np.array(pointing_errors)
    mean_error = np.mean(pointing_errors)
    max_error = np.max(pointing_errors)
    
    print(f"\nPointing accuracy:")
    print(f"  Mean error: {np.degrees(mean_error):.2f} deg")
    print(f"  Max error:  {np.degrees(max_error):.2f} deg")
    
    # Plot results
    print("\nGenerating plots...")
    save_path = Path(__file__).parent / "guidance_tracking_results.png"
    sim.plot_results(save_path=str(save_path))
    
    print(f"\n✓ Results saved to: {save_path}")
    print("\nRemember: This is educational only. Not for real rockets!\n")
    
    return sim, history


if __name__ == "__main__":
    run_guidance_tracking()
