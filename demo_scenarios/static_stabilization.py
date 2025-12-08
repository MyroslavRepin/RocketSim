"""
Demo Scenario: Static Stabilization

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This scenario demonstrates basic stabilization where the rocket maintains
zero orientation (pointing straight up) from various initial conditions.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import RocketSimulation, SimulationConfig


def run_static_stabilization():
    """
    Run static stabilization scenario
    
    Initial condition: 30 degrees roll, 20 degrees pitch
    Target: Zero orientation
    """
    print("\n" + "="*70)
    print("Static Stabilization Scenario")
    print("="*70)
    print("\n⚠️  Educational simulation only - not for real aerospace use\n")
    
    # Configuration
    config = SimulationConfig(
        dt=0.01,
        duration=30.0
    )
    
    # Create simulation
    sim = RocketSimulation(config)
    
    # Set initial conditions (30 deg roll, 20 deg pitch)
    initial_euler = np.radians([30.0, 20.0, 0.0])
    initial_quaternion = sim.physics.euler_to_quaternion(*initial_euler)
    sim.reset(initial_quaternion=initial_quaternion)
    
    # Set target: zero orientation
    sim.set_target_attitude(euler=np.zeros(3))
    
    print("Initial conditions:")
    print(f"  Roll:  {initial_euler[0]:.1f} deg")
    print(f"  Pitch: {initial_euler[1]:.1f} deg")
    print(f"  Yaw:   {initial_euler[2]:.1f} deg")
    print(f"\nTarget: Zero orientation")
    print(f"Duration: {config.duration}s\n")
    
    # Run simulation
    history = sim.run(verbose=True)
    
    # Display results
    final_euler = history['euler_angles'][-1]
    print(f"\nFinal attitude:")
    print(f"  Roll:  {np.degrees(final_euler[0]):.2f} deg")
    print(f"  Pitch: {np.degrees(final_euler[1]):.2f} deg")
    print(f"  Yaw:   {np.degrees(final_euler[2]):.2f} deg")
    
    # Plot results
    print("\nGenerating plots...")
    save_path = Path(__file__).parent / "static_stabilization_results.png"
    sim.plot_results(save_path=str(save_path))
    
    print(f"\n✓ Results saved to: {save_path}")
    print("\nRemember: This is educational only. Not for real rockets!\n")
    
    return sim, history


if __name__ == "__main__":
    run_static_stabilization()
