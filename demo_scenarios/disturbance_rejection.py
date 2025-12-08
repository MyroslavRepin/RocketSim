"""
Demo Scenario: Disturbance Rejection

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This scenario demonstrates the controller's ability to reject external
disturbances. A torque impulse is applied at t=5s, and the controller
must recover and maintain attitude.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation import RocketSimulation, SimulationConfig


def run_disturbance_rejection():
    """
    Run disturbance rejection scenario
    
    Initial condition: Zero orientation
    Disturbance: Torque impulse at t=5s for 0.5s
    Target: Maintain zero orientation
    """
    print("\n" + "="*70)
    print("Disturbance Rejection Scenario")
    print("="*70)
    print("\n⚠️  Educational simulation only - not for real aerospace use\n")
    
    # Configuration
    config = SimulationConfig(
        dt=0.01,
        duration=20.0
    )
    
    # Create simulation
    sim = RocketSimulation(config)
    
    # Set initial conditions (zero orientation)
    sim.reset()
    
    # Set target: zero orientation
    sim.set_target_attitude(euler=np.zeros(3))
    
    print("Initial conditions: Zero orientation")
    print("Target: Maintain zero orientation")
    print("\nDisturbance:")
    print("  Time: 5.0s - 5.5s")
    print("  Torque: [2.0, 1.0, 0.5] N·m (roll, pitch, yaw)")
    print(f"\nDuration: {config.duration}s\n")
    
    # Define disturbance function
    def disturbance_torque(t):
        """Apply torque impulse between 5s and 5.5s"""
        if 5.0 <= t <= 5.5:
            return np.array([2.0, 1.0, 0.5])
        return np.zeros(3)
    
    # Run simulation
    history = sim.run(external_torque_func=disturbance_torque, verbose=True)
    
    # Display results
    # Find peak attitude excursion
    euler_angles = history['euler_angles']
    peak_euler = np.max(np.abs(euler_angles), axis=0)
    
    print(f"\nPeak attitude excursion:")
    print(f"  Roll:  {np.degrees(peak_euler[0]):.2f} deg")
    print(f"  Pitch: {np.degrees(peak_euler[1]):.2f} deg")
    print(f"  Yaw:   {np.degrees(peak_euler[2]):.2f} deg")
    
    # Final attitude (should have recovered)
    final_euler = history['euler_angles'][-1]
    print(f"\nFinal attitude (after recovery):")
    print(f"  Roll:  {np.degrees(final_euler[0]):.2f} deg")
    print(f"  Pitch: {np.degrees(final_euler[1]):.2f} deg")
    print(f"  Yaw:   {np.degrees(final_euler[2]):.2f} deg")
    
    # Plot results
    print("\nGenerating plots...")
    save_path = Path(__file__).parent / "disturbance_rejection_results.png"
    sim.plot_results(save_path=str(save_path))
    
    print(f"\n✓ Results saved to: {save_path}")
    print("\nRemember: This is educational only. Not for real rockets!\n")
    
    return sim, history


if __name__ == "__main__":
    run_disturbance_rejection()
