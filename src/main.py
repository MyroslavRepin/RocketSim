"""
RocketSim - Command Line Interface

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.
Do not use for actual rocket construction, testing, or operation.

This is the main entry point for running rocket stabilization simulations.
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simulation import RocketSimulation, SimulationConfig


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RocketSim - Educational Rocket Stabilization Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  SAFETY WARNING ⚠️
This simulator is for educational purposes ONLY.
Do not use for real aerospace applications.

Examples:
  python main.py                           # Run default stabilization
  python main.py --scenario disturbance    # Run disturbance rejection
  python main.py --duration 60 --dt 0.005  # Custom simulation parameters
  python main.py --plot-only results.png   # Plot results only
        """
    )
    
    # Simulation parameters
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Simulation duration in seconds (default: 30.0)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step in seconds (default: 0.01)'
    )
    
    # Scenario selection
    parser.add_argument(
        '--scenario',
        type=str,
        default='stabilization',
        choices=['stabilization', 'disturbance', 'guidance'],
        help='Simulation scenario (default: stabilization)'
    )
    
    # Initial conditions
    parser.add_argument(
        '--initial-roll',
        type=float,
        default=0.0,
        help='Initial roll angle in degrees (default: 0.0)'
    )
    parser.add_argument(
        '--initial-pitch',
        type=float,
        default=0.0,
        help='Initial pitch angle in degrees (default: 0.0)'
    )
    parser.add_argument(
        '--initial-yaw',
        type=float,
        default=0.0,
        help='Initial yaw angle in degrees (default: 0.0)'
    )
    
    # Visualization
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Display plots after simulation'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save plots to specified path'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Show 3D animation of rocket orientation'
    )
    parser.add_argument(
        '--save-animation',
        type=str,
        default=None,
        help='Save animation to specified path (.gif)'
    )
    
    # Output
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed simulation progress'
    )
    
    args = parser.parse_args()
    
    # Print safety warning
    print("\n" + "="*70)
    print("⚠️  EDUCATIONAL SIMULATION - NOT FOR REAL AEROSPACE USE  ⚠️")
    print("="*70 + "\n")
    
    # Create simulation configuration
    config = SimulationConfig(
        dt=args.dt,
        duration=args.duration
    )
    
    # Initialize simulation
    print(f"Initializing RocketSim...")
    print(f"  Scenario: {args.scenario}")
    print(f"  Duration: {args.duration}s")
    print(f"  Time step: {args.dt}s")
    print()
    
    sim = RocketSimulation(config)
    
    # Set initial conditions
    initial_euler = np.radians([args.initial_roll, args.initial_pitch, args.initial_yaw])
    initial_quaternion = sim.physics.euler_to_quaternion(*initial_euler)
    sim.reset(initial_quaternion=initial_quaternion)
    
    # Configure scenario
    if args.scenario == 'stabilization':
        # Static stabilization - maintain zero orientation
        sim.set_target_attitude(euler=np.array([0.0, 0.0, 0.0]))
        print("Scenario: Static stabilization (maintain zero orientation)")
        
        # Run simulation
        history = sim.run(verbose=args.verbose)
        
    elif args.scenario == 'disturbance':
        # Disturbance rejection - apply impulse torque
        sim.set_target_attitude(euler=np.array([0.0, 0.0, 0.0]))
        print("Scenario: Disturbance rejection (torque impulse at t=5s)")
        
        # Define disturbance function
        def disturbance_torque(t):
            if 5.0 <= t <= 5.5:
                return np.array([2.0, 1.0, 0.5])  # Torque impulse [N·m]
            return np.zeros(3)
        
        # Run simulation
        history = sim.run(external_torque_func=disturbance_torque, verbose=args.verbose)
        
    elif args.scenario == 'guidance':
        # Point tracking - rocket points toward moving target
        print("Scenario: Guidance (track moving target point)")
        
        # Define target position (moves in a circle)
        def guidance_update(t):
            radius = 50.0
            omega = 0.2
            target = np.array([
                radius * np.cos(omega * t),
                radius * np.sin(omega * t),
                10.0
            ])
            sim.set_point_tracking(target)
        
        # Run simulation with guidance updates
        steps = int(args.duration / args.dt)
        
        if args.verbose:
            print(f"Running simulation for {args.duration}s ({steps} steps)...")
        
        for i in range(steps):
            # Update target
            guidance_update(sim.time)
            
            # Execute step
            sim.step()
            
            # Progress
            if args.verbose and (i % (steps // 10) == 0):
                progress = 100 * i / steps
                print(f"  Progress: {progress:.0f}%")
        
        if args.verbose:
            print("Simulation complete!")
            if sim.safety:
                sim.safety.print_summary()
        
        history = sim.get_history()
    
    # Visualization
    if args.plot or args.save_plot:
        print("\nGenerating plots...")
        sim.plot_results(save_path=args.save_plot)
    
    if args.animate or args.save_animation:
        print("\nCreating 3D animation...")
        sim.animate_orientation(save_path=args.save_animation)
    
    # Summary statistics
    print("\n" + "="*70)
    print("Simulation Summary")
    print("="*70)
    
    # Final attitude error
    final_euler = history['euler_angles'][-1]
    print(f"Final attitude (deg): Roll={np.degrees(final_euler[0]):.2f}, " +
          f"Pitch={np.degrees(final_euler[1]):.2f}, " +
          f"Yaw={np.degrees(final_euler[2]):.2f}")
    
    # Final rates
    final_rates = history['angular_velocity'][-1]
    print(f"Final rates (rad/s): p={final_rates[0]:.4f}, " +
          f"q={final_rates[1]:.4f}, " +
          f"r={final_rates[2]:.4f}")
    
    # RMS errors
    target_euler = np.array([
        sim.physics.quaternion_to_euler(q) for q in history['target_quaternion']
    ])
    attitude_errors = history['euler_angles'] - target_euler
    rms_error = np.sqrt(np.mean(attitude_errors**2, axis=0))
    print(f"RMS attitude errors (deg): Roll={np.degrees(rms_error[0]):.2f}, " +
          f"Pitch={np.degrees(rms_error[1]):.2f}, " +
          f"Yaw={np.degrees(rms_error[2]):.2f}")
    
    print("="*70 + "\n")
    
    print("✓ Simulation completed successfully!")
    print("\nRemember: This is for educational use only. Not for real aerospace systems.")
    

if __name__ == "__main__":
    main()
