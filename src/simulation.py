"""
RocketSim - Main Simulation Class

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.
Do not use for actual rocket construction, testing, or operation.

This module integrates all subsystems into a complete rocket stabilization simulator.
"""

import numpy as np
import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from physics import RocketPhysics
from sensors import SimplifiedSensors
from estimator import StateEstimator
from guidance import GuidanceSystem
from controllers import DualLoopController
from control_allocation import ControlAllocator
from aerodynamics import AerodynamicModel
from safety import SafetyMonitor
from visualization import Visualizer

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    dt: float = 0.01                    # Time step [s]
    duration: float = 30.0              # Simulation duration [s]
    
    # Physics
    inertia: np.ndarray = None
    
    # Sensors
    gyro_noise_std: float = 0.001
    accel_noise_std: float = 0.02
    
    # Estimator
    estimator_type: str = "complementary"
    estimator_alpha: float = 0.98
    
    # Controller
    outer_kp: np.ndarray = None
    outer_kd: np.ndarray = None
    inner_kp: np.ndarray = None
    inner_ki: np.ndarray = None
    max_rate: float = 2.0
    max_torque: float = 10.0
    
    # Aerodynamics
    velocity: float = 50.0
    altitude: float = 0.0
    
    # Safety
    enable_safety: bool = True
    max_angular_rate: float = 5.0


class RocketSimulation:
    """
    Main rocket stabilization simulator
    
    Integrates all subsystems: physics, sensors, estimation, guidance,
    control, control allocation, aerodynamics, safety, and visualization.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize rocket simulation
        
        Args:
            config: Simulation configuration. If None, uses defaults.
        """
        if config is None:
            config = SimulationConfig()
        
        self.config = config
        self.dt = config.dt
        self.duration = config.duration
        
        # Initialize subsystems
        self.physics = RocketPhysics(inertia=config.inertia)
        self.sensors = SimplifiedSensors(
            gyro_noise_std=config.gyro_noise_std,
            accel_noise_std=config.accel_noise_std,
            sample_rate=1.0/config.dt
        )
        self.estimator = StateEstimator(
            estimator_type=config.estimator_type,
            alpha=config.estimator_alpha
        )
        self.guidance = GuidanceSystem()
        self.controller = DualLoopController(
            outer_kp=config.outer_kp,
            outer_kd=config.outer_kd,
            inner_kp=config.inner_kp,
            inner_ki=config.inner_ki,
            max_rate=config.max_rate,
            max_torque=config.max_torque
        )
        self.control_allocator = ControlAllocator(max_deflection=0.35, max_rate=3.0, dt=config.dt)
        self.aerodynamics = AerodynamicModel(
            velocity=config.velocity,
            altitude=config.altitude
        )
        
        if config.enable_safety:
            self.safety = SafetyMonitor(
                max_angular_rate=config.max_angular_rate,
                damping_threshold=config.max_angular_rate * 0.8
            )
        else:
            self.safety = None
        
        self.visualizer = Visualizer()
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # Data logging
        self.history = {
            'time': [],
            'quaternion': [],
            'angular_velocity': [],
            'euler_angles': [],
            'estimated_quaternion': [],
            'estimated_angular_velocity': [],
            'target_quaternion': [],
            'control_torque': [],
            'deflections': [],
            'gyro_meas': [],
            'accel_meas': []
        }
    
    def reset(
        self,
        initial_quaternion: Optional[np.ndarray] = None,
        initial_angular_velocity: Optional[np.ndarray] = None
    ):
        """
        Reset simulation to initial conditions
        
        Args:
            initial_quaternion: Initial attitude [w, x, y, z]
            initial_angular_velocity: Initial rates [p, q, r] rad/s
        """
        self.time = 0.0
        self.step_count = 0
        
        # Reset all subsystems
        self.physics.reset(initial_quaternion, initial_angular_velocity)
        self.sensors.reset()
        self.estimator.reset(initial_quaternion)
        self.controller.reset()
        self.control_allocator.reset()
        
        if self.safety:
            self.safety.reset()
        
        # Clear history
        for key in self.history:
            self.history[key].clear()
    
    def set_target_attitude(self, quaternion: Optional[np.ndarray] = None, euler: Optional[np.ndarray] = None):
        """Set target attitude for guidance"""
        self.guidance.set_attitude_target(quaternion=quaternion, euler=euler)
    
    def set_point_tracking(self, target_position: np.ndarray):
        """Set point tracking mode"""
        self.guidance.set_point_tracking_target(target_position)
    
    def step(self, external_torque: Optional[np.ndarray] = None) -> Dict:
        """
        Execute one simulation step
        
        Args:
            external_torque: External disturbance torque [N·m]
            
        Returns:
            Dictionary with current state information
        """
        # Get true state from physics
        true_quaternion, true_angular_velocity = self.physics.get_state()
        rotation_matrix = self.physics.rotation_matrix()
        
        # Sensor measurements
        gyro_meas, accel_meas = self.sensors.measure(
            true_angular_velocity,
            rotation_matrix
        )
        
        # State estimation
        est_quaternion, est_angular_velocity = self.estimator.update(
            gyro_meas,
            accel_meas,
            self.dt
        )
        
        # Guidance: get target attitude
        target_quaternion = self.guidance.update()
        
        # Control: compute control torque
        control_torque, desired_rates = self.controller.update(
            est_quaternion,
            target_quaternion,
            est_angular_velocity,
            self.dt
        )
        
        # Safety monitoring
        damping_mode = False
        if self.safety:
            # Calculate attitude error magnitude
            attitude_error_vec = self.controller._quaternion_error(est_quaternion, target_quaternion)
            attitude_error = np.linalg.norm(attitude_error_vec)
            
            is_safe, damping_mode = self.safety.check_state(
                self.time,
                true_angular_velocity,
                attitude_error,
                control_torque
            )
            
            # Apply damping if needed
            if damping_mode:
                damping_torque = self.safety.apply_damping(true_angular_velocity)
                control_torque = damping_torque
        
        # Control allocation: torque to elevon deflections
        deflections = self.control_allocator.allocate(control_torque)
        
        # Aerodynamics: deflections to actual torque
        aero_torque = self.aerodynamics.compute_torque(deflections)
        
        # Total torque (aero + external disturbances)
        total_torque = aero_torque
        if external_torque is not None:
            total_torque = total_torque + external_torque
        
        # Physics integration
        self.physics.step(total_torque, self.dt)
        
        # Logging
        euler_angles = self.physics.quaternion_to_euler()
        self.history['time'].append(self.time)
        self.history['quaternion'].append(true_quaternion.copy())
        self.history['angular_velocity'].append(true_angular_velocity.copy())
        self.history['euler_angles'].append(euler_angles.copy())
        self.history['estimated_quaternion'].append(est_quaternion.copy())
        self.history['estimated_angular_velocity'].append(est_angular_velocity.copy())
        self.history['target_quaternion'].append(target_quaternion.copy())
        self.history['control_torque'].append(control_torque.copy())
        self.history['deflections'].append(deflections.copy())
        self.history['gyro_meas'].append(gyro_meas.copy())
        self.history['accel_meas'].append(accel_meas.copy())
        
        # Update time
        self.time += self.dt
        self.step_count += 1
        
        # Return current state
        return {
            'time': self.time,
            'quaternion': true_quaternion,
            'angular_velocity': true_angular_velocity,
            'euler_angles': euler_angles,
            'control_torque': control_torque,
            'deflections': deflections,
            'damping_mode': damping_mode
        }
    
    def run(
        self,
        external_torque_func=None,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete simulation
        
        Args:
            external_torque_func: Function(time) -> torque [N·m] for disturbances
            verbose: Print progress
            
        Returns:
            History dictionary with all logged data
        """
        steps = int(self.duration / self.dt)
        
        if verbose:
            print(f"Running simulation for {self.duration}s ({steps} steps)...")
        
        for i in range(steps):
            # Get external torque if function provided
            ext_torque = None
            if external_torque_func is not None:
                ext_torque = external_torque_func(self.time)
            
            # Execute step
            self.step(ext_torque)
            
            # Progress indicator
            if verbose and (i % (steps // 10) == 0):
                progress = 100 * i / steps
                print(f"  Progress: {progress:.0f}%")
        
        if verbose:
            print("Simulation complete!")
            
            # Print safety summary if enabled
            if self.safety:
                self.safety.print_summary()
        
        return self.get_history()
    
    def get_history(self) -> Dict:
        """
        Get simulation history as numpy arrays
        
        Returns:
            Dictionary with time series data
        """
        return {
            'time': np.array(self.history['time']),
            'quaternion': np.array(self.history['quaternion']),
            'angular_velocity': np.array(self.history['angular_velocity']),
            'euler_angles': np.array(self.history['euler_angles']),
            'estimated_quaternion': np.array(self.history['estimated_quaternion']),
            'estimated_angular_velocity': np.array(self.history['estimated_angular_velocity']),
            'target_quaternion': np.array(self.history['target_quaternion']),
            'control_torque': np.array(self.history['control_torque']),
            'deflections': np.array(self.history['deflections']),
            'gyro_meas': np.array(self.history['gyro_meas']),
            'accel_meas': np.array(self.history['accel_meas'])
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot simulation results"""
        history = self.get_history()
        
        # Convert target quaternions to Euler angles
        target_euler = np.array([
            self.physics.quaternion_to_euler(q) for q in history['target_quaternion']
        ])
        
        self.visualizer.plot_simulation_results(
            history['time'],
            history['euler_angles'],
            history['angular_velocity'],
            history['control_torque'],
            history['deflections'],
            target_euler=target_euler,
            save_path=save_path
        )
    
    def animate_orientation(self, save_path: Optional[str] = None):
        """Create 3D animation of rocket orientation"""
        history = self.get_history()
        self.visualizer.animate_3d_orientation(
            history['quaternion'],
            history['time'],
            save_path=save_path
        )
