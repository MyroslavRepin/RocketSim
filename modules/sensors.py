"""
Sensors Module - IMU Simulation

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module simulates an Inertial Measurement Unit (IMU) with realistic
noise characteristics, including:
- Gyroscope: angular rate measurement with white noise and bias drift
- Accelerometer: specific force measurement with white noise
- Optional: GPS, airspeed sensor

Rationale:
- Realistic sensor modeling is essential for testing estimator robustness
- Noise models based on typical MEMS IMU specifications
- Bias drift simulates real sensor imperfections
"""

import numpy as np
from typing import Tuple


class IMU:
    """Simulates an Inertial Measurement Unit with realistic noise"""
    
    def __init__(
        self,
        gyro_noise_std: float = 0.001,      # rad/s (typical: 0.001-0.01)
        gyro_bias_std: float = 0.0001,       # rad/s (bias drift rate)
        accel_noise_std: float = 0.02,       # m/s² (typical: 0.01-0.1)
        accel_bias_std: float = 0.001,       # m/s² (bias drift rate)
        sample_rate: float = 100.0,          # Hz
        seed: int = None
    ):
        """
        Initialize IMU sensor model
        
        Args:
            gyro_noise_std: Gyroscope white noise standard deviation [rad/s]
            gyro_bias_std: Gyroscope bias random walk [rad/s/√Hz]
            accel_noise_std: Accelerometer white noise std [m/s²]
            accel_bias_std: Accelerometer bias random walk [m/s²/√Hz]
            sample_rate: Sensor sampling frequency [Hz]
            seed: Random seed for reproducibility
        """
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_std = gyro_bias_std
        self.accel_noise_std = accel_noise_std
        self.accel_bias_std = accel_bias_std
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize biases
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # For time-based sampling
        self.last_sample_time = 0.0
        self.accumulated_time = 0.0
    
    def reset(self):
        """Reset sensor biases and timing"""
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.last_sample_time = 0.0
        self.accumulated_time = 0.0
    
    def measure_gyro(self, true_angular_velocity: np.ndarray) -> np.ndarray:
        """
        Simulate gyroscope measurement
        
        Model: ω_meas = ω_true + bias + noise
        
        Args:
            true_angular_velocity: True angular velocity [p, q, r] rad/s
            
        Returns:
            Measured angular velocity with noise and bias
        """
        # Update bias with random walk
        self.gyro_bias += np.random.normal(0, self.gyro_bias_std * np.sqrt(self.dt), 3)
        
        # Add white noise
        noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        # Measured = true + bias + noise
        measured = true_angular_velocity + self.gyro_bias + noise
        
        return measured
    
    def measure_accelerometer(
        self,
        true_acceleration: np.ndarray,
        gravity_body: np.ndarray
    ) -> np.ndarray:
        """
        Simulate accelerometer measurement
        
        Accelerometer measures specific force: a_meas = a_true - g_body + bias + noise
        (Note: accelerometer cannot distinguish between acceleration and gravity)
        
        Args:
            true_acceleration: True linear acceleration in body frame [m/s²]
            gravity_body: Gravity vector in body frame [m/s²]
            
        Returns:
            Measured specific force
        """
        # Update bias with random walk
        self.accel_bias += np.random.normal(0, self.accel_bias_std * np.sqrt(self.dt), 3)
        
        # Add white noise
        noise = np.random.normal(0, self.accel_noise_std, 3)
        
        # Specific force = acceleration - gravity (what accelerometer sees)
        specific_force = true_acceleration - gravity_body
        
        # Measured = specific_force + bias + noise
        measured = specific_force + self.accel_bias + noise
        
        return measured
    
    def should_sample(self, current_time: float) -> bool:
        """
        Check if sensor should produce a new sample
        
        Args:
            current_time: Current simulation time [s]
            
        Returns:
            True if sensor should sample
        """
        self.accumulated_time = current_time
        
        if current_time - self.last_sample_time >= self.dt:
            self.last_sample_time = current_time
            return True
        return False
    
    def get_sample_period(self) -> float:
        """Return sensor sampling period [s]"""
        return self.dt


class SimplifiedSensors:
    """
    Simplified sensor suite for rocket simulation
    
    Includes:
    - IMU (gyro + accelerometer)
    - Gravity model (for accelerometer simulation)
    """
    
    def __init__(
        self,
        gyro_noise_std: float = 0.001,
        accel_noise_std: float = 0.02,
        sample_rate: float = 100.0,
        seed: int = None
    ):
        """Initialize sensor suite"""
        self.imu = IMU(
            gyro_noise_std=gyro_noise_std,
            accel_noise_std=accel_noise_std,
            sample_rate=sample_rate,
            seed=seed
        )
        
        # Earth's gravity (NED frame)
        self.gravity_ned = np.array([0.0, 0.0, 9.81])  # m/s²
    
    def reset(self):
        """Reset all sensors"""
        self.imu.reset()
    
    def measure(
        self,
        angular_velocity: np.ndarray,
        rotation_matrix: np.ndarray,
        linear_acceleration: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce sensor measurements
        
        Args:
            angular_velocity: True angular velocity [p, q, r] rad/s
            rotation_matrix: Rotation from body to NED frame
            linear_acceleration: Linear acceleration in body frame [m/s²]
                                If None, assumes hovering (zero acceleration)
            
        Returns:
            (gyro_measurement, accel_measurement)
        """
        # Gyroscope measurement
        gyro_meas = self.imu.measure_gyro(angular_velocity)
        
        # Transform gravity to body frame
        # R^T transforms from NED to body
        gravity_body = rotation_matrix.T @ self.gravity_ned
        
        # Linear acceleration (zero if not provided - hovering assumption)
        if linear_acceleration is None:
            linear_acceleration = np.zeros(3)
        
        # Accelerometer measurement
        accel_meas = self.imu.measure_accelerometer(linear_acceleration, gravity_body)
        
        return gyro_meas, accel_meas
    
    def should_sample(self, current_time: float) -> bool:
        """Check if sensors should sample"""
        return self.imu.should_sample(current_time)
    
    def get_sample_period(self) -> float:
        """Return sensor sampling period"""
        return self.imu.get_sample_period()
