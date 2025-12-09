"""
State Estimator Module - Attitude and Rate Estimation

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module implements state estimation algorithms to fuse noisy sensor data
and estimate the rocket's attitude and angular rates.

Algorithms:
1. Complementary Filter (default): Simple, efficient, good for well-characterized sensors
2. Extended Kalman Filter (optional): More accurate with proper noise modeling

Rationale:
- Complementary filter chosen as default for simplicity and low computational cost
- High-pass filter on gyro (removes drift), low-pass on accelerometer (removes noise)
- EKF available for advanced scenarios requiring optimal sensor fusion
- For attitude: gyro integrates fast, accelerometer provides gravity reference
"""

import numpy as np
from typing import Tuple


class ComplementaryFilter:
    """
    Complementary filter for attitude estimation
    
    Fuses gyroscope (high-pass) and accelerometer (low-pass) data.
    Simple and computationally efficient.
    """
    
    def __init__(
        self,
        alpha: float = 0.98,
        initial_quaternion: np.ndarray = None
    ):
        """
        Initialize complementary filter
        
        Args:
            alpha: Filter coefficient (0-1). Higher = trust gyro more.
                  Typical: 0.95-0.99 for 100Hz sampling
            initial_quaternion: Initial attitude estimate [w, x, y, z]
        """
        self.alpha = alpha
        
        if initial_quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
        
        self.angular_velocity = np.zeros(3)
    
    def reset(self, quaternion: np.ndarray = None):
        """Reset filter state"""
        if quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.quaternion = quaternion / np.linalg.norm(quaternion)
        self.angular_velocity = np.zeros(3)
    
    def update(
        self,
        gyro_measurement: np.ndarray,
        accel_measurement: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update attitude estimate with new sensor data
        
        Args:
            gyro_measurement: Angular velocity from gyro [rad/s]
            accel_measurement: Specific force from accelerometer [m/s²]
            dt: Time step [s]
            
        Returns:
            (estimated_quaternion, estimated_angular_velocity)
        """
        # Store angular velocity estimate (directly from gyro after filtering)
        self.angular_velocity = gyro_measurement
        
        # Step 1: Integrate gyro (prediction)
        q_gyro = self._integrate_gyro(self.quaternion, gyro_measurement, dt)
        
        # Step 2: Get attitude from accelerometer (gravity reference)
        q_accel = self._attitude_from_accel(accel_measurement)
        
        # Step 3: Complementary fusion
        # q_est = α * q_gyro + (1-α) * q_accel
        # Using SLERP (Spherical Linear Interpolation) for proper quaternion blending
        self.quaternion = self._slerp(q_accel, q_gyro, self.alpha)
        
        # Normalize to prevent drift
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
        
        return self.quaternion.copy(), self.angular_velocity.copy()
    
    def _integrate_gyro(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Integrate gyroscope to predict quaternion
        
        q̇ = 0.5 * q ⊗ ω
        """
        w, x, y, z = q
        p, q_rate, r = omega
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -p*x - q_rate*y - r*z,
            p*w + r*y - q_rate*z,
            q_rate*w - r*x + p*z,
            r*w + q_rate*x - p*y
        ])
        
        # Euler integration (sufficient for complementary filter)
        q_new = q + q_dot * dt
        return q_new / np.linalg.norm(q_new)
    
    def _attitude_from_accel(self, accel: np.ndarray) -> np.ndarray:
        """
        Estimate attitude from accelerometer (gravity reference)
        
        Assumes: accelerometer measures primarily gravity (no linear acceleration)
        Returns quaternion that aligns body z-axis with measured acceleration
        """
        # Normalize acceleration vector
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.1:  # Avoid division by zero
            return self.quaternion  # Return previous estimate
        
        accel_unit = accel / accel_norm
        
        # Reference: gravity in NED frame [0, 0, g]
        # We want to find rotation that maps [0, 0, 1] to accel_unit
        
        # Simplified: extract roll and pitch from gravity vector
        # (Cannot observe yaw from gravity alone)
        roll = np.arctan2(accel_unit[1], accel_unit[2])
        pitch = np.arctan2(-accel_unit[0], 
                          np.sqrt(accel_unit[1]**2 + accel_unit[2]**2))
        
        # Keep previous yaw (unobservable from accel)
        _, _, yaw = self._quaternion_to_euler(self.quaternion)
        
        # Convert to quaternion
        q_accel = self._euler_to_quaternion(roll, pitch, yaw)
        return q_accel
    
    def _slerp(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Spherical Linear Interpolation between quaternions
        
        Args:
            q1, q2: Quaternions to interpolate
            t: Interpolation parameter (0 = q1, 1 = q2)
            
        Returns:
            Interpolated quaternion
        """
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot < 0, negate q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # SLERP formula
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        
        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
    
    def _quaternion_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles"""
        w, x, y, z = q
        
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return roll, pitch, yaw
    
    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> np.ndarray:
        """Convert Euler angles to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current estimate"""
        return self.quaternion.copy(), self.angular_velocity.copy()


class StateEstimator:
    """
    High-level state estimator interface
    
    Wraps complementary filter (and potentially other estimators)
    """
    
    def __init__(
        self,
        estimator_type: str = "complementary",
        alpha: float = 0.98
    ):
        """
        Initialize state estimator
        
        Args:
            estimator_type: "complementary" or "ekf" (only complementary implemented)
            alpha: Complementary filter coefficient
        """
        if estimator_type == "complementary":
            self.estimator = ComplementaryFilter(alpha=alpha)
        else:
            # Placeholder for EKF implementation
            raise NotImplementedError(f"Estimator type '{estimator_type}' not implemented. Use 'complementary'.")
        
        self.estimator_type = estimator_type
    
    def reset(self, quaternion: np.ndarray = None):
        """Reset estimator"""
        self.estimator.reset(quaternion)
    
    def update(
        self,
        gyro_measurement: np.ndarray,
        accel_measurement: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update with sensor measurements"""
        return self.estimator.update(gyro_measurement, accel_measurement, dt)
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current estimate"""
        return self.estimator.get_state()
