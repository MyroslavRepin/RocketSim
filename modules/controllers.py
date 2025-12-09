"""
Controllers Module - Dual-Loop Attitude Control

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module implements a dual-loop control architecture:
1. Outer loop: Attitude error → desired angular rates (PD controller)
2. Inner loop: Rate error → control torques (PI controller)

Rationale:
- PD for outer loop: No integral needed for orientation tracking (prevents windup)
- PI for inner loop: Integral removes steady-state rate errors
- Two-loop structure separates slow attitude dynamics from fast rate dynamics
- This improves stability and allows independent tuning of each loop
- Alternative LQR controller available but requires state-space model tuning

Design Notes:
- Quaternion-based attitude error avoids singularities
- Anti-windup included in inner loop integral term
- Derivative term filtered to reduce noise amplification
"""

import numpy as np
from typing import Tuple


class PDController:
    """
    PD (Proportional-Derivative) controller
    
    Used for outer loop: attitude error → desired angular rates
    """
    
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        """
        Initialize PD controller
        
        Args:
            kp: Proportional gains [3] for roll, pitch, yaw
            kd: Derivative gains [3]
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        
        self.prev_error = np.zeros(3)
        self.error_derivative = np.zeros(3)
    
    def reset(self):
        """Reset controller state"""
        self.prev_error = np.zeros(3)
        self.error_derivative = np.zeros(3)
    
    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate control output
        
        Args:
            error: Error signal [3]
            dt: Time step [s]
            
        Returns:
            Control output [3]
        """
        # Derivative term (filtered using simple first-order filter)
        alpha = 0.1  # Derivative filter coefficient
        error_derivative_raw = (error - self.prev_error) / dt if dt > 0 else np.zeros(3)
        self.error_derivative = alpha * error_derivative_raw + (1 - alpha) * self.error_derivative
        
        # PD control law
        output = self.kp * error + self.kd * self.error_derivative
        
        # Store for next iteration
        self.prev_error = error.copy()
        
        return output


class PIController:
    """
    PI (Proportional-Integral) controller
    
    Used for inner loop: rate error → control torques
    """
    
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        integral_limit: np.ndarray = None
    ):
        """
        Initialize PI controller
        
        Args:
            kp: Proportional gains [3]
            ki: Integral gains [3]
            integral_limit: Anti-windup limits for integral [3]
        """
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        
        if integral_limit is None:
            self.integral_limit = np.array([10.0, 10.0, 10.0])  # Default limits
        else:
            self.integral_limit = np.array(integral_limit)
        
        self.integral = np.zeros(3)
    
    def reset(self):
        """Reset controller state"""
        self.integral = np.zeros(3)
    
    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate control output
        
        Args:
            error: Error signal [3]
            dt: Time step [s]
            
        Returns:
            Control output [3]
        """
        # Integrate error
        self.integral += error * dt
        
        # Anti-windup: clamp integral
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # PI control law
        output = self.kp * error + self.ki * self.integral
        
        return output


class DualLoopController:
    """
    Dual-loop attitude controller
    
    Architecture:
    - Outer loop: Quaternion error → desired angular rates
    - Inner loop: Rate error → control torques
    """
    
    def __init__(
        self,
        outer_kp: np.ndarray = None,
        outer_kd: np.ndarray = None,
        inner_kp: np.ndarray = None,
        inner_ki: np.ndarray = None,
        max_rate: float = 2.0,
        max_torque: float = 10.0
    ):
        """
        Initialize dual-loop controller
        
        Args:
            outer_kp: Outer loop proportional gains [3]
            outer_kd: Outer loop derivative gains [3]
            inner_kp: Inner loop proportional gains [3]
            inner_ki: Inner loop integral gains [3]
            max_rate: Maximum commanded angular rate [rad/s]
            max_torque: Maximum control torque [N·m]
        """
        # Default gains (tuned for typical rocket)
        if outer_kp is None:
            outer_kp = np.array([2.0, 2.0, 2.0])
        if outer_kd is None:
            outer_kd = np.array([1.0, 1.0, 1.0])
        if inner_kp is None:
            inner_kp = np.array([0.5, 0.5, 0.5])
        if inner_ki is None:
            inner_ki = np.array([0.1, 0.1, 0.1])
        
        self.outer_loop = PDController(outer_kp, outer_kd)
        self.inner_loop = PIController(inner_kp, inner_ki)
        
        self.max_rate = max_rate
        self.max_torque = max_torque
        
        self.desired_rates = np.zeros(3)
        self.control_torque = np.zeros(3)
    
    def reset(self):
        """Reset both control loops"""
        self.outer_loop.reset()
        self.inner_loop.reset()
        self.desired_rates = np.zeros(3)
        self.control_torque = np.zeros(3)
    
    def update(
        self,
        current_quaternion: np.ndarray,
        target_quaternion: np.ndarray,
        current_rates: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update controller and compute control torque
        
        Args:
            current_quaternion: Current attitude [w, x, y, z]
            target_quaternion: Desired attitude [w, x, y, z]
            current_rates: Current angular rates [p, q, r] rad/s
            dt: Time step [s]
            
        Returns:
            (control_torque, desired_rates)
        """
        # Outer loop: attitude error → desired rates
        attitude_error = self._quaternion_error(current_quaternion, target_quaternion)
        self.desired_rates = self.outer_loop.update(attitude_error, dt)
        
        # Saturate desired rates
        self.desired_rates = np.clip(self.desired_rates, -self.max_rate, self.max_rate)
        
        # Inner loop: rate error → torque
        rate_error = self.desired_rates - current_rates
        self.control_torque = self.inner_loop.update(rate_error, dt)
        
        # Saturate control torque
        self.control_torque = np.clip(self.control_torque, -self.max_torque, self.max_torque)
        
        return self.control_torque.copy(), self.desired_rates.copy()
    
    def _quaternion_error(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray
    ) -> np.ndarray:
        """
        Calculate quaternion error
        
        Error quaternion: q_error = q_target * q_current^(-1)
        Convert to angle-axis for control (small angle approximation)
        
        Args:
            q_current: Current quaternion [w, x, y, z]
            q_target: Target quaternion [w, x, y, z]
            
        Returns:
            Error vector [3] (axis * angle)
        """
        # Quaternion conjugate (inverse for unit quaternion)
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Error quaternion
        q_error = self._quaternion_multiply(q_target, q_current_conj)
        
        # Extract error axis-angle
        # For small angles: error ≈ 2 * [x, y, z]
        # For larger angles: use proper axis-angle conversion
        w, x, y, z = q_error
        
        if w < 0:  # Ensure shortest rotation
            w, x, y, z = -w, -x, -y, -z
        
        # Axis-angle representation
        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        
        if angle < 1e-6:
            # Small angle: use simplified formula
            error = 2.0 * np.array([x, y, z])
        else:
            # General case
            sin_half_angle = np.sin(angle / 2.0)
            if abs(sin_half_angle) > 1e-6:
                axis = np.array([x, y, z]) / sin_half_angle
                error = axis * angle
            else:
                error = np.zeros(3)
        
        return error
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def get_desired_rates(self) -> np.ndarray:
        """Get current desired angular rates"""
        return self.desired_rates.copy()
    
    def get_control_torque(self) -> np.ndarray:
        """Get current control torque"""
        return self.control_torque.copy()
