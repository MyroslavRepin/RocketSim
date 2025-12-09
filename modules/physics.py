"""
Physics Module - 3-Axis Rotational Dynamics

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module models the rotational dynamics of a rocket using quaternions
for attitude representation and Euler's rotational equations.

Rationale:
- Quaternions avoid gimbal lock and provide singularity-free representation
- Euler's equations describe rotational dynamics: I·ω̇ = τ - ω × (I·ω)
- Using RK4 integration for good accuracy with reasonable time steps
"""

import numpy as np
from typing import Tuple


class RocketPhysics:
    """Models 3-axis rotational dynamics of a rocket"""
    
    def __init__(self, inertia: np.ndarray = None):
        """
        Initialize rocket physics model
        
        Args:
            inertia: 3x3 inertia tensor [kg·m²]. If None, uses default values.
        
        Default inertia represents a typical small rocket:
        - Ixx, Iyy: Roll moment of inertia (smaller - rocket is long and thin)
        - Izz: Pitch/yaw moment of inertia (larger)
        """
        if inertia is None:
            # Default inertia for small rocket (approximate values)
            self.inertia = np.array([
                [0.05, 0.0, 0.0],    # Ixx
                [0.0, 1.0, 0.0],      # Iyy
                [0.0, 0.0, 1.0]       # Izz
            ])
        else:
            self.inertia = inertia
            
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # State: [quaternion (4), angular_velocity (3)]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] - identity
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # [p, q, r] rad/s
        
    def reset(self, quaternion: np.ndarray = None, angular_velocity: np.ndarray = None):
        """Reset state to initial conditions"""
        if quaternion is not None:
            self.quaternion = quaternion / np.linalg.norm(quaternion)  # Normalize
        else:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity
        else:
            self.angular_velocity = np.array([0.0, 0.0, 0.0])
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current attitude (quaternion) and angular velocity"""
        return self.quaternion.copy(), self.angular_velocity.copy()
    
    def quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute quaternion time derivative
        
        q̇ = 0.5 * Ω(ω) * q
        where Ω is the skew-symmetric matrix formed from ω
        """
        w, x, y, z = q
        p, q_rate, r = omega
        
        q_dot = 0.5 * np.array([
            -p*x - q_rate*y - r*z,
            p*w + r*y - q_rate*z,
            q_rate*w - r*x + p*z,
            r*w + q_rate*x - p*y
        ])
        return q_dot
    
    def angular_acceleration(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """
        Compute angular acceleration from Euler's rotational equations
        
        I·ω̇ = τ - ω × (I·ω)
        
        Args:
            omega: Angular velocity [p, q, r] rad/s
            torque: Applied torque [τx, τy, τz] N·m
            
        Returns:
            Angular acceleration [ṗ, q̇, ṙ] rad/s²
        """
        # Gyroscopic term: ω × (I·ω)
        I_omega = self.inertia @ omega
        gyroscopic = np.cross(omega, I_omega)
        
        # ω̇ = I⁻¹(τ - ω × (I·ω))
        omega_dot = self.inv_inertia @ (torque - gyroscopic)
        return omega_dot
    
    def step(self, torque: np.ndarray, dt: float):
        """
        Integrate dynamics forward one time step using RK4
        
        Args:
            torque: Control torque [τx, τy, τz] N·m
            dt: Time step [s]
        """
        # RK4 integration for angular velocity
        omega = self.angular_velocity
        
        k1_omega = self.angular_acceleration(omega, torque)
        k2_omega = self.angular_acceleration(omega + 0.5*dt*k1_omega, torque)
        k3_omega = self.angular_acceleration(omega + 0.5*dt*k2_omega, torque)
        k4_omega = self.angular_acceleration(omega + dt*k3_omega, torque)
        
        self.angular_velocity = omega + (dt/6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        
        # RK4 integration for quaternion
        q = self.quaternion
        
        k1_q = self.quaternion_derivative(q, omega)
        k2_q = self.quaternion_derivative(q + 0.5*dt*k1_q, omega + 0.5*dt*k1_omega)
        k3_q = self.quaternion_derivative(q + 0.5*dt*k2_q, omega + 0.5*dt*k2_omega)
        k4_q = self.quaternion_derivative(q + dt*k3_q, omega + dt*k3_omega)
        
        self.quaternion = q + (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        
        # Normalize quaternion to prevent numerical drift
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
    
    def quaternion_to_euler(self, q: np.ndarray = None) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        
        Args:
            q: Quaternion [w, x, y, z]. If None, uses current state.
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        if q is None:
            q = self.quaternion
            
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to quaternion
        
        Args:
            roll, pitch, yaw: Euler angles in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
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
    
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions: q1 * q2
        
        Used for composing rotations
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Return quaternion conjugate (inverse rotation)"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def rotation_matrix(self, q: np.ndarray = None) -> np.ndarray:
        """
        Convert quaternion to rotation matrix
        
        Args:
            q: Quaternion [w, x, y, z]. If None, uses current state.
            
        Returns:
            3x3 rotation matrix
        """
        if q is None:
            q = self.quaternion
            
        w, x, y, z = q
        
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        
        return R
