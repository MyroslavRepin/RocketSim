"""
Guidance Module - Target Orientation Calculation

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module calculates desired attitude (target quaternion) based on mission objectives.

Modes:
1. Direct attitude command: specify target orientation directly
2. Point tracking: calculate attitude to point rocket toward a target location

Rationale:
- Separates high-level mission logic from low-level control
- Provides target quaternion for controller to track
"""

import numpy as np
from typing import Optional, Tuple


class GuidanceSystem:
    """Calculates desired rocket attitude based on mission objectives"""
    
    def __init__(self):
        """Initialize guidance system"""
        # Default: point straight up (identity quaternion)
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.mode = "attitude"  # "attitude" or "point_tracking"
        
        # For point tracking mode
        self.target_position = None
        self.rocket_position = None
    
    def set_attitude_target(self, quaternion: np.ndarray = None, euler: np.ndarray = None):
        """
        Set direct attitude target
        
        Args:
            quaternion: Target quaternion [w, x, y, z]
            euler: Target Euler angles [roll, pitch, yaw] in radians
                  (used if quaternion is None)
        """
        self.mode = "attitude"
        
        if quaternion is not None:
            self.target_quaternion = quaternion / np.linalg.norm(quaternion)
        elif euler is not None:
            self.target_quaternion = self._euler_to_quaternion(euler[0], euler[1], euler[2])
        else:
            # Default: identity (zero orientation)
            self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    def set_point_tracking_target(
        self,
        target_position: np.ndarray,
        rocket_position: np.ndarray = None
    ):
        """
        Set point tracking mode - rocket points toward target location
        
        Args:
            target_position: Target point in NED frame [x, y, z] meters
            rocket_position: Rocket position in NED frame [x, y, z] meters
                            If None, assumes rocket at origin
        """
        self.mode = "point_tracking"
        self.target_position = target_position
        
        if rocket_position is None:
            self.rocket_position = np.array([0.0, 0.0, 0.0])
        else:
            self.rocket_position = rocket_position
    
    def update(self, rocket_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update and return target quaternion
        
        Args:
            rocket_position: Current rocket position (for point tracking mode)
            
        Returns:
            Target quaternion [w, x, y, z]
        """
        if self.mode == "attitude":
            # Direct attitude command - target already set
            return self.target_quaternion.copy()
        
        elif self.mode == "point_tracking":
            # Calculate attitude to point toward target
            if rocket_position is not None:
                self.rocket_position = rocket_position
            
            # Vector from rocket to target
            direction = self.target_position - self.rocket_position
            distance = np.linalg.norm(direction)
            
            if distance < 0.01:  # Target reached
                # Maintain current orientation or point up
                return self.target_quaternion.copy()
            
            # Normalize direction vector
            direction_unit = direction / distance
            
            # Calculate quaternion to align rocket's longitudinal axis (+x) with direction
            # Reference body axis: [1, 0, 0] (rocket points along x)
            # Target direction: direction_unit
            
            self.target_quaternion = self._rotation_between_vectors(
                np.array([1.0, 0.0, 0.0]),  # Rocket's longitudinal axis
                direction_unit
            )
            
            return self.target_quaternion.copy()
        
        else:
            return self.target_quaternion.copy()
    
    def get_target(self) -> np.ndarray:
        """Get current target quaternion"""
        return self.target_quaternion.copy()
    
    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Calculate quaternion that rotates v1 to v2
        
        Args:
            v1, v2: Unit vectors
            
        Returns:
            Quaternion representing rotation from v1 to v2
        """
        # Normalize inputs
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Rotation axis: cross product
        axis = np.cross(v1, v2)
        axis_length = np.linalg.norm(axis)
        
        # Angle: dot product
        cos_angle = np.dot(v1, v2)
        
        # Handle parallel vectors
        if axis_length < 1e-6:
            if cos_angle > 0:
                # Vectors are parallel (same direction)
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # Vectors are anti-parallel (opposite direction)
                # Choose arbitrary perpendicular axis
                if abs(v1[0]) < 0.9:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                axis = np.cross(v1, axis)
                axis = axis / np.linalg.norm(axis)
                # 180 degree rotation
                return np.array([0.0, axis[0], axis[1], axis[2]])
        
        # Normalize axis
        axis = axis / axis_length
        
        # Quaternion from axis-angle
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        half_angle = angle / 2.0
        
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
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
    
    def quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles (for display/logging)
        
        Returns:
            [roll, pitch, yaw] in radians
        """
        w, x, y, z = q
        
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return np.array([roll, pitch, yaw])
