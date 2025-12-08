"""
Control Allocation Module - Torque to Elevon Mapping

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module maps desired control torques to individual elevon deflections
using an effectiveness matrix. Accounts for:
- 4 elevon configuration (redundant system)
- Saturation limits (physical deflection constraints)
- Servo rate limits
- Pseudo-inverse allocation for redundancy

Rationale:
- Pseudo-inverse provides least-squares solution for over-actuated system
- Saturation prevents unrealistic deflections
- Rate limiting models servo dynamics
"""

import numpy as np
from typing import Optional


class ControlAllocator:
    """
    Allocates control torques to 4 elevon deflections
    
    Configuration:
    - 4 elevons arranged symmetrically around rocket body
    - Each elevon can produce torque in roll, pitch, yaw
    """
    
    def __init__(
        self,
        effectiveness_matrix: Optional[np.ndarray] = None,
        max_deflection: float = 0.35,      # radians (~20 degrees)
        max_rate: float = 3.0,             # rad/s (servo speed)
        dt: float = 0.01
    ):
        """
        Initialize control allocator
        
        Args:
            effectiveness_matrix: 3x4 matrix mapping deflections to torques
                                 torque = B @ deflections
            max_deflection: Maximum elevon deflection [rad]
            max_rate: Maximum deflection rate [rad/s]
            dt: Time step for rate limiting [s]
        """
        if effectiveness_matrix is None:
            # Default effectiveness matrix for 4 elevons
            # Elevons at positions: [+x+y, +x-y, -x+y, -x-y]
            # Each row: [roll, pitch, yaw]
            # Each column: elevon contribution
            self.effectiveness = np.array([
                [ 1.0,  1.0, -1.0, -1.0],  # Roll (x-axis)
                [ 1.0, -1.0,  1.0, -1.0],  # Pitch (y-axis)
                [ 0.5, -0.5, -0.5,  0.5]   # Yaw (z-axis)
            ])
        else:
            self.effectiveness = effectiveness_matrix
        
        # Pseudo-inverse for allocation
        self.effectiveness_inv = np.linalg.pinv(self.effectiveness)
        
        self.max_deflection = max_deflection
        self.max_rate = max_rate
        self.dt = dt
        
        # Current deflections
        self.deflections = np.zeros(4)
    
    def reset(self):
        """Reset elevon deflections to neutral"""
        self.deflections = np.zeros(4)
    
    def allocate(self, desired_torque: np.ndarray) -> np.ndarray:
        """
        Allocate control torque to elevon deflections
        
        Solves: B @ δ = τ
        Using pseudo-inverse: δ = B⁺ @ τ
        
        Args:
            desired_torque: Desired torque [τx, τy, τz] N·m
            
        Returns:
            Elevon deflections [δ1, δ2, δ3, δ4] rad
        """
        # Pseudo-inverse allocation
        commanded_deflections = self.effectiveness_inv @ desired_torque
        
        # Apply rate limiting (servo dynamics)
        deflection_change = commanded_deflections - self.deflections
        max_change = self.max_rate * self.dt
        deflection_change = np.clip(deflection_change, -max_change, max_change)
        
        # Update deflections
        self.deflections = self.deflections + deflection_change
        
        # Apply saturation limits
        self.deflections = np.clip(self.deflections, -self.max_deflection, self.max_deflection)
        
        return self.deflections.copy()
    
    def get_deflections(self) -> np.ndarray:
        """Get current elevon deflections"""
        return self.deflections.copy()
    
    def compute_torque(self, deflections: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute torque from elevon deflections
        
        Args:
            deflections: Elevon deflections [4]. If None, uses current.
            
        Returns:
            Torque [τx, τy, τz] N·m
        """
        if deflections is None:
            deflections = self.deflections
        
        torque = self.effectiveness @ deflections
        return torque
    
    def update_effectiveness(self, dynamic_pressure: float):
        """
        Update effectiveness matrix based on dynamic pressure (gain scheduling)
        
        In reality, aerodynamic effectiveness varies with q = 0.5 * ρ * V²
        This is typically handled in the aerodynamics module.
        
        Args:
            dynamic_pressure: q [Pa]
        """
        # Placeholder for gain scheduling
        # Could scale effectiveness matrix based on dynamic pressure
        pass
