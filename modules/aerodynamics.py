"""
Aerodynamics Module - Aerodynamic Forces and Torques

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module models aerodynamic torques produced by control surface deflections.

Features:
- Dynamic pressure dependent effectiveness: q = 0.5 * ρ * V²
- Gain scheduling: effectiveness varies with flight conditions
- Simplified linear model (no coupling or nonlinear effects)

Rationale:
- Simple model captures essential dependency on flight conditions
- Linear effectiveness suitable for small deflections
- Dynamic pressure scaling represents real aerodynamic behavior
"""

import numpy as np
from typing import Tuple


class AerodynamicModel:
    """
    Models aerodynamic torques from control surfaces
    
    Simplified model: τ = C(q) @ δ
    where C(q) is effectiveness matrix scaled by dynamic pressure
    """
    
    def __init__(
        self,
        reference_area: float = 0.01,      # m² (fin area)
        reference_length: float = 0.5,     # m (rocket length)
        base_effectiveness: np.ndarray = None,
        altitude: float = 0.0,             # m (for density)
        velocity: float = 50.0             # m/s (assumed constant)
    ):
        """
        Initialize aerodynamic model
        
        Args:
            reference_area: Reference area for aero coefficients [m²]
            reference_length: Reference length for moment arm [m]
            base_effectiveness: Base effectiveness coefficients
            altitude: Altitude for atmospheric density [m]
            velocity: Rocket velocity [m/s] (simplified: constant)
        """
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.altitude = altitude
        self.velocity = velocity
        
        # Base effectiveness (at reference dynamic pressure)
        if base_effectiveness is None:
            # Default: torque coefficients [N·m/rad] per deflection
            self.base_effectiveness = np.array([
                [ 1.0,  1.0, -1.0, -1.0],  # Roll
                [ 1.0, -1.0,  1.0, -1.0],  # Pitch
                [ 0.5, -0.5, -0.5,  0.5]   # Yaw
            ])
        else:
            self.base_effectiveness = base_effectiveness
        
        # Atmospheric model
        self.rho = self._atmospheric_density(altitude)
        
        # Dynamic pressure
        self.q = 0.5 * self.rho * self.velocity**2
        
        # Current effectiveness (scaled by dynamic pressure)
        self._update_effectiveness()
    
    def _atmospheric_density(self, altitude: float) -> float:
        """
        Calculate atmospheric density at altitude
        
        Simplified exponential model
        
        Args:
            altitude: Height above sea level [m]
            
        Returns:
            Air density [kg/m³]
        """
        rho_0 = 1.225  # Sea level density [kg/m³]
        scale_height = 8500.0  # Scale height [m]
        
        rho = rho_0 * np.exp(-altitude / scale_height)
        return rho
    
    def _update_effectiveness(self):
        """Update effectiveness matrix based on current dynamic pressure"""
        # Scale base effectiveness by dynamic pressure and reference dimensions
        scale_factor = self.q * self.reference_area * self.reference_length
        
        # Normalize by reference dynamic pressure (q = 1000 Pa)
        q_ref = 1000.0
        scale_factor = scale_factor / q_ref
        
        self.effectiveness = self.base_effectiveness * scale_factor
    
    def set_flight_conditions(self, velocity: float, altitude: float = None):
        """
        Update flight conditions and recalculate effectiveness
        
        Args:
            velocity: Rocket velocity [m/s]
            altitude: Altitude [m] (if None, keeps current)
        """
        self.velocity = velocity
        
        if altitude is not None:
            self.altitude = altitude
            self.rho = self._atmospheric_density(altitude)
        
        # Update dynamic pressure
        self.q = 0.5 * self.rho * self.velocity**2
        
        # Update effectiveness
        self._update_effectiveness()
    
    def compute_torque(self, deflections: np.ndarray) -> np.ndarray:
        """
        Compute aerodynamic torque from elevon deflections
        
        Args:
            deflections: Elevon deflections [δ1, δ2, δ3, δ4] rad
            
        Returns:
            Aerodynamic torque [τx, τy, τz] N·m
        """
        torque = self.effectiveness @ deflections
        return torque
    
    def get_effectiveness_matrix(self) -> np.ndarray:
        """Get current effectiveness matrix"""
        return self.effectiveness.copy()
    
    def get_dynamic_pressure(self) -> float:
        """Get current dynamic pressure [Pa]"""
        return self.q
    
    def get_flight_conditions(self) -> Tuple[float, float, float]:
        """
        Get current flight conditions
        
        Returns:
            (velocity [m/s], altitude [m], dynamic_pressure [Pa])
        """
        return self.velocity, self.altitude, self.q
