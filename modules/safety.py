"""
Safety Module - Constraint Enforcement and Monitoring

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

This module enforces operational constraints and implements safety features:
- Control saturation monitoring
- Angular rate limiting with automatic damping mode
- Anomaly detection and logging
- Emergency shutdown conditions

Rationale:
- Essential for preventing unrealistic behavior in simulation
- Studies failure modes and safety margins
- Demonstrates real-world safety considerations
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SafetyEvent:
    """Records a safety-related event"""
    time: float
    event_type: str
    severity: str  # "info", "warning", "critical"
    description: str
    data: Dict


class SafetyMonitor:
    """
    Monitors system state and enforces safety constraints
    
    Features:
    - Rate limiting with automatic damping
    - Constraint violation detection
    - Event logging
    """
    
    def __init__(
        self,
        max_angular_rate: float = 5.0,     # rad/s
        damping_threshold: float = 4.0,     # rad/s (triggers damping mode)
        max_attitude_error: float = 1.57,   # rad (~90 degrees)
        enable_auto_damping: bool = True
    ):
        """
        Initialize safety monitor
        
        Args:
            max_angular_rate: Maximum allowable angular rate [rad/s]
            damping_threshold: Rate threshold to trigger damping mode [rad/s]
            max_attitude_error: Maximum allowable attitude error [rad]
            enable_auto_damping: Enable automatic damping mode
        """
        self.max_angular_rate = max_angular_rate
        self.damping_threshold = damping_threshold
        self.max_attitude_error = max_attitude_error
        self.enable_auto_damping = enable_auto_damping
        
        # Safety state
        self.damping_mode = False
        self.emergency_stop = False
        
        # Event log
        self.events: List[SafetyEvent] = []
        
        # Statistics
        self.max_rate_violations = 0
        self.attitude_error_violations = 0
    
    def reset(self):
        """Reset safety monitor"""
        self.damping_mode = False
        self.emergency_stop = False
        self.events.clear()
        self.max_rate_violations = 0
        self.attitude_error_violations = 0
    
    def check_state(
        self,
        time: float,
        angular_velocity: np.ndarray,
        attitude_error: float,
        control_torque: np.ndarray
    ) -> Tuple[bool, bool]:
        """
        Check system state for safety violations
        
        Args:
            time: Current simulation time [s]
            angular_velocity: Angular rates [p, q, r] rad/s
            attitude_error: Attitude error magnitude [rad]
            control_torque: Control torque [τx, τy, τz] N·m
            
        Returns:
            (is_safe, damping_mode_active)
        """
        is_safe = True
        
        # Check angular rate limits
        rate_magnitude = np.linalg.norm(angular_velocity)
        
        if rate_magnitude > self.max_angular_rate:
            is_safe = False
            self.max_rate_violations += 1
            self._log_event(
                time,
                "rate_violation",
                "critical",
                f"Angular rate exceeded limit: {rate_magnitude:.3f} > {self.max_angular_rate:.3f} rad/s",
                {"rate": rate_magnitude, "limit": self.max_angular_rate}
            )
        
        # Check for damping mode trigger
        if self.enable_auto_damping and rate_magnitude > self.damping_threshold:
            if not self.damping_mode:
                self.damping_mode = True
                self._log_event(
                    time,
                    "damping_activated",
                    "warning",
                    f"Damping mode activated due to high rate: {rate_magnitude:.3f} rad/s",
                    {"rate": rate_magnitude}
                )
        else:
            if self.damping_mode:
                self.damping_mode = False
                self._log_event(
                    time,
                    "damping_deactivated",
                    "info",
                    "Damping mode deactivated - rates nominal",
                    {"rate": rate_magnitude}
                )
        
        # Check attitude error
        if attitude_error > self.max_attitude_error:
            self.attitude_error_violations += 1
            self._log_event(
                time,
                "attitude_error",
                "warning",
                f"Large attitude error: {np.degrees(attitude_error):.1f} degrees",
                {"error_rad": attitude_error, "error_deg": np.degrees(attitude_error)}
            )
        
        return is_safe, self.damping_mode
    
    def apply_damping(
        self,
        angular_velocity: np.ndarray,
        damping_gain: float = 0.2
    ) -> np.ndarray:
        """
        Calculate damping torque to reduce angular rates
        
        Simple proportional damping: τ_damp = -k * ω
        
        Args:
            angular_velocity: Current angular rates [rad/s]
            damping_gain: Damping coefficient
            
        Returns:
            Damping torque [N·m]
        """
        damping_torque = -damping_gain * angular_velocity
        return damping_torque
    
    def saturate_control(
        self,
        control_signal: np.ndarray,
        limits: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Apply saturation limits to control signal
        
        Args:
            control_signal: Control signal to saturate
            limits: Saturation limits (positive values)
            
        Returns:
            (saturated_signal, was_saturated)
        """
        saturated = np.clip(control_signal, -limits, limits)
        was_saturated = not np.allclose(saturated, control_signal)
        
        return saturated, was_saturated
    
    def _log_event(
        self,
        time: float,
        event_type: str,
        severity: str,
        description: str,
        data: Dict
    ):
        """Log a safety event"""
        event = SafetyEvent(
            time=time,
            event_type=event_type,
            severity=severity,
            description=description,
            data=data
        )
        self.events.append(event)
    
    def get_events(self, severity: str = None) -> List[SafetyEvent]:
        """
        Get logged events
        
        Args:
            severity: Filter by severity ("info", "warning", "critical")
                     If None, returns all events
                     
        Returns:
            List of safety events
        """
        if severity is None:
            return self.events.copy()
        else:
            return [e for e in self.events if e.severity == severity]
    
    def get_statistics(self) -> Dict:
        """
        Get safety statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_events": len(self.events),
            "rate_violations": self.max_rate_violations,
            "attitude_errors": self.attitude_error_violations,
            "critical_events": len([e for e in self.events if e.severity == "critical"]),
            "warning_events": len([e for e in self.events if e.severity == "warning"]),
            "info_events": len([e for e in self.events if e.severity == "info"])
        }
    
    def print_summary(self):
        """Print safety event summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Safety Monitor Summary")
        print("="*60)
        print(f"Total Events: {stats['total_events']}")
        print(f"  - Critical: {stats['critical_events']}")
        print(f"  - Warning:  {stats['warning_events']}")
        print(f"  - Info:     {stats['info_events']}")
        print(f"\nViolations:")
        print(f"  - Rate Limit: {stats['rate_violations']}")
        print(f"  - Attitude Error: {stats['attitude_errors']}")
        print("="*60 + "\n")
        
        # Print recent critical events
        critical = [e for e in self.events if e.severity == "critical"]
        if critical:
            print("Critical Events:")
            for event in critical[-5:]:  # Last 5
                print(f"  [{event.time:.2f}s] {event.description}")
            print()
