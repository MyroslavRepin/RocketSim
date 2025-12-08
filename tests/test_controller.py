"""
Unit Tests for Controllers

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

Tests for the dual-loop controller implementation.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from controllers import PDController, PIController, DualLoopController


class TestPDController(unittest.TestCase):
    """Test cases for PD controller"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.kp = np.array([1.0, 1.0, 1.0])
        self.kd = np.array([0.5, 0.5, 0.5])
        self.controller = PDController(self.kp, self.kd)
        self.dt = 0.01
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.prev_error.shape, (3,))
        self.assertEqual(self.controller.error_derivative.shape, (3,))
    
    def test_zero_error(self):
        """Test that zero error produces zero output"""
        error = np.zeros(3)
        output = self.controller.update(error, self.dt)
        
        np.testing.assert_array_almost_equal(output, np.zeros(3))
    
    def test_proportional_response(self):
        """Test proportional response to error"""
        error = np.array([1.0, 0.0, 0.0])
        
        # First update (derivative is zero initially)
        output = self.controller.update(error, self.dt)
        
        # Should have proportional component
        self.assertGreater(output[0], 0)
        self.assertEqual(output[1], 0)
        self.assertEqual(output[2], 0)
    
    def test_derivative_response(self):
        """Test derivative response to changing error"""
        # Apply increasing error
        for i in range(5):
            error = np.array([i * 0.1, 0.0, 0.0])
            output = self.controller.update(error, self.dt)
        
        # Derivative term should contribute to output
        self.assertGreater(output[0], 0)
    
    def test_reset(self):
        """Test controller reset"""
        # Build up some state
        for _ in range(10):
            self.controller.update(np.random.randn(3), self.dt)
        
        # Reset
        self.controller.reset()
        
        np.testing.assert_array_equal(self.controller.prev_error, np.zeros(3))
        np.testing.assert_array_equal(self.controller.error_derivative, np.zeros(3))


class TestPIController(unittest.TestCase):
    """Test cases for PI controller"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.kp = np.array([1.0, 1.0, 1.0])
        self.ki = np.array([0.1, 0.1, 0.1])
        self.controller = PIController(self.kp, self.ki)
        self.dt = 0.01
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.integral.shape, (3,))
        np.testing.assert_array_equal(self.controller.integral, np.zeros(3))
    
    def test_integral_buildup(self):
        """Test integral term builds up over time"""
        error = np.array([0.1, 0.0, 0.0])
        
        outputs = []
        for _ in range(100):
            output = self.controller.update(error, self.dt)
            outputs.append(output[0])
        
        # Output should increase due to integral term
        self.assertGreater(outputs[-1], outputs[0])
    
    def test_integral_windup_protection(self):
        """Test that integral is clamped (anti-windup)"""
        large_error = np.array([100.0, 0.0, 0.0])
        
        # Apply large error for many steps
        for _ in range(1000):
            self.controller.update(large_error, self.dt)
        
        # Integral should be limited
        self.assertLessEqual(abs(self.controller.integral[0]), 
                           self.controller.integral_limit[0])
    
    def test_reset(self):
        """Test controller reset"""
        # Build up integral
        for _ in range(100):
            self.controller.update(np.array([1.0, 0.0, 0.0]), self.dt)
        
        # Reset
        self.controller.reset()
        
        np.testing.assert_array_equal(self.controller.integral, np.zeros(3))


class TestDualLoopController(unittest.TestCase):
    """Test cases for dual-loop controller"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = DualLoopController()
        self.dt = 0.01
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertIsNotNone(self.controller.outer_loop)
        self.assertIsNotNone(self.controller.inner_loop)
    
    def test_zero_error_stable(self):
        """Test that zero error produces zero output"""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        q_target = np.array([1.0, 0.0, 0.0, 0.0])
        rates = np.zeros(3)
        
        torque, desired_rates = self.controller.update(
            q_current, q_target, rates, self.dt
        )
        
        # Both outputs should be near zero
        np.testing.assert_array_almost_equal(torque, np.zeros(3), decimal=5)
        np.testing.assert_array_almost_equal(desired_rates, np.zeros(3), decimal=5)
    
    def test_attitude_error_response(self):
        """Test response to attitude error"""
        # Current: identity, Target: 45 deg roll
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        
        # 45 deg roll quaternion
        angle = np.radians(45)
        q_target = np.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])
        
        rates = np.zeros(3)
        
        torque, desired_rates = self.controller.update(
            q_current, q_target, rates, self.dt
        )
        
        # Should command positive roll rate and torque
        self.assertGreater(desired_rates[0], 0)
        self.assertGreater(torque[0], 0)
    
    def test_rate_saturation(self):
        """Test that desired rates are saturated"""
        # Large attitude error
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        q_target = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg
        rates = np.zeros(3)
        
        torque, desired_rates = self.controller.update(
            q_current, q_target, rates, self.dt
        )
        
        # Desired rates should be limited
        self.assertLessEqual(np.max(np.abs(desired_rates)), 
                           self.controller.max_rate)
    
    def test_torque_saturation(self):
        """Test that control torque is saturated"""
        q_current = np.array([1.0, 0.0, 0.0, 0.0])
        q_target = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Large rate error
        rates = np.array([10.0, 10.0, 10.0])
        
        torque, desired_rates = self.controller.update(
            q_current, q_target, rates, self.dt
        )
        
        # Torque should be limited
        self.assertLessEqual(np.max(np.abs(torque)), 
                           self.controller.max_torque)
    
    def test_reset(self):
        """Test controller reset"""
        # Run some updates
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rates = np.random.randn(3)
        
        for _ in range(10):
            self.controller.update(q, q, rates, self.dt)
        
        # Reset
        self.controller.reset()
        
        np.testing.assert_array_equal(self.controller.desired_rates, np.zeros(3))
        np.testing.assert_array_equal(self.controller.control_torque, np.zeros(3))


if __name__ == '__main__':
    unittest.main()
