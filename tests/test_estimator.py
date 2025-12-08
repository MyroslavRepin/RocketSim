"""
Unit Tests for State Estimator

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

Tests for the complementary filter implementation in the estimator module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from estimator import ComplementaryFilter, StateEstimator


class TestComplementaryFilter(unittest.TestCase):
    """Test cases for ComplementaryFilter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filter = ComplementaryFilter(alpha=0.98)
        self.dt = 0.01
    
    def test_initialization(self):
        """Test filter initialization"""
        q, omega = self.filter.get_state()
        
        # Should start with identity quaternion
        np.testing.assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(omega, [0.0, 0.0, 0.0])
    
    def test_reset(self):
        """Test filter reset"""
        # Run some updates
        for _ in range(10):
            self.filter.update(
                np.array([0.1, 0.1, 0.1]),
                np.array([0.0, 0.0, 9.81]),
                self.dt
            )
        
        # Reset
        self.filter.reset()
        
        q, omega = self.filter.get_state()
        np.testing.assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(omega, [0.0, 0.0, 0.0])
    
    def test_zero_rates_stable(self):
        """Test that zero rates maintain attitude"""
        initial_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.filter.quaternion = initial_q.copy()
        
        # Run with zero gyro rates
        for _ in range(100):
            q, omega = self.filter.update(
                np.array([0.0, 0.0, 0.0]),  # Zero rates
                np.array([0.0, 0.0, 9.81]),  # Gravity down
                self.dt
            )
        
        # Quaternion should remain close to identity
        np.testing.assert_array_almost_equal(q, initial_q, decimal=2)
    
    def test_constant_rate_integration(self):
        """Test integration of constant angular rate"""
        # Start at identity
        self.filter.reset()
        
        # Apply constant roll rate
        roll_rate = 0.5  # rad/s
        duration = 2.0   # seconds
        steps = int(duration / self.dt)
        
        for _ in range(steps):
            self.filter.update(
                np.array([roll_rate, 0.0, 0.0]),
                np.array([0.0, 0.0, 9.81]),
                self.dt
            )
        
        q, _ = self.filter.get_state()
        
        # Convert to Euler and check roll angle
        roll, pitch, yaw = self.filter._quaternion_to_euler(q)
        
        # Expected roll after 2s at 0.5 rad/s = 1.0 rad
        expected_roll = roll_rate * duration
        
        # Should be close (some error due to complementary filter fusion)
        self.assertAlmostEqual(roll, expected_roll, delta=0.2)
    
    def test_quaternion_normalization(self):
        """Test that quaternion stays normalized"""
        for _ in range(1000):
            q, _ = self.filter.update(
                np.random.randn(3) * 0.1,
                np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1,
                self.dt
            )
            
            # Quaternion should always be unit length
            norm = np.linalg.norm(q)
            self.assertAlmostEqual(norm, 1.0, places=6)


class TestStateEstimator(unittest.TestCase):
    """Test cases for StateEstimator wrapper"""
    
    def test_estimator_creation(self):
        """Test creating estimator"""
        estimator = StateEstimator(estimator_type="complementary")
        self.assertIsNotNone(estimator)
    
    def test_invalid_estimator_type(self):
        """Test that invalid estimator type raises error"""
        with self.assertRaises(NotImplementedError):
            StateEstimator(estimator_type="invalid")
    
    def test_estimator_update(self):
        """Test estimator update"""
        estimator = StateEstimator()
        
        q, omega = estimator.update(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 9.81]),
            0.01
        )
        
        self.assertEqual(q.shape, (4,))
        self.assertEqual(omega.shape, (3,))


if __name__ == '__main__':
    unittest.main()
