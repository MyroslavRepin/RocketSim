"""
Unit Tests for Physics Module

⚠️ WARNING: For educational simulation only. Not for real aerospace systems.

Tests for the physics module (quaternion dynamics and Euler's equations).
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from physics import RocketPhysics


class TestRocketPhysics(unittest.TestCase):
    """Test cases for RocketPhysics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.physics = RocketPhysics()
        self.dt = 0.01
    
    def test_initialization(self):
        """Test physics initialization"""
        q, omega = self.physics.get_state()
        
        # Should start at identity quaternion
        np.testing.assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(omega, [0.0, 0.0, 0.0])
    
    def test_quaternion_normalization(self):
        """Test that quaternion stays normalized during integration"""
        # Apply random torques
        for _ in range(1000):
            torque = np.random.randn(3) * 0.1
            self.physics.step(torque, self.dt)
        
        q, _ = self.physics.get_state()
        norm = np.linalg.norm(q)
        
        self.assertAlmostEqual(norm, 1.0, places=6)
    
    def test_zero_torque_stable(self):
        """Test that zero torque maintains state"""
        # Set initial rates to zero
        self.physics.reset()
        
        initial_q, initial_omega = self.physics.get_state()
        
        # Run with zero torque
        for _ in range(100):
            self.physics.step(np.zeros(3), self.dt)
        
        final_q, final_omega = self.physics.get_state()
        
        # State should remain unchanged
        np.testing.assert_array_almost_equal(final_q, initial_q, decimal=5)
        np.testing.assert_array_almost_equal(final_omega, initial_omega, decimal=5)
    
    def test_constant_torque_acceleration(self):
        """Test that constant torque produces angular acceleration"""
        self.physics.reset()
        
        # Apply constant torque
        torque = np.array([0.1, 0.0, 0.0])
        
        # Get initial rate
        _, initial_omega = self.physics.get_state()
        
        # Integrate for some time
        for _ in range(100):
            self.physics.step(torque, self.dt)
        
        # Get final rate
        _, final_omega = self.physics.get_state()
        
        # Rate should have increased
        self.assertGreater(final_omega[0], initial_omega[0])
    
    def test_euler_quaternion_conversion(self):
        """Test conversion between Euler angles and quaternions"""
        # Test several angles
        test_angles = [
            (0.0, 0.0, 0.0),
            (np.pi/4, 0.0, 0.0),
            (0.0, np.pi/6, 0.0),
            (0.0, 0.0, np.pi/3),
            (0.1, 0.2, 0.3)
        ]
        
        for roll, pitch, yaw in test_angles:
            # Convert to quaternion
            q = self.physics.euler_to_quaternion(roll, pitch, yaw)
            
            # Convert back to Euler
            roll_back, pitch_back, yaw_back = self.physics.quaternion_to_euler(q)
            
            # Should match (with some tolerance)
            self.assertAlmostEqual(roll, roll_back, places=6)
            self.assertAlmostEqual(pitch, pitch_back, places=6)
            self.assertAlmostEqual(yaw, yaw_back, places=6)
    
    def test_quaternion_multiplication(self):
        """Test quaternion multiplication"""
        # Identity * q = q
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.707, 0.707, 0.0, 0.0])
        q2 = q2 / np.linalg.norm(q2)  # Normalize
        
        result = self.physics.quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, q2)
        
        # q * q^(-1) = identity
        q_conj = self.physics.quaternion_conjugate(q2)
        result = self.physics.quaternion_multiply(q2, q_conj)
        
        # Should be close to identity (allowing for numerical error)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        np.testing.assert_array_almost_equal(result[1:], [0.0, 0.0, 0.0], decimal=5)
    
    def test_rotation_matrix(self):
        """Test rotation matrix generation"""
        # Identity quaternion should give identity matrix
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = self.physics.rotation_matrix(q)
        
        np.testing.assert_array_almost_equal(R, np.eye(3))
        
        # Rotation matrix should be orthogonal
        q = self.physics.euler_to_quaternion(0.3, 0.4, 0.5)
        R = self.physics.rotation_matrix(q)
        
        # R * R^T = I
        product = R @ R.T
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)
        
        # det(R) = 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=6)
    
    def test_reset(self):
        """Test physics reset"""
        # Disturb the state
        for _ in range(100):
            self.physics.step(np.random.randn(3), self.dt)
        
        # Reset
        self.physics.reset()
        
        q, omega = self.physics.get_state()
        np.testing.assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(omega, [0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
