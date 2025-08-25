"""
Unit tests for SEU solvers module.

Tests the entity alignment solvers including Hungarian and Sinkhorn algorithms.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from seu.core.solvers import hungarian_solve, sinkhorn_solve


class TestHungarianSolver:
    """Test cases for hungarian_solve function."""
    
    def test_hungarian_solve_identity_matrix(self):
        """Test solving perfect identity similarity matrix."""
        # Perfect identity matrix
        similarity_matrix = np.eye(3)
        
        # Solve
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should return numpy arrays
        assert isinstance(row_indices, np.ndarray)
        assert isinstance(col_indices, np.ndarray)
        assert len(row_indices) == 3
        assert len(col_indices) == 3
        
        # For identity matrix, should get perfect assignment
        assignments = dict(zip(row_indices, col_indices))
        expected_assignments = {0: 0, 1: 1, 2: 2}
        assert assignments == expected_assignments
    
    def test_hungarian_solve_small_matrix(self):
        """Test solving small similarity matrix."""
        # Test matrix where optimal assignment is not identity
        similarity_matrix = np.array([
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.3],
            [0.2, 0.3, 0.9]
        ])
        
        # Solve
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should return numpy arrays
        assert isinstance(row_indices, np.ndarray)
        assert isinstance(col_indices, np.ndarray)
        assert len(row_indices) == 3
        assert len(col_indices) == 3
        
        # Check that it's a valid assignment (each row and column used exactly once)
        assert set(row_indices) == {0, 1, 2}
        assert set(col_indices) == {0, 1, 2}
        assert len(set(row_indices)) == 3
        assert len(set(col_indices)) == 3
    
    def test_hungarian_solve_rectangular_matrix(self):
        """Test solving rectangular similarity matrix."""
        # 3x4 matrix
        similarity_matrix = np.array([
            [0.1, 0.9, 0.2, 0.3],
            [0.8, 0.1, 0.3, 0.4],
            [0.2, 0.3, 0.9, 0.1]
        ])
        
        # Solve
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should return numpy arrays
        assert isinstance(row_indices, np.ndarray)
        assert isinstance(col_indices, np.ndarray)
        assert len(row_indices) == 3  # min(3, 4)
        assert len(col_indices) == 3
        
        # Check validity
        assert len(set(row_indices)) == 3  # All rows should be assigned
        assert len(set(col_indices)) == 3  # 3 different columns should be assigned
        assert max(col_indices) < 4  # Column indices should be valid
    
    def test_hungarian_solve_large_matrix(self):
        """Test solving large similarity matrix."""
        # Generate random 50x50 matrix
        np.random.seed(42)  # For reproducible results
        similarity_matrix = np.random.rand(50, 50)
        
        # Solve
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should return numpy arrays
        assert isinstance(row_indices, np.ndarray)
        assert isinstance(col_indices, np.ndarray)
        assert len(row_indices) == 50
        assert len(col_indices) == 50
        
        # Check that it's a valid assignment
        assert set(row_indices) == set(range(50))
        assert set(col_indices) == set(range(50))
        assert len(set(row_indices)) == 50
        assert len(set(col_indices)) == 50
    
    def test_hungarian_solve_tensor_input(self):
        """Test solving with TensorFlow tensor input."""
        try:
            import tensorflow as tf
            similarity_matrix = tf.constant([[1.0, 0.2], [0.3, 0.9]])
            
            # Should work with tensor input (converted internally)
            row_indices, col_indices = hungarian_solve(similarity_matrix)
            
            assert isinstance(row_indices, np.ndarray)
            assert isinstance(col_indices, np.ndarray)
            assert len(row_indices) == 2
            assert len(col_indices) == 2
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_hungarian_solve_maximization(self):
        """Test that Hungarian algorithm maximizes similarity."""
        # Matrix where maximum assignment is clear
        similarity_matrix = np.array([
            [1.0, 0.1],
            [0.1, 1.0]
        ])
        
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should assign 0->0 and 1->1 for maximum total similarity
        assignments = dict(zip(row_indices, col_indices))
        assert assignments[0] == 0
        assert assignments[1] == 1
    
    @patch('seu.core.solvers.optimize.linear_sum_assignment')
    def test_hungarian_solve_scipy_integration(self, mock_linear_sum):
        """Test integration with scipy.optimize.linear_sum_assignment."""
        # Mock scipy to return specific values
        mock_linear_sum.return_value = (np.array([0, 1, 2]), np.array([2, 0, 1]))
        
        similarity_matrix = np.eye(3)
        row_indices, col_indices = hungarian_solve(similarity_matrix)
        
        # Should call scipy with maximize=True
        mock_linear_sum.assert_called_once_with(similarity_matrix, maximize=True)
        
        # Should return the mocked values
        np.testing.assert_array_equal(row_indices, [0, 1, 2])
        np.testing.assert_array_equal(col_indices, [2, 0, 1])


class TestSinkhornSolver:
    """Test cases for sinkhorn_solve function."""
    
    def test_sinkhorn_solve_identity_matrix(self):
        """Test solving perfect identity similarity matrix."""
        # Perfect identity matrix
        similarity_matrix = np.eye(3)
        
        # Solve with moderate temperature for testing
        result_matrix = sinkhorn_solve(similarity_matrix, temperature=10.0, max_iterations=10)
        
        # Should return numpy array
        assert isinstance(result_matrix, np.ndarray)
        assert result_matrix.shape == (3, 3)
        
        # Result should be doubly stochastic (rows and columns sum to ~1)
        row_sums = np.sum(result_matrix, axis=1)
        col_sums = np.sum(result_matrix, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-2)
        
        # All elements should be non-negative
        assert np.all(result_matrix >= 0)
    
    def test_sinkhorn_solve_small_matrix(self):
        """Test solving small similarity matrix."""
        similarity_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        result_matrix = sinkhorn_solve(similarity_matrix, temperature=10.0, max_iterations=10)
        
        # Should return numpy array
        assert isinstance(result_matrix, np.ndarray)
        assert result_matrix.shape == (3, 3)
        
        # Result should be doubly stochastic
        row_sums = np.sum(result_matrix, axis=1)
        col_sums = np.sum(result_matrix, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-2)
        
        # All elements should be non-negative
        assert np.all(result_matrix >= 0)
    
    def test_sinkhorn_solve_temperature_effects(self):
        """Test temperature parameter effects on solution."""
        similarity_matrix = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        
        # Low temperature should make solution more deterministic
        result_low = sinkhorn_solve(similarity_matrix, temperature=1.0, max_iterations=10)
        
        # High temperature should make solution more uniform
        result_high = sinkhorn_solve(similarity_matrix, temperature=50.0, max_iterations=10)
        
        # Both should be doubly stochastic
        for result in [result_low, result_high]:
            row_sums = np.sum(result, axis=1)
            col_sums = np.sum(result, axis=0)
            np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
            np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)
        
        # Results should be different
        assert not np.allclose(result_low, result_high, atol=1e-2)
        
        # Both should have valid values
        assert np.all(result_low >= 0) and np.all(result_low <= 1)
        assert np.all(result_high >= 0) and np.all(result_high <= 1)
    
    def test_sinkhorn_solve_max_iterations(self):
        """Test max_iterations parameter."""
        similarity_matrix = np.eye(3)
        
        # Test with different iteration counts
        result_few = sinkhorn_solve(similarity_matrix, temperature=10.0, max_iterations=1)
        result_many = sinkhorn_solve(similarity_matrix, temperature=10.0, max_iterations=20)
        
        # Both should be valid doubly stochastic matrices
        for result in [result_few, result_many]:
            assert isinstance(result, np.ndarray)
            assert result.shape == (3, 3)
            assert np.all(result >= 0)
            
            # Check doubly stochastic property (more lenient for few iterations)
            row_sums = np.sum(result, axis=1)
            col_sums = np.sum(result, axis=0)
            np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
            np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)
    
    def test_sinkhorn_solve_tensorflow_backend(self):
        """Test TensorFlow backend when available."""
        try:
            import tensorflow as tf
            similarity_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
            
            # Should work with TensorFlow backend
            result = sinkhorn_solve(similarity_matrix, temperature=5.0, max_iterations=10)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 2)
            
            # Result should be doubly stochastic
            row_sums = np.sum(result, axis=1)
            col_sums = np.sum(result, axis=0)
            np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
            np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_sinkhorn_solve_numpy_fallback(self):
        """Test NumPy fallback when TensorFlow is not available."""
        similarity_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        # Force NumPy backend by mocking HAS_TF
        with patch('seu.core.solvers.HAS_TF', False):
            result = sinkhorn_solve(similarity_matrix, temperature=5.0, max_iterations=10)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        
        # Result should be doubly stochastic
        row_sums = np.sum(result, axis=1)
        col_sums = np.sum(result, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-2)
    
    def test_sinkhorn_solve_tensor_input(self):
        """Test with TensorFlow tensor input."""
        try:
            import tensorflow as tf
            similarity_matrix = tf.constant([[0.8, 0.2], [0.3, 0.7]], dtype=tf.float32)
            
            result = sinkhorn_solve(similarity_matrix, temperature=5.0, max_iterations=5)
            
            # Should return numpy array
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 2)
            
            # Result should be doubly stochastic
            row_sums = np.sum(result, axis=1)
            col_sums = np.sum(result, axis=0)
            np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
            np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_sinkhorn_solve_convergence(self):
        """Test algorithm convergence properties."""
        similarity_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # With perfect diagonal matrix, should converge to identity-like result
        result = sinkhorn_solve(similarity_matrix, temperature=1.0, max_iterations=50)
        
        # Should be doubly stochastic
        row_sums = np.sum(result, axis=1)
        col_sums = np.sum(result, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-3)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-3)
        
        # For identity matrix with low temperature, diagonal should dominate
        diagonal = np.diag(result)
        off_diagonal = result - np.diag(diagonal)
        assert np.mean(diagonal) > np.mean(np.abs(off_diagonal))
    
    def test_sinkhorn_solve_numerical_stability(self):
        """Test numerical stability with challenging values."""
        # Matrix with large and small values
        similarity_matrix = np.array([[10.0, 0.1], [0.1, 10.0]])
        
        result = sinkhorn_solve(similarity_matrix, temperature=0.5, max_iterations=20)
        
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result >= 0)
        
        # Should still be approximately doubly stochastic
        row_sums = np.sum(result, axis=1)
        col_sums = np.sum(result, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-2)
    
    def test_sinkhorn_solve_edge_cases(self):
        """Test edge cases like single element matrix."""
        # Single element matrix
        single_matrix = np.array([[0.5]])
        result_single = sinkhorn_solve(single_matrix, temperature=1.0, max_iterations=5)
        
        assert result_single.shape == (1, 1)
        assert abs(result_single[0, 0] - 1.0) < 1e-6  # Should be exactly 1
        
        # 2x2 matrix with zeros
        zero_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
        result_zero = sinkhorn_solve(zero_matrix, temperature=1.0, max_iterations=5)
        
        assert result_zero.shape == (2, 2)
        # After exponentiation, should become uniform
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(result_zero, expected, atol=1e-2)
    
    def test_sinkhorn_solve_consistency(self):
        """Test that results are consistent for same inputs."""
        similarity_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        result1 = sinkhorn_solve(similarity_matrix, temperature=5.0, max_iterations=10)
        result2 = sinkhorn_solve(similarity_matrix, temperature=5.0, max_iterations=10)
        
        # Should produce identical results
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    def test_sinkhorn_solve_notebook_compatibility(self):
        """Test compatibility with notebook parameters."""
        # Use exact parameters from main.ipynb
        similarity_matrix = np.random.rand(5, 5)
        np.random.seed(42)
        
        result = sinkhorn_solve(similarity_matrix, temperature=50.0, max_iterations=10)
        
        # Should work with notebook parameters
        assert result.shape == (5, 5)
        row_sums = np.sum(result, axis=1)
        col_sums = np.sum(result, axis=0)
        np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)