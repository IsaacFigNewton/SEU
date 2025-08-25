"""
Unit tests for SEU evaluation module.

Tests the test function and evaluation metrics functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from seu.core.evaluation import test


class TestEvaluationFunction:
    """Test cases for test function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create test similarity matrices
        self.small_sims = np.array([
            [1.0, 0.8, 0.3],
            [0.4, 0.9, 0.5],
            [0.2, 0.3, 1.0]
        ])
        
        # Perfect alignment case
        self.perfect_sims = np.eye(5)
        
        # Hungarian algorithm test result
        self.hungarian_result = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
        self.hungarian_result_bad = (np.array([0, 1, 2, 3, 4]), np.array([4, 3, 2, 1, 0]))
    
    def test_sinkhorn_mode_perfect_alignment(self):
        """Test Sinkhorn mode with perfect alignment."""
        with patch('builtins.print') as mock_print:
            test(self.perfect_sims, mode="sinkhorn", batch_size=512)
        
        # Check that print was called (numba may produce debug output)
        assert mock_print.called
        # Get the last call which should be the actual result
        last_call = mock_print.call_args[0][0]
        assert "hits@1" in last_call
        assert "hits@10" in last_call
        assert "MRR" in last_call
    
    def test_sinkhorn_mode_with_batch_processing(self):
        """Test Sinkhorn mode with batch processing."""
        # Create larger matrix to test batching
        large_sims = np.random.rand(100, 100) + np.eye(100) * 2  # Add diagonal bias
        
        with patch('builtins.print') as mock_print:
            test(large_sims, mode="sinkhorn", batch_size=32)
        
        # Should complete without errors
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "hits@10" in call_args
        assert "MRR" in call_args
    
    def test_hungarian_mode_perfect(self):
        """Test Hungarian mode with perfect results."""
        with patch('builtins.print') as mock_print:
            test(self.hungarian_result, mode="hungarian")
        
        # Check that print was called with expected format (only hits@1 for Hungarian)
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "100.00%" in call_args  # Perfect Hungarian should give 100%
    
    def test_hungarian_mode_bad_alignment(self):
        """Test Hungarian mode with bad alignment."""
        with patch('builtins.print') as mock_print:
            test(self.hungarian_result_bad, mode="hungarian")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        # Should show low performance for bad alignment
        assert "20.00%" in call_args  # 1 out of 5 correct = 20%
    
    def test_invalid_mode(self):
        """Test with invalid mode parameter."""
        # The current implementation doesn't validate mode, it just uses if/else
        # So invalid mode will go to the else branch (Hungarian)
        with patch('builtins.print') as mock_print:
            test(self.perfect_sims, mode="invalid_mode")  # Will be treated as Hungarian
        
        assert mock_print.called
    
    def test_sinkhorn_with_tensorflow_operations(self):
        """Test that TensorFlow operations work correctly in Sinkhorn mode."""
        # This tests the internal tf.argsort and tf.where operations
        sims = np.array([[1.0, 0.5], [0.3, 1.0]])
        
        with patch('builtins.print'):
            # Should not raise any TensorFlow-related errors
            test(sims, mode="sinkhorn", batch_size=512)
    
    def test_numba_jit_compilation(self):
        """Test that numba compilation works correctly."""
        # The cal function uses @nb.jit(nopython=True)
        sims = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.4], [0.1, 0.2, 1.0]])
        
        with patch('builtins.print'):
            # Should not raise numba-related errors
            test(sims, mode="sinkhorn", batch_size=512)
    
    def test_edge_case_single_entity(self):
        """Test with single entity similarity matrix."""
        single_sim = np.array([[1.0]])
        
        with patch('builtins.print') as mock_print:
            test(single_sim, mode="sinkhorn", batch_size=512)
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "100.00%" in call_args  # Single entity should be 100% correct
    
    def test_edge_case_empty_matrix(self):
        """Test with empty similarity matrix."""
        empty_sims = np.array([]).reshape(0, 0)
        
        # Empty matrix will cause division by zero, so expect an error
        with pytest.raises(ZeroDivisionError):
            test(empty_sims, mode="sinkhorn", batch_size=512)
    
    def test_batch_size_parameter(self):
        """Test different batch sizes."""
        sims = np.random.rand(50, 50) + np.eye(50)
        
        # Test with different batch sizes
        batch_sizes = [10, 25, 100]  # Including one larger than matrix size
        
        for batch_size in batch_sizes:
            with patch('builtins.print'):
                # Should work regardless of batch size
                test(sims, mode="sinkhorn", batch_size=batch_size)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Matrix with very large and small values
        extreme_sims = np.array([
            [100.0, 0.001, 0.001],
            [0.001, 100.0, 0.001],
            [0.001, 0.001, 100.0]
        ])
        
        with patch('builtins.print'):
            # Should handle extreme values without numerical issues
            test(extreme_sims, mode="sinkhorn", batch_size=512)
    
    def test_hungarian_tuple_validation(self):
        """Test validation of Hungarian tuple format."""
        # Test with invalid tuple format
        invalid_hungarian = ([0, 1, 2], [0, 1, 2])  # Lists instead of arrays
        
        with patch('builtins.print'):
            # Should handle lists as well as arrays
            test(invalid_hungarian, mode="hungarian")
    
    def test_tensorflow_tensor_input(self):
        """Test with TensorFlow tensor input."""
        tf_sims = tf.constant([[1.0, 0.5], [0.3, 1.0]], dtype=tf.float32)
        
        with patch('builtins.print'):
            # Should work with TensorFlow tensors
            test(tf_sims, mode="sinkhorn", batch_size=512)
    
    def test_default_batch_size(self):
        """Test with default batch size."""
        with patch('builtins.print'):
            # Should use default batch size when not specified
            test(self.small_sims, mode="sinkhorn")  # No batch_size parameter
    
    def test_print_formatting(self):
        """Test that print output has correct formatting."""
        with patch('builtins.print') as mock_print:
            test(self.perfect_sims, mode="sinkhorn", batch_size=512)
        
        call_args = mock_print.call_args[0][0]
        # Check specific formatting matches original notebook
        assert "hits@1 :" in call_args
        assert "hits@10 :" in call_args  
        assert "MRR :" in call_args
        assert "%" in call_args
    
    def test_consistent_results_with_seed(self):
        """Test that results are consistent when using same random seed."""
        np.random.seed(42)
        with patch('builtins.print') as mock_print1:
            test(self.small_sims, mode="sinkhorn", batch_size=512)
        result1 = mock_print1.call_args[0][0]
        
        np.random.seed(42)  
        with patch('builtins.print') as mock_print2:
            test(self.small_sims, mode="sinkhorn", batch_size=512)
        result2 = mock_print2.call_args[0][0]
        
        # Results should be identical with same seed
        assert result1 == result2