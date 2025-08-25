"""
Unit tests for SEU evaluation module.

Tests the Evaluator class and evaluation metrics functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from seu.core.evaluation import Evaluator


class TestEvaluator:
    """Test cases for Evaluator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = Evaluator(batch_size=512)
        
        # Create test similarity matrices
        self.small_sims = np.array([
            [1.0, 0.8, 0.3],
            [0.4, 0.9, 0.5],
            [0.2, 0.3, 1.0]
        ])
        
        # Perfect alignment case
        self.perfect_sims = np.eye(5)
        
        # Worst case alignment (reversed diagonal)
        self.worst_sims = np.flipud(np.eye(5))
        
        # Truly worst case - completely wrong similarities (4x4 to avoid center element)
        self.truly_worst_sims = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0]
        ])
        
        # Hungarian algorithm test result
        self.hungarian_result = (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))
        self.hungarian_result_bad = (np.array([0, 1, 2, 3, 4]), np.array([4, 3, 2, 1, 0]))
        self.hungarian_result_worst = (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]))
    
    def test_init(self):
        """Test Evaluator initialization."""
        evaluator = Evaluator()
        assert evaluator.batch_size == 1024  # default value
        
        evaluator_custom = Evaluator(batch_size=256)
        assert evaluator_custom.batch_size == 256
    
    def test_calculate_hits_at_k_sinkhorn_perfect_alignment(self):
        """Test hits@k calculation with perfect alignment using Sinkhorn mode."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.perfect_sims, k=[1, 5, 10], mode="sinkhorn"
        )
        
        # Perfect alignment should give 100% for all k values
        assert hits_result[1] == 100.0
        assert hits_result[5] == 100.0
        assert hits_result[10] == 100.0
    
    def test_calculate_hits_at_k_sinkhorn_worst_alignment(self):
        """Test hits@k calculation with worst case alignment using Sinkhorn mode."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.worst_sims, k=[1, 5, 10], mode="sinkhorn"
        )
        
        # Worst case (anti-diagonal) should have low hits@1 (20% for 5x5 matrix due to middle element)
        assert hits_result[1] == 20.0  # One correct match in center of anti-diagonal
        assert hits_result[5] >= hits_result[1]  # Should be better or equal
        assert hits_result[10] >= hits_result[5]  # Should be better or equal
    
    def test_calculate_hits_at_k_hungarian_perfect(self):
        """Test hits@k calculation with perfect Hungarian results."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.hungarian_result, k=[1, 5, 10], mode="hungarian"
        )
        
        # Perfect Hungarian alignment should give 100%
        assert hits_result[1] == 100.0
        assert hits_result[5] == 100.0  # Same as hits@1 for Hungarian
        assert hits_result[10] == 100.0  # Same as hits@1 for Hungarian
    
    def test_calculate_hits_at_k_hungarian_bad(self):
        """Test hits@k calculation with bad Hungarian results."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.hungarian_result_bad, k=[1, 5, 10], mode="hungarian"
        )
        
        # Bad Hungarian alignment (0->4, 1->3, 2->2, 3->1, 4->0) has 1 correct match (2->2)
        assert hits_result[1] == 20.0  # 1 out of 5 correct = 20%
        assert hits_result[5] == 20.0  # Same as hits@1 for Hungarian
        assert hits_result[10] == 20.0  # Same as hits@1 for Hungarian
    
    def test_calculate_hits_at_k_invalid_mode(self):
        """Test hits@k calculation with invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            self.evaluator.calculate_hits_at_k(self.perfect_sims, mode="invalid")
    
    def test_calculate_hits_at_k_invalid_hungarian_input(self):
        """Test hits@k calculation with invalid Hungarian input."""
        with pytest.raises(ValueError, match="Hungarian result must be a tuple"):
            self.evaluator.calculate_hits_at_k(self.perfect_sims, mode="hungarian")
    
    def test_calculate_mrr_sinkhorn_perfect(self):
        """Test MRR calculation with perfect alignment using Sinkhorn mode."""
        mrr_result = self.evaluator.calculate_mrr(self.perfect_sims, mode="sinkhorn")
        
        # Perfect alignment should give 100% MRR
        assert mrr_result == 100.0
    
    def test_calculate_mrr_sinkhorn_worst(self):
        """Test MRR calculation with worst alignment using Sinkhorn mode."""
        mrr_result = self.evaluator.calculate_mrr(self.worst_sims, mode="sinkhorn")
        
        # Worst case should have low MRR (but not necessarily 0 due to ranking)
        assert 0.0 <= mrr_result <= 100.0
        assert mrr_result < 50.0  # Should be significantly lower than random
    
    def test_calculate_mrr_hungarian_perfect(self):
        """Test MRR calculation with perfect Hungarian results."""
        mrr_result = self.evaluator.calculate_mrr(self.hungarian_result, mode="hungarian")
        
        # Perfect Hungarian should give 100% MRR
        assert mrr_result == 100.0
    
    def test_calculate_mrr_hungarian_bad(self):
        """Test MRR calculation with bad Hungarian results."""
        mrr_result = self.evaluator.calculate_mrr(self.hungarian_result_bad, mode="hungarian")
        
        # Bad Hungarian should give same as hits@1 (20% due to 1 correct match out of 5)
        assert mrr_result == 20.0
    
    def test_calculate_hits_at_k_hungarian_worst(self):
        """Test hits@k calculation with worst Hungarian results (no correct matches)."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.hungarian_result_worst, k=[1, 5, 10], mode="hungarian"
        )
        
        # Worst Hungarian alignment (0->1, 1->2, 2->3, 3->0) has no correct matches
        assert hits_result[1] == 0.0  # 0 out of 4 correct = 0%
        assert hits_result[5] == 0.0  # Same as hits@1 for Hungarian
        assert hits_result[10] == 0.0  # Same as hits@1 for Hungarian
    
    def test_calculate_hits_at_k_sinkhorn_truly_worst(self):
        """Test hits@k calculation with truly worst case using Sinkhorn mode."""
        hits_result = self.evaluator.calculate_hits_at_k(
            self.truly_worst_sims, k=[1, 5, 10], mode="sinkhorn"
        )
        
        # Truly worst case should have 0% hits@1 (no diagonal elements)
        assert hits_result[1] == 0.0  # No correct top-1 matches
        assert hits_result[5] >= hits_result[1]  # Should be better or equal
        assert hits_result[10] >= hits_result[5]  # Should be better or equal
    
    def test_calculate_mrr_invalid_mode(self):
        """Test MRR calculation with invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            self.evaluator.calculate_mrr(self.perfect_sims, mode="invalid")
    
    def test_calculate_precision_recall_perfect(self):
        """Test precision and recall calculation with perfect alignment."""
        precision, recall = self.evaluator.calculate_precision_recall(
            self.perfect_sims, threshold=0.9
        )
        
        # Perfect alignment with high threshold should give 100% precision and recall
        assert precision == 100.0
        assert recall == 100.0
    
    def test_calculate_precision_recall_zero_threshold(self):
        """Test precision and recall with zero threshold."""
        precision, recall = self.evaluator.calculate_precision_recall(
            self.perfect_sims, threshold=0.0
        )
        
        # With threshold 0, all similarities are considered positive
        # This should result in lower precision but high recall
        assert 0.0 <= precision <= 100.0
        assert recall == 100.0  # All true positives are found
    
    def test_calculate_precision_recall_high_threshold(self):
        """Test precision and recall with very high threshold."""
        precision, recall = self.evaluator.calculate_precision_recall(
            self.perfect_sims, threshold=1.5  # Higher than any similarity
        )
        
        # With impossibly high threshold, no predictions are positive
        assert precision == 0.0  # No true positives
        assert recall == 0.0     # No true positives
    
    def test_generate_evaluation_report_complete(self):
        """Test evaluation report generation with complete results."""
        results = {
            "dataset": "dbp_ja_en",
            "mode": "sinkhorn",
            "total_entities": 5000,
            "hits_at_k": {1: 95.5, 5: 98.2, 10: 99.1},
            "mrr": 96.8,
            "precision": 94.2,
            "recall": 93.8,
            "execution_time": 45.2,
            "memory_usage": 512.3
        }
        
        report = self.evaluator.generate_evaluation_report(results)
        
        # Check that all components are included
        assert "SEU Entity Alignment Evaluation Report" in report
        assert "dbp_ja_en" in report
        assert "sinkhorn" in report
        assert "5000" in report
        assert "95.50%" in report  # hits@1
        assert "98.20%" in report  # hits@5
        assert "99.10%" in report  # hits@10
        assert "96.80%" in report  # MRR
        assert "94.20%" in report  # precision
        assert "93.80%" in report  # recall
        assert "45.20 seconds" in report
        assert "512.30 MB" in report
        assert "F1-Score" in report  # F1 should be calculated
    
    def test_generate_evaluation_report_minimal(self):
        """Test evaluation report generation with minimal results."""
        results = {
            "mode": "hungarian",
            "hits_at_k": {1: 85.0}
        }
        
        report = self.evaluator.generate_evaluation_report(results)
        
        assert "SEU Entity Alignment Evaluation Report" in report
        assert "hungarian" in report
        assert "85.00%" in report
    
    def test_generate_evaluation_report_empty(self):
        """Test evaluation report generation with empty results."""
        results = {}
        
        report = self.evaluator.generate_evaluation_report(results)
        
        # Should still generate basic report structure
        assert "SEU Entity Alignment Evaluation Report" in report
        assert "Report generated by SEU Evaluation Module" in report
    
    def test_test_method_sinkhorn(self):
        """Test the comprehensive test method with Sinkhorn mode."""
        # Capture stdout to test verbose output
        with patch('builtins.print') as mock_print:
            results = self.evaluator.test(self.perfect_sims, mode="sinkhorn", verbose=True)
        
        # Check that print was called with expected format
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "hits@10" in call_args
        assert "MRR" in call_args
        
        # Check results structure
        assert "mode" in results
        assert "hits_at_k" in results
        assert "mrr" in results
        assert "precision" in results
        assert "recall" in results
        
        assert results["mode"] == "sinkhorn"
        assert 1 in results["hits_at_k"]
        assert 10 in results["hits_at_k"]
    
    def test_test_method_hungarian(self):
        """Test the comprehensive test method with Hungarian mode."""
        with patch('builtins.print') as mock_print:
            results = self.evaluator.test(self.hungarian_result, mode="hungarian", verbose=True)
        
        # Check that print was called with expected format (only hits@1 for Hungarian)
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "hits@10" not in call_args  # Hungarian only shows hits@1
        assert "MRR" not in call_args      # Hungarian only shows hits@1
        
        # Check results structure
        assert results["mode"] == "hungarian"
        assert "hits_at_k" in results
        assert "mrr" in results
        assert "precision" not in results  # Not calculated for Hungarian
        assert "recall" not in results     # Not calculated for Hungarian
    
    def test_test_method_non_verbose(self):
        """Test the test method with verbose=False."""
        with patch('builtins.print') as mock_print:
            results = self.evaluator.test(self.perfect_sims, mode="sinkhorn", verbose=False)
        
        # Should not call print
        mock_print.assert_not_called()
        
        # Should still return results
        assert "mode" in results
        assert "hits_at_k" in results
    
    def test_test_method_custom_batch_size(self):
        """Test the test method with custom batch size."""
        original_batch_size = self.evaluator.batch_size
        
        results = self.evaluator.test(
            self.perfect_sims, mode="sinkhorn", batch_size=128, verbose=False
        )
        
        # Batch size should be restored after execution
        assert self.evaluator.batch_size == original_batch_size
        
        # Should still return valid results
        assert "mode" in results
        assert "hits_at_k" in results
    
    def test_batch_processing_large_similarity_matrix(self):
        """Test batch processing with a large similarity matrix."""
        # Create a larger matrix to test batch processing
        large_sims = np.random.rand(2000, 2000)
        # Make it somewhat diagonal for reasonable results
        large_sims += 2 * np.eye(2000)
        
        evaluator_small_batch = Evaluator(batch_size=100)
        
        # This should not raise an error and should process in batches
        results = evaluator_small_batch.calculate_hits_at_k(
            large_sims, k=[1, 5], mode="sinkhorn"
        )
        
        assert isinstance(results, dict)
        assert 1 in results
        assert 5 in results
        assert 0.0 <= results[1] <= 100.0
        assert 0.0 <= results[5] <= 100.0
    
    def test_edge_case_single_entity(self):
        """Test evaluation with single entity."""
        single_sim = np.array([[1.0]])
        
        hits_result = self.evaluator.calculate_hits_at_k(single_sim, k=[1], mode="sinkhorn")
        mrr_result = self.evaluator.calculate_mrr(single_sim, mode="sinkhorn")
        precision, recall = self.evaluator.calculate_precision_recall(single_sim)
        
        # Single entity should always be correctly aligned
        assert hits_result[1] == 100.0
        assert mrr_result == 100.0
        assert precision == 100.0
        assert recall == 100.0
    
    def test_edge_case_empty_similarity_matrix(self):
        """Test evaluation with empty similarity matrix."""
        empty_sims = np.array([]).reshape(0, 0)
        
        hits_result = self.evaluator.calculate_hits_at_k(empty_sims, k=[1], mode="sinkhorn")
        
        # Empty matrix should return 0% hits (no entities to align)
        assert hits_result[1] == 0.0


class TestBackwardCompatibilityFunction:
    """Test cases for the backward compatibility via Evaluator.test method."""
    
    def test_backward_compatibility_sinkhorn(self):
        """Test backward compatibility function with Sinkhorn mode."""
        sims = np.eye(3)
        evaluator = Evaluator(batch_size=512)
        
        with patch('builtins.print') as mock_print:
            evaluator.test(sims, mode="sinkhorn", verbose=True)
        
        # Should print results in original format
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "hits@10" in call_args
        assert "MRR" in call_args
    
    def test_backward_compatibility_hungarian(self):
        """Test backward compatibility function with Hungarian mode."""
        result = (np.array([0, 1, 2]), np.array([0, 1, 2]))
        evaluator = Evaluator()
        
        with patch('builtins.print') as mock_print:
            evaluator.test(result, mode="hungarian", verbose=True)
        
        # Should print results in original format (only hits@1)
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "hits@1" in call_args
        assert "100.00%" in call_args


class TestIntegrationWithTensorFlow:
    """Test integration with TensorFlow operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = Evaluator()
    
    def test_tensorflow_integration(self):
        """Test that TensorFlow operations work correctly."""
        # Create similarity matrix using TensorFlow operations
        tf_sims = tf.constant([[1.0, 0.5, 0.2], [0.3, 0.9, 0.4], [0.1, 0.3, 1.0]])
        np_sims = tf_sims.numpy()
        
        # Should work with numpy array converted from TensorFlow
        results = self.evaluator.calculate_hits_at_k(np_sims, k=[1], mode="sinkhorn")
        
        assert isinstance(results, dict)
        assert 1 in results
    
    def test_tensorflow_operations_in_calculation(self):
        """Test that internal TensorFlow operations work correctly."""
        # This tests the internal tf.argsort and tf.where operations
        sims = np.array([[1.0, 0.5], [0.3, 1.0]])
        
        # Should not raise any TensorFlow-related errors
        results = self.evaluator.calculate_hits_at_k(sims, k=[1, 2], mode="sinkhorn")
        mrr = self.evaluator.calculate_mrr(sims, mode="sinkhorn")
        
        assert isinstance(results, dict)
        assert isinstance(mrr, float)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = Evaluator()
    
    def test_invalid_similarity_shape(self):
        """Test behavior with invalid similarity matrix shapes."""
        # Non-square matrix should still work for some operations
        non_square = np.random.rand(3, 5)
        
        # This should work as long as we have enough entities to process
        with pytest.raises(Exception):  # Might raise various exceptions depending on TF operations
            self.evaluator.calculate_hits_at_k(non_square, k=[1], mode="sinkhorn")
    
    def test_negative_similarities(self):
        """Test behavior with negative similarities."""
        negative_sims = np.array([[-1.0, -0.5, -0.8], [-0.3, -0.1, -0.9], [-0.7, -0.6, -0.2]])
        
        # Should still work (rankings are relative)
        results = self.evaluator.calculate_hits_at_k(negative_sims, k=[1], mode="sinkhorn")
        assert isinstance(results, dict)
        assert 0.0 <= results[1] <= 100.0
    
    def test_nan_similarities(self):
        """Test behavior with NaN similarities."""
        nan_sims = np.array([[1.0, np.nan, 0.5], [0.3, 1.0, np.nan], [np.nan, 0.2, 1.0]])
        
        # This might raise an error or produce invalid results
        # The exact behavior depends on TensorFlow's handling of NaN values
        try:
            results = self.evaluator.calculate_hits_at_k(nan_sims, k=[1], mode="sinkhorn")
            # If it doesn't raise an error, results should be a dict
            assert isinstance(results, dict)
        except Exception:
            # It's acceptable for this to raise an error
            pass
    
    def test_zero_batch_size(self):
        """Test error handling with zero batch size."""
        with pytest.raises(Exception):  # Should raise some kind of error
            evaluator = Evaluator(batch_size=0)
            evaluator.calculate_hits_at_k(np.eye(5), k=[1], mode="sinkhorn")
    
    def test_very_large_k_values(self):
        """Test behavior with k values larger than matrix size."""
        small_sims = np.eye(3)
        
        # Should handle gracefully (k=10 when only 3 entities exist)
        results = self.evaluator.calculate_hits_at_k(small_sims, k=[1, 10, 100], mode="sinkhorn")
        
        assert isinstance(results, dict)
        assert results[10] >= results[1]  # hits@10 should be >= hits@1
        assert results[100] >= results[10]  # hits@100 should be >= hits@10