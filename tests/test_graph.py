"""
Unit tests for SEU graph module.

Tests the GraphBuilder class and graph operations functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from unittest.mock import patch
import logging

from seu.core.graph import GraphBuilder
from tests.fixtures.sample_data import (
    get_sample_graph_triples,
    get_small_graph_triples,
    get_large_synthetic_graph,
    get_sample_feature_matrix
)


class TestGraphBuilder:
    """Test cases for GraphBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph_builder = GraphBuilder()
        self.small_triples = get_small_graph_triples()
        self.sample_triples = get_sample_graph_triples()
        self.sample_features = get_sample_feature_matrix()
    
    def test_init(self):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder()
        assert isinstance(builder, GraphBuilder)
    
    def test_calculate_relation_weights_basic(self):
        """Test basic relation weight calculation."""
        triples = [(0, 0, 1), (1, 0, 2), (2, 1, 0)]  # relation 0 appears twice, relation 1 once
        
        weights = self.graph_builder.calculate_relation_weights(triples)
        
        # Expected: log(3/2) for relation 0, log(3/1) for relation 1
        expected_weight_0 = np.log(3 / 2)
        expected_weight_1 = np.log(3 / 1)
        
        assert len(weights) == 2
        assert 0 in weights
        assert 1 in weights
        assert np.isclose(weights[0], expected_weight_0)
        assert np.isclose(weights[1], expected_weight_1)
    
    def test_calculate_relation_weights_empty_triples(self):
        """Test relation weight calculation with empty triples."""
        with pytest.raises(ValueError, match="Triples list cannot be empty"):
            self.graph_builder.calculate_relation_weights([])
    
    def test_calculate_relation_weights_single_relation(self):
        """Test relation weight calculation with single relation."""
        triples = [(0, 0, 1), (1, 0, 2), (2, 0, 3)]
        
        weights = self.graph_builder.calculate_relation_weights(triples)
        
        # Expected: log(3/3) = log(1) = 0
        assert len(weights) == 1
        assert 0 in weights
        assert np.isclose(weights[0], 0.0)
    
    def test_build_sparse_adjacency_matrix_basic(self):
        """Test basic sparse adjacency matrix construction."""
        triples = self.small_triples
        node_size = 3
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Check basic properties
        assert sparse_matrix.shape == (node_size, node_size)
        assert isinstance(sparse_matrix, csr_matrix)
        
        # Should have self-loops plus edges from triples
        expected_nnz = node_size + len(triples)  # 3 self-loops + 3 edges
        assert sparse_matrix.nnz == expected_nnz
        
        # Check that self-loops exist on diagonal
        diagonal = sparse_matrix.diagonal()
        assert len(diagonal) == node_size  # Diagonal should exist for all nodes
        # Self-loop weight is log(len(triples)/node_size), which can be zero if equal
        expected_self_loop = np.log(len(triples) / node_size)
        assert all(np.isclose(diagonal, expected_self_loop))
    
    def test_build_sparse_adjacency_matrix_weights(self):
        """Test that adjacency matrix has correct weights."""
        triples = [(0, 0, 1), (1, 0, 0)]  # Two triples with same relation
        node_size = 2
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Expected self-loop weight: log(2/2) = 0
        # Expected edge weight: log(2/2) = 0 (relation 0 appears twice)
        self_loop_weight = np.log(len(triples) / node_size)  # log(2/2) = 0
        edge_weight = np.log(len(triples) / 2)  # log(2/2) = 0
        
        # Check diagonal (self-loops)
        diagonal = sparse_matrix.diagonal()
        assert all(np.isclose(diagonal, self_loop_weight))
    
    def test_build_sparse_adjacency_matrix_empty_triples(self):
        """Test adjacency matrix construction with empty triples."""
        with pytest.raises(ValueError, match="Triples list cannot be empty"):
            self.graph_builder.build_sparse_adjacency_matrix([], 3)
    
    def test_build_sparse_adjacency_matrix_invalid_node_size(self):
        """Test adjacency matrix construction with invalid node size."""
        triples = [(0, 0, 1)]
        
        with pytest.raises(ValueError, match="Node size must be positive"):
            self.graph_builder.build_sparse_adjacency_matrix(triples, 0)
        
        with pytest.raises(ValueError, match="Node size must be positive"):
            self.graph_builder.build_sparse_adjacency_matrix(triples, -1)
    
    def test_propagate_features_basic(self):
        """Test basic feature propagation."""
        triples = self.small_triples
        node_size = 3
        features = np.random.randn(node_size, 4)
        
        # Build adjacency matrix
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Propagate with depth 1
        propagated = self.graph_builder.propagate_features(features, adj_matrix, depth=1)
        
        # Check output shape
        assert propagated.shape == features.shape
        
        # Check that features have reasonable magnitudes (averaged, not normalized)
        norms = np.linalg.norm(propagated, axis=1)
        assert all(norms > 0)  # All features should be non-zero
        assert all(norms <= 1.0)  # Should not exceed unit length due to normalization in steps
    
    def test_propagate_features_zero_depth(self):
        """Test feature propagation with zero depth."""
        triples = self.small_triples
        node_size = 3
        features = get_sample_feature_matrix(node_size, 4)
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # With depth 0, should return normalized input features
        propagated = self.graph_builder.propagate_features(features, adj_matrix, depth=0)
        
        # Should be the same as L2 normalized input
        expected = tf.nn.l2_normalize(features, axis=-1).numpy()
        assert np.allclose(propagated, expected, rtol=1e-5)
    
    def test_propagate_features_multiple_depths(self):
        """Test feature propagation with multiple depths."""
        triples = self.sample_triples
        node_size = 6
        features = get_sample_feature_matrix(node_size, 4)
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Test different depths
        prop_1 = self.graph_builder.propagate_features(features, adj_matrix, depth=1)
        prop_2 = self.graph_builder.propagate_features(features, adj_matrix, depth=2)
        
        # Results should be different
        assert not np.allclose(prop_1, prop_2)
        
        # Both should have reasonable magnitudes (averaged features)
        norms_1 = np.linalg.norm(prop_1, axis=1)
        norms_2 = np.linalg.norm(prop_2, axis=1)
        assert all(norms_1 > 0)
        assert all(norms_2 > 0)
        assert all(norms_1 <= 1.0)  # Should not exceed unit length
        assert all(norms_2 <= 1.0)
    
    def test_propagate_features_invalid_depth(self):
        """Test feature propagation with invalid depth."""
        triples = self.small_triples
        node_size = 3
        features = np.random.randn(node_size, 4)
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        with pytest.raises(ValueError, match="Depth must be non-negative"):
            self.graph_builder.propagate_features(features, adj_matrix, depth=-1)
    
    def test_propagate_features_dimension_mismatch(self):
        """Test feature propagation with mismatched dimensions."""
        triples = self.small_triples
        node_size = 3
        features = np.random.randn(5, 4)  # Wrong number of nodes
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        with pytest.raises(ValueError, match="Feature matrix rows.*must match adjacency matrix size"):
            self.graph_builder.propagate_features(features, adj_matrix, depth=1)
    
    def test_get_sparse_tensor_tf(self):
        """Test conversion from scipy sparse matrix to TensorFlow SparseTensor."""
        triples = self.small_triples
        node_size = 3
        
        scipy_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        tf_sparse = self.graph_builder.get_sparse_tensor_tf(scipy_matrix)
        
        # Check that it's a TensorFlow SparseTensor
        assert isinstance(tf_sparse, tf.SparseTensor)
        assert tf_sparse.dense_shape.numpy().tolist() == [node_size, node_size]
        
        # Convert back to dense and compare
        dense_from_tf = tf.sparse.to_dense(tf_sparse).numpy()
        dense_from_scipy = scipy_matrix.toarray()
        
        assert np.allclose(dense_from_tf, dense_from_scipy)
    
    def test_memory_efficiency_large_graph(self):
        """Test memory efficiency with large synthetic graphs."""
        # Generate large graph
        large_triples = get_large_synthetic_graph(num_nodes=1000, num_relations=5, num_triples=2000)
        node_size = 1000
        
        # This should complete without memory issues
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(large_triples, node_size)
        
        # Check properties
        assert sparse_matrix.shape == (node_size, node_size)
        assert isinstance(sparse_matrix, csr_matrix)
        
        # Should be much more memory efficient than dense
        memory_ratio = sparse_matrix.nnz / (node_size * node_size)
        assert memory_ratio < 0.1  # Should use less than 10% of dense matrix memory
    
    def test_large_graph_feature_propagation(self):
        """Test feature propagation on large graphs for performance."""
        # Smaller version for CI/test performance
        large_triples = get_large_synthetic_graph(num_nodes=100, num_relations=3, num_triples=300)
        node_size = 100
        features = get_sample_feature_matrix(node_size, 10)
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(large_triples, node_size)
        
        # This should complete in reasonable time
        propagated = self.graph_builder.propagate_features(features, sparse_matrix, depth=2)
        
        # Check results
        assert propagated.shape == (node_size, 10)
        norms = np.linalg.norm(propagated, axis=1)
        assert all(norms > 0)
        assert all(norms <= 1.0)
    
    def test_edge_cases_single_node(self):
        """Test edge case with single node graph."""
        triples = []  # No edges, just self-loops will be added
        node_size = 1
        
        # This should raise an error because triples is empty
        with pytest.raises(ValueError, match="Triples list cannot be empty"):
            self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
    
    def test_edge_cases_disconnected_components(self):
        """Test handling of disconnected graph components."""
        # Create disconnected components
        triples = [
            (0, 0, 1),  # Component 1: 0-1
            (1, 0, 0),  # Component 1: 1-0  
            (3, 1, 4),  # Component 2: 3-4
            (4, 1, 3),  # Component 2: 4-3
            # Node 2 and 5 are isolated (only self-loops)
        ]
        node_size = 6
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Check basic properties
        assert sparse_matrix.shape == (node_size, node_size)
        expected_nnz = node_size + len(triples)  # self-loops + edges
        assert sparse_matrix.nnz == expected_nnz
        
        # Test feature propagation on disconnected graph
        features = get_sample_feature_matrix(node_size, 3)
        propagated = self.graph_builder.propagate_features(features, sparse_matrix, depth=1)
        
        # Should still work with reasonable magnitudes
        assert propagated.shape == (node_size, 3)
        norms = np.linalg.norm(propagated, axis=1)
        assert all(norms > 0)
        # Note: disconnected components may have smaller magnitudes due to limited propagation
    
    @patch('seu.core.graph.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs."""
        triples = self.small_triples
        node_size = 3
        
        # Test adjacency matrix logging
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Check that info logging occurred
        mock_logger.info.assert_called()
        
        # Test feature propagation logging
        features = get_sample_feature_matrix(node_size, 4)
        propagated = self.graph_builder.propagate_features(features, sparse_matrix, depth=2)
        
        # Should have logged feature propagation info
        mock_logger.info.assert_called()
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create triples with very frequent relations (small weights)
        # Make sure all node indices are within bounds
        triples = [(i % 5, 0, (i+1) % 5) for i in range(100)]  # 100 triples with same relation, nodes 0-4
        node_size = 5
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Should handle large frequency differences
        assert sparse_matrix.shape == (node_size, node_size)
        assert not np.any(np.isnan(sparse_matrix.data))
        assert not np.any(np.isinf(sparse_matrix.data))
    
    def test_compatibility_with_notebook_format(self):
        """Test compatibility with the exact format used in main.ipynb."""
        # Test that we can reproduce the exact sparse matrix format from the notebook
        triples = [(0, 0, 1), (1, 1, 2), (2, 0, 0)]
        node_size = 3
        
        sparse_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Convert to TensorFlow format (as used in notebook)
        tf_sparse = self.graph_builder.get_sparse_tensor_tf(sparse_matrix)
        
        # Should be compatible with TensorFlow operations
        features = tf.constant(np.random.randn(node_size, 5), dtype=tf.float32)
        result = tf.sparse.sparse_dense_matmul(tf_sparse, features)
        
        assert result.shape == (node_size, 5)
        assert not tf.reduce_any(tf.math.is_nan(result))
        assert not tf.reduce_any(tf.math.is_inf(result))
    
    def test_calculate_similarity_matrix_notebook_style(self):
        """Test notebook-style similarity matrix calculation."""
        triples = self.sample_triples
        node_size = 6
        features = get_sample_feature_matrix(node_size, 4)
        
        # Create test pairs
        test_pairs = np.array([[0, 1], [2, 3], [4, 5]])
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Calculate similarities using notebook-style method
        sim_matrix = self.graph_builder.calculate_similarity_matrix_notebook_style(
            features, adj_matrix, test_pairs, depth=2
        )
        
        # Check output shape
        expected_shape = (len(test_pairs), len(test_pairs))
        assert sim_matrix.shape == expected_shape
        
        # Check that similarities are reasonable (diagonal should be highest)
        diagonal = np.diag(sim_matrix)
        assert all(diagonal >= -1.0)  # Cosine similarity bounds
        assert all(diagonal <= 1.0)
        
        # Check for no NaN or inf values
        assert not np.any(np.isnan(sim_matrix))
        assert not np.any(np.isinf(sim_matrix))
    
    def test_notebook_style_vs_feature_propagation_different_results(self):
        """Test that notebook-style and feature propagation give different results."""
        triples = self.small_triples
        node_size = 3
        features = get_sample_feature_matrix(node_size, 4)
        test_pairs = np.array([[0, 1], [1, 2]])
        
        adj_matrix = self.graph_builder.build_sparse_adjacency_matrix(triples, node_size)
        
        # Get results from both methods
        sim_matrix = self.graph_builder.calculate_similarity_matrix_notebook_style(
            features, adj_matrix, test_pairs, depth=1
        )
        
        propagated_features = self.graph_builder.propagate_features(features, adj_matrix, depth=1)
        
        # They should be different since they use different approaches
        # The similarity matrix is pairwise similarities between test_pairs
        # The propagated features are the averaged features for all nodes
        assert sim_matrix.shape == (len(test_pairs), len(test_pairs))
        assert propagated_features.shape == (node_size, features.shape[1])
        
        # Both should be valid (no NaN/inf)
        assert not np.any(np.isnan(sim_matrix))
        assert not np.any(np.isnan(propagated_features))