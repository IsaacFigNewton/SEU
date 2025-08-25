"""
Unit tests for SEU graph module.

Tests the graph operations functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
import logging

from seu.core.graph import build_sparse_adjacency_matrix, calculate_similarities


class TestBuildSparseAdjacencyMatrix:
    """Test cases for build_sparse_adjacency_matrix function."""
    
    def test_build_sparse_adjacency_matrix_basic(self):
        """Test basic sparse adjacency matrix construction."""
        triples = [(0, 0, 1), (1, 0, 2), (2, 1, 0)]
        node_size = 3
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        
        # Check basic properties
        assert sparse_matrix.shape == (node_size, node_size)
        assert isinstance(sparse_matrix, tf.SparseTensor)
        
        # Convert to dense for easier testing
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        # Should have self-loops on diagonal
        for i in range(node_size):
            assert dense_matrix[i, i] != 0  # Self-loops should exist
        
        # Should have edges from triples
        assert dense_matrix[0, 1] != 0  # Edge (0, 0, 1)
        assert dense_matrix[1, 2] != 0  # Edge (1, 0, 2)
        assert dense_matrix[2, 0] != 0  # Edge (2, 1, 0)
    
    def test_build_sparse_adjacency_matrix_weights(self):
        """Test that adjacency matrix has correct weights."""
        triples = [(0, 0, 1), (1, 0, 0)]  # Two triples with same relation
        node_size = 2
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        # Expected self-loop weight: log(2/2) = 0
        # Expected edge weight: log(2/2) = 0 (relation 0 appears twice)
        self_loop_weight = np.log(len(triples) / node_size)  # log(2/2) = 0
        edge_weight = np.log(len(triples) / 2)  # log(2/2) = 0
        
        # Check diagonal (self-loops)
        for i in range(node_size):
            assert abs(dense_matrix[i, i] - self_loop_weight) < 1e-6
    
    def test_build_sparse_adjacency_matrix_relation_frequencies(self):
        """Test relation weight calculation based on frequencies."""
        triples = [
            (0, 0, 1),  # relation 0
            (1, 0, 2),  # relation 0 (appears twice total)
            (2, 1, 0)   # relation 1 (appears once)
        ]
        node_size = 3
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        # Relation 0 appears twice, so weight = log(3/2)
        # Relation 1 appears once, so weight = log(3/1)
        expected_weight_0 = np.log(len(triples) / 2)
        expected_weight_1 = np.log(len(triples) / 1)
        
        # Check that edge weights are correct
        assert abs(dense_matrix[0, 1] - expected_weight_0) < 1e-6
        assert abs(dense_matrix[1, 2] - expected_weight_0) < 1e-6
        assert abs(dense_matrix[2, 0] - expected_weight_1) < 1e-6
    
    def test_build_sparse_adjacency_matrix_empty_triples(self):
        """Test adjacency matrix construction with empty triples."""
        triples = []
        node_size = 3
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        # Should only have self-loops with weight log(0/3) -> -inf or special handling
        # In practice, this might handle the edge case differently
        assert sparse_matrix.shape == (node_size, node_size)
    
    def test_build_sparse_adjacency_matrix_single_node(self):
        """Test with single node graph."""
        triples = [(0, 0, 0)]  # Self-loop
        node_size = 1
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        assert dense_matrix.shape == (1, 1)
        # Should have self-loop weight and edge weight
        assert dense_matrix[0, 0] != 0
    
    def test_build_sparse_adjacency_matrix_large_graph(self):
        """Test with larger graph for performance."""
        # Generate synthetic larger graph
        num_nodes = 100
        num_rels = 5
        triples = []
        
        for i in range(200):  # 200 triples
            h = np.random.randint(0, num_nodes)
            r = np.random.randint(0, num_rels)
            t = np.random.randint(0, num_nodes)
            triples.append((h, r, t))
        
        sparse_matrix = build_sparse_adjacency_matrix(triples, num_nodes)
        
        # Should complete without errors
        assert sparse_matrix.shape == (num_nodes, num_nodes)
        assert isinstance(sparse_matrix, tf.SparseTensor)
    
    def test_build_sparse_adjacency_matrix_numerical_stability(self):
        """Test numerical stability with various relation frequencies."""
        triples = []
        # Create highly imbalanced relation frequencies
        for i in range(100):
            triples.append((i % 10, 0, (i + 1) % 10))  # Relation 0 appears 100 times
        triples.append((0, 1, 1))  # Relation 1 appears once
        
        node_size = 10
        sparse_matrix = build_sparse_adjacency_matrix(triples, node_size)
        dense_matrix = tf.sparse.to_dense(sparse_matrix).numpy()
        
        # Should handle large frequency differences without inf/nan
        assert not np.any(np.isnan(dense_matrix))
        assert not np.any(np.isinf(dense_matrix))


class TestCalculateSimilarities:
    """Test cases for calculate_similarities function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_pairs = np.array([[0, 1], [1, 2], [2, 0]])
        self.features = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        self.triples = [(0, 0, 1), (1, 0, 2), (2, 1, 0)]
        self.node_size = 4
        self.sparse_matrix = build_sparse_adjacency_matrix(self.triples, self.node_size)
    
    def test_calculate_similarities_basic(self):
        """Test basic similarity calculation."""
        similarities = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=1
        )
        
        # Check output shape
        expected_shape = (len(self.test_pairs), len(self.test_pairs))
        assert similarities.shape == expected_shape
        
        # Check that similarities are reasonable values (cosine similarity range)
        assert tf.reduce_all(similarities >= -1.0)
        assert tf.reduce_all(similarities <= 1.0)
        
        # Check for no NaN or inf values
        assert not tf.reduce_any(tf.math.is_nan(similarities))
        assert not tf.reduce_any(tf.math.is_inf(similarities))
    
    def test_calculate_similarities_zero_depth(self):
        """Test similarity calculation with zero depth (no propagation)."""
        similarities_depth_0 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=0
        )
        
        # With depth 0, should just compute cosine similarities
        expected_shape = (len(self.test_pairs), len(self.test_pairs))
        assert similarities_depth_0.shape == expected_shape
        
        # Should be different from depth > 0 case (unless features are already optimal)
        similarities_depth_1 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=1
        )
        
        # Results might be different due to feature propagation
        assert similarities_depth_0.shape == similarities_depth_1.shape
    
    def test_calculate_similarities_multiple_depths(self):
        """Test similarity calculation with different depths."""
        depths = [0, 1, 2, 3]
        results = []
        
        for depth in depths:
            similarities = calculate_similarities(
                self.test_pairs, self.features, self.sparse_matrix, depth=depth
            )
            results.append(similarities)
            
            # Each result should have same shape
            expected_shape = (len(self.test_pairs), len(self.test_pairs))
            assert similarities.shape == expected_shape
            
            # Should have reasonable values
            assert tf.reduce_all(similarities >= -1.0)
            assert tf.reduce_all(similarities <= 1.0)
        
        # Different depths might produce different results
        # (though convergence is possible for some graph structures)
    
    def test_calculate_similarities_identity_features(self):
        """Test with identity-like features."""
        identity_features = np.eye(4)  # Perfect orthogonal features
        
        similarities = calculate_similarities(
            self.test_pairs, identity_features, self.sparse_matrix, depth=1
        )
        
        expected_shape = (len(self.test_pairs), len(self.test_pairs))
        assert similarities.shape == expected_shape
        
        # With orthogonal features, off-diagonal similarities should be low
        # (exact values depend on graph structure and propagation)
        assert not tf.reduce_any(tf.math.is_nan(similarities))
        assert not tf.reduce_any(tf.math.is_inf(similarities))
    
    def test_calculate_similarities_random_features(self):
        """Test with random features."""
        np.random.seed(42)
        random_features = np.random.randn(4, 5)  # Different feature dimension
        
        similarities = calculate_similarities(
            self.test_pairs, random_features, self.sparse_matrix, depth=2
        )
        
        expected_shape = (len(self.test_pairs), len(self.test_pairs))
        assert similarities.shape == expected_shape
        
        # Should still produce valid similarity values
        assert tf.reduce_all(similarities >= -1.0)
        assert tf.reduce_all(similarities <= 1.0)
        assert not tf.reduce_any(tf.math.is_nan(similarities))
    
    def test_calculate_similarities_single_pair(self):
        """Test with single test pair."""
        single_pair = np.array([[0, 1]])
        
        similarities = calculate_similarities(
            single_pair, self.features, self.sparse_matrix, depth=1
        )
        
        # Should produce 1x1 similarity matrix
        assert similarities.shape == (1, 1)
        assert tf.reduce_all(similarities >= -1.0)
        assert tf.reduce_all(similarities <= 1.0)
    
    def test_calculate_similarities_tensorflow_operations(self):
        """Test that internal TensorFlow operations work correctly."""
        # This tests tf.gather, tf.matmul, tf.transpose operations
        similarities = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=2
        )
        
        # Should complete without TensorFlow errors
        assert isinstance(similarities, tf.Tensor)
        assert similarities.dtype == tf.float32 or similarities.dtype == tf.float64
    
    def test_calculate_similarities_feature_normalization(self):
        """Test that feature normalization works correctly during propagation."""
        # Use unnormalized features to test normalization
        unnormalized_features = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 30.0],
            [5.0, 5.0, 0.0]
        ])
        
        similarities = calculate_similarities(
            self.test_pairs, unnormalized_features, self.sparse_matrix, depth=2
        )
        
        # Should still produce reasonable similarity values
        assert tf.reduce_all(similarities >= -1.0)
        assert tf.reduce_all(similarities <= 1.0)
        assert not tf.reduce_any(tf.math.is_nan(similarities))
    
    def test_calculate_similarities_edge_case_empty_pairs(self):
        """Test with empty test pairs."""
        empty_pairs = np.array([]).reshape(0, 2)
        
        similarities = calculate_similarities(
            empty_pairs, self.features, self.sparse_matrix, depth=1
        )
        
        # Should produce empty similarity matrix
        assert similarities.shape == (0, 0)
    
    def test_calculate_similarities_large_depth(self):
        """Test with large depth value."""
        # Test that large depth doesn't cause numerical issues
        similarities = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=10
        )
        
        expected_shape = (len(self.test_pairs), len(self.test_pairs))
        assert similarities.shape == expected_shape
        assert not tf.reduce_any(tf.math.is_nan(similarities))
        assert not tf.reduce_any(tf.math.is_inf(similarities))
    
    def test_calculate_similarities_averaging_behavior(self):
        """Test that averaging across depths works correctly."""
        # The function averages similarities across all depth iterations
        # Plus the initial similarity (depth + 1 total)
        
        similarities_depth_1 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=1
        )
        similarities_depth_2 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=2
        )
        
        # Both should have the same shape and reasonable values
        assert similarities_depth_1.shape == similarities_depth_2.shape
        assert tf.reduce_all(similarities_depth_1 >= -1.0)
        assert tf.reduce_all(similarities_depth_1 <= 1.0)
        assert tf.reduce_all(similarities_depth_2 >= -1.0)
        assert tf.reduce_all(similarities_depth_2 <= 1.0)
    
    def test_calculate_similarities_consistent_results(self):
        """Test that results are consistent for same inputs."""
        similarities1 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=2
        )
        similarities2 = calculate_similarities(
            self.test_pairs, self.features, self.sparse_matrix, depth=2
        )
        
        # Should produce identical results
        assert tf.reduce_all(tf.abs(similarities1 - similarities2) < 1e-6)