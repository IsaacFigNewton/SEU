"""
SEU Graph Module

This module handles graph operations for entity alignment.
Extracted directly from main.ipynb cell 10 and 12.
"""

from typing import List, Tuple
import numpy as np
import tensorflow as tf


def build_sparse_adjacency_matrix(all_triples: List[Tuple[int, int, int]], 
                                 node_size: int) -> tf.SparseTensor:
    """
    Build sparse adjacency matrix from triples.
    
    Extracted from main.ipynb cell 10.
    """
    # Calculate relation frequencies
    dr = {}
    for x, r, y in all_triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1
    
    sparse_rel_matrix = []
    
    # Add self-loops
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])
    
    # Add edges with relation weights
    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])
    
    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
    sparse_rel_matrix = tf.SparseTensor(
        indices=sparse_rel_matrix[:, :2],
        values=sparse_rel_matrix[:, 2],
        dense_shape=(node_size, node_size)
    )
    
    return sparse_rel_matrix


def calculate_similarities(test_pair: np.ndarray,
                          feature: np.ndarray,
                          sparse_rel_matrix: tf.SparseTensor,
                          depth: int = 2) -> tf.Tensor:
    """
    Calculate similarity matrix with feature propagation.
    
    Extracted from main.ipynb cell 12.
    """
    def cal_sims(test_pair, feature):
        feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
        feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
        return tf.matmul(feature_a, tf.transpose(feature_b, [1, 0]))
    
    sims = cal_sims(test_pair, feature)
    
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        sims += cal_sims(test_pair, feature)
    
    sims /= depth + 1
    return sims