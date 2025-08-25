"""
SEU Solvers Module

This module contains essential entity alignment solvers.
"""

from typing import Tuple
import numpy as np
from scipy import optimize
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def hungarian_solve(similarity_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve entity alignment using Hungarian algorithm.
    
    Args:
        similarity_matrix: Matrix of entity similarities (numpy array or tf.Tensor)
        
    Returns:
        Tuple of (row_indices, col_indices) from Hungarian algorithm
    """
    # Convert to numpy if it's a tensor
    if hasattr(similarity_matrix, 'numpy'):
        similarity_matrix = similarity_matrix.numpy()
        
    return optimize.linear_sum_assignment(similarity_matrix, maximize=True)


def sinkhorn_solve(similarity_matrix, temperature: float = 50.0, max_iterations: int = 10) -> np.ndarray:
    """
    Solve entity alignment using Sinkhorn operations.
    
    Exact implementation from main.ipynb cell 14.
    
    Args:
        similarity_matrix: Matrix of entity similarities (numpy array or tf.Tensor)
        temperature: Temperature parameter for scaling (default: 50.0)
        max_iterations: Number of iterations (default: 10)
        
    Returns:
        Doubly stochastic matrix from Sinkhorn iterations
    """
    if HAS_TF:
        # TensorFlow implementation (exact from notebook)
        # Convert to tensor only if it's not already a tensor
        if hasattr(similarity_matrix, 'numpy'):
            # Already a tensor
            sims = similarity_matrix
        else:
            sims = tf.constant(similarity_matrix, dtype=tf.float32)
            
        sims = tf.exp(sims * temperature)
        
        for k in range(max_iterations):
            sims = sims / tf.reduce_sum(sims, axis=1, keepdims=True)
            sims = sims / tf.reduce_sum(sims, axis=0, keepdims=True)
        
        return sims.numpy()
    else:
        # NumPy fallback
        # Convert to numpy if it's a tensor
        if hasattr(similarity_matrix, 'numpy'):
            similarity_matrix = similarity_matrix.numpy()
            
        sims = np.exp(similarity_matrix * temperature)
        
        for k in range(max_iterations):
            sims = sims / np.sum(sims, axis=1, keepdims=True)
            sims = sims / np.sum(sims, axis=0, keepdims=True)
        
        return sims