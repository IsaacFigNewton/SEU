"""
SEU (Entity Alignment) Package

A Python package implementing "From Alignment to Assignment: Frustratingly Simple 
but Effective Unsupervised Entity Alignment without Neural Networks".

This package provides tools for unsupervised entity alignment between knowledge graphs
using word embeddings and graph-based features.

Modules:
    core: Core functionality for embeddings, features, graph operations, solvers, and evaluation
    data: Data loading utilities for knowledge graph datasets
    utils: General utility functions and helpers
    config: Configuration management and settings
    cli: Command line interface for running experiments
"""

__version__ = "0.1.0"
__author__ = "SEU Development Team"
__email__ = ""
__description__ = "SEU (Entity Alignment) - Frustratingly Simple but Effective Unsupervised Entity Alignment without Neural Networks"

# Import core functions for easy access
from .core.embeddings import load_word_vectors, download_and_extract_file
from .core.features import generate_features
from .core.graph import build_sparse_adjacency_matrix, calculate_similarities
from .core.solvers import hungarian_solve, sinkhorn_solve
from .core.evaluation import test
from .data.loaders import load_triples, load_aligned_pair, load_entity_names

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    # Core functions
    "load_word_vectors",
    "download_and_extract_file",
    "generate_features", 
    "build_sparse_adjacency_matrix",
    "calculate_similarities",
    "hungarian_solve", 
    "sinkhorn_solve",
    "test",
    "load_triples",
    "load_aligned_pair", 
    "load_entity_names",
]