"""
SEU Features Module

This module handles feature generation for entity alignment.
Extracted directly from main.ipynb cells 8-11.
"""

from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf


def generate_bigram_dictionary(entity_names: List[Tuple[int, List[str]]]) -> Dict[str, int]:
    """
    Generate bigram dictionary from entity names.
    
    Extracted from main.ipynb cell 8.
    """
    d = {}
    count = 0
    for _, name in entity_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1
    return d


def generate_features(entity_names: List[Tuple[int, List[str]]],
                     word_vectors: Dict[str, np.ndarray],
                     node_size: int,
                     mode: str = 'hybrid-level') -> np.ndarray:
    """
    Generate entity features based on main.ipynb cell 9.
    
    Args:
        entity_names: List of (entity_id, name_words) pairs
        word_vectors: Dictionary of word vectors
        node_size: Total number of nodes
        mode: Feature mode ('word-level', 'char-level', 'hybrid-level')
    
    Returns:
        Feature matrix normalized with tf.nn.l2_normalize
    """
    # Generate bigram dictionary for character features
    if mode in ['char-level', 'hybrid-level']:
        d = generate_bigram_dictionary(entity_names)
    
    # Generate word-level features (from cell 9)
    ent_vec = np.zeros((node_size, 300))
    for i, name in entity_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vectors:
                ent_vec[i] += word_vectors[word]
                k += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
    
    # Generate character-level features if needed
    if mode in ['char-level', 'hybrid-level']:
        char_vec = np.zeros((node_size, len(d)))
        for i, name in entity_names:
            for word in name:
                word = word.lower()
                for idx in range(len(word)-1):
                    if word[idx:idx+2] in d:
                        char_vec[i, d[word[idx:idx+2]]] += 1
            if np.sum(char_vec[i]) == 0:
                char_vec[i] = np.random.random(len(d)) - 0.5
            char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])
    
    # Combine features based on mode (from cell 11)
    if mode == "word-level":
        feature = ent_vec
    elif mode == "char-level":
        feature = char_vec
    else:  # hybrid-level
        feature = np.concatenate([ent_vec, char_vec], -1)
    
    # L2 normalize as in original
    feature = tf.nn.l2_normalize(feature, axis=-1)
    return feature.numpy() if hasattr(feature, 'numpy') else feature