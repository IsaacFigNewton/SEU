"""
Unit tests for SEU features module.

Tests the feature generation functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
from typing import Dict, List

from seu.core.features import generate_bigram_dictionary, generate_features


class TestBigramDictionary:
    """Test cases for generate_bigram_dictionary function."""
    
    def test_generate_bigram_dictionary_basic(self):
        """Test basic bigram dictionary generation."""
        entity_names = [
            (0, ["new", "york"]),
            (1, ["city", "town"]),
            (2, ["test"])
        ]
        
        bigram_dict = generate_bigram_dictionary(entity_names)
        
        # Check that it's a dictionary with string keys and int values
        assert isinstance(bigram_dict, dict)
        for key, value in bigram_dict.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert len(key) == 2  # Bigrams are 2-character sequences
        
        # Verify some expected bigrams are present
        expected_bigrams = ['ne', 'ew', 'yo', 'or', 'rk', 'ci', 'it', 'ty', 'ow', 'te', 'es', 'st']
        for bigram in expected_bigrams:
            assert bigram in bigram_dict
    
    def test_generate_bigram_dictionary_empty_input(self):
        """Test bigram dictionary generation with empty input."""
        bigram_dict = generate_bigram_dictionary([])
        assert bigram_dict == {}
    
    def test_generate_bigram_dictionary_case_insensitive(self):
        """Test that bigram generation is case insensitive."""
        entity_names = [
            (0, ["NEW", "york"]),
            (1, ["New", "YORK"])
        ]
        
        bigram_dict = generate_bigram_dictionary(entity_names)
        
        # Should have bigrams from lowercase versions
        assert 'ne' in bigram_dict  # from "new"
        assert 'yo' in bigram_dict  # from "york"
        # Should not have uppercase versions
        assert 'NE' not in bigram_dict
        assert 'YO' not in bigram_dict
    
    def test_generate_bigram_dictionary_short_words(self):
        """Test bigram generation with words shorter than 2 characters."""
        entity_names = [
            (0, ["a", "I"]),  # Single character words
            (1, ["to", "be"])  # Two character words
        ]
        
        bigram_dict = generate_bigram_dictionary(entity_names)
        
        # Should have bigrams from 2-character words only
        assert 'to' in bigram_dict
        assert 'be' in bigram_dict
        # Single character words don't produce bigrams
    
    def test_generate_bigram_dictionary_unique_indices(self):
        """Test that bigram indices are unique."""
        entity_names = [
            (0, ["hello", "world"]),
            (1, ["hello", "there"])  # "hello" appears twice
        ]
        
        bigram_dict = generate_bigram_dictionary(entity_names)
        
        # All values should be unique
        values = list(bigram_dict.values())
        assert len(values) == len(set(values))
        
        # Indices should start from 0 and be consecutive
        assert set(values) == set(range(len(values)))


class TestGenerateFeatures:
    """Test cases for generate_features function."""
    
    def setup_method(self):
        """Setup test data."""
        self.entity_names = [
            (0, ["new", "york", "city"]),
            (1, ["los", "angeles"]),
            (2, ["chicago"]),
            (3, ["houston"]),
            (4, ["unknown", "entity"])  # Words not in vocabulary
        ]
        
        # Simple word vectors for testing (300-dimensional to match expected size)
        self.word_vectors = {
            "new": np.random.randn(300),
            "york": np.random.randn(300),
            "city": np.random.randn(300),
            "los": np.random.randn(300),
            "angeles": np.random.randn(300),
            "chicago": np.random.randn(300),
            "houston": np.random.randn(300)
        }
        
        self.node_size = 5
    
    def test_generate_features_word_level(self):
        """Test word-level feature generation."""
        features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='word-level'
        )
        
        # Check output shape
        assert features.shape == (self.node_size, 300)  # Default embedding dimension
        
        # Check that features are normalized (from tf.nn.l2_normalize)
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        
        # Check that entities with known words have meaningful features
        # Entity 0: ["new", "york", "city"] - all words known
        assert not np.allclose(features[0], 0)
        
        # Entity 4: ["unknown", "entity"] - no words in vocabulary, should be random but normalized
        assert not np.allclose(features[4], 0)
        norm_4 = np.linalg.norm(features[4])
        assert abs(norm_4 - 1.0) < 1e-6
    
    def test_generate_features_char_level(self):
        """Test character-level feature generation."""
        features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='char-level'
        )
        
        # Check output shape - should have bigram dimensions
        assert features.shape[0] == self.node_size
        assert features.shape[1] > 0  # Should have some bigram dimensions
        
        # Check that features are normalized
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        
        # Check that entities have meaningful character features
        assert not np.allclose(features[0], 0)
    
    def test_generate_features_hybrid_level(self):
        """Test hybrid-level feature generation."""
        features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='hybrid-level'
        )
        
        # Should concatenate word and character features
        assert features.shape[0] == self.node_size
        assert features.shape[1] > 300  # Should be word_dim (300) + char_dim
        
        # Check that features are normalized
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    
    def test_generate_features_invalid_mode(self):
        """Test feature generation with invalid mode."""
        # The current implementation uses if/elif/else, so invalid mode goes to hybrid
        features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='invalid-mode'
        )
        
        # Should default to hybrid-level mode
        assert features.shape[0] == self.node_size
        assert features.shape[1] > 300  # Should be word_dim (300) + char_dim
    
    def test_generate_features_empty_word_vectors(self):
        """Test feature generation with empty word vectors."""
        # Should work but use random vectors for all entities
        features = generate_features(
            self.entity_names, {}, self.node_size, mode='word-level'
        )
        
        # Should still produce normalized vectors
        assert features.shape == (self.node_size, 300)
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    
    def test_generate_features_word_averaging(self):
        """Test that word vectors are properly averaged."""
        # Create simple test case
        simple_entity_names = [(0, ["test", "word"])]
        simple_word_vectors = {
            "test": np.array([1.0, 0.0, 0.0] * 100),  # 300-dim vector
            "word": np.array([0.0, 1.0, 0.0] * 100)   # 300-dim vector
        }
        
        features = generate_features(
            simple_entity_names, simple_word_vectors, 1, mode='word-level'
        )
        
        # The averaged vector before normalization would be [0.5, 0.5, 0.0, ...]
        # After normalization, it should still have the same relative proportions
        feature = features[0]
        
        # First two dimensions should be equal (averaged from [1,0] and [0,1])
        # Pattern repeats every 3 elements
        for i in range(0, 300, 3):
            assert abs(feature[i] - feature[i+1]) < 1e-6
            assert abs(feature[i+2]) < 1e-6  # Third element should be 0
    
    def test_generate_features_tensorflow_integration(self):
        """Test that TensorFlow operations work correctly."""
        # Test that the tf.nn.l2_normalize operation works
        features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='hybrid-level'
        )
        
        # Should return numpy array (converted from TensorFlow)
        assert isinstance(features, np.ndarray)
        assert not hasattr(features, 'numpy')  # Should not be a TensorFlow tensor
    
    def test_generate_features_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        entity_names_no_words = [(0, ["unknown1", "unknown2"])]
        
        # Generate features twice with same seed
        np.random.seed(42)
        features1 = generate_features(entity_names_no_words, {}, 1, mode='word-level')
        
        np.random.seed(42)
        features2 = generate_features(entity_names_no_words, {}, 1, mode='word-level')
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_generate_features_edge_case_empty_names(self):
        """Test with empty entity names."""
        empty_entity_names = [(0, [])]
        
        features = generate_features(empty_entity_names, self.word_vectors, 1, mode='word-level')
        
        # Should produce random normalized vector
        assert features.shape == (1, 300)
        norm = np.linalg.norm(features[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_generate_features_partial_vocabulary_coverage(self):
        """Test with partial vocabulary coverage."""
        entity_names = [(0, ["known", "unknown"])]
        word_vectors = {"known": np.array([1.0] * 300)}
        
        features = generate_features(entity_names, word_vectors, 1, mode='word-level')
        
        # Should average the known word with zeros (for unknown word)
        # then normalize
        assert features.shape == (1, 300)
        norm = np.linalg.norm(features[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_generate_features_consistent_dimensions(self):
        """Test that feature dimensions are consistent across modes."""
        word_features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='word-level'
        )
        char_features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='char-level'
        )
        hybrid_features = generate_features(
            self.entity_names, self.word_vectors, self.node_size, mode='hybrid-level'
        )
        
        # Check dimensions
        assert word_features.shape[0] == self.node_size
        assert char_features.shape[0] == self.node_size
        assert hybrid_features.shape[0] == self.node_size
        
        # Hybrid should be concatenation of word + char
        expected_hybrid_dim = word_features.shape[1] + char_features.shape[1]
        assert hybrid_features.shape[1] == expected_hybrid_dim