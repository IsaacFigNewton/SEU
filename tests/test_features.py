"""
Unit tests for SEU features module.

Tests the FeatureGenerator class and feature generation functionality.
"""

import pytest
import numpy as np
from typing import Dict, List

from seu.core.features import FeatureGenerator
from tests.fixtures.sample_data import (
    get_sample_entity_names_formatted,
    get_sample_word_vectors
)


class TestFeatureGenerator:
    """Test cases for FeatureGenerator class."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        self.feature_generator = FeatureGenerator()
        self.entity_names = get_sample_entity_names_formatted()
        self.word_vectors = get_sample_word_vectors()
        self.node_size = 5
        self.vector_dim = 3
        
    def test_init(self):
        """Test FeatureGenerator initialization."""
        fg = FeatureGenerator()
        assert fg.bigram_dict is None
        assert isinstance(fg, FeatureGenerator)
    
    def test_generate_bigram_dictionary_basic(self):
        """Test basic bigram dictionary generation."""
        bigram_dict = self.feature_generator.generate_bigram_dictionary(self.entity_names)
        
        # Check that it's a dictionary with string keys and int values
        assert isinstance(bigram_dict, dict)
        for key, value in bigram_dict.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert len(key) == 2  # Bigrams are 2-character sequences
        
        # Check that the dictionary is stored in the instance
        assert self.feature_generator.bigram_dict == bigram_dict
        
        # Verify some expected bigrams are present
        expected_bigrams = ['ne', 'ew', 'yo', 'or', 'rk', 'ci', 'it', 'ty']
        for bigram in expected_bigrams:
            assert bigram in bigram_dict
    
    def test_generate_bigram_dictionary_empty_input(self):
        """Test bigram dictionary generation with empty input."""
        bigram_dict = self.feature_generator.generate_bigram_dictionary([])
        assert bigram_dict == {}
        
    def test_generate_bigram_dictionary_malformed_input(self):
        """Test bigram dictionary generation with malformed input."""
        malformed_input = [
            (0, "not_a_list"),  # Should be list of words
            (1, ["valid", "words"]),
            (2, [123, "mixed_types"]),  # Non-string word
        ]
        
        # Should not crash, but should log warnings
        bigram_dict = self.feature_generator.generate_bigram_dictionary(malformed_input)
        assert isinstance(bigram_dict, dict)
        # Should still process the valid entry
        assert 'va' in bigram_dict  # from "valid"
        assert 'or' in bigram_dict  # from "words"
    
    def test_generate_word_features_basic(self):
        """Test basic word feature generation."""
        word_features = self.feature_generator.generate_word_features(
            self.entity_names, self.word_vectors, self.node_size, self.vector_dim
        )
        
        # Check output shape
        assert word_features.shape == (self.node_size, self.vector_dim)
        
        # Check that features are L2 normalized
        for i in range(word_features.shape[0]):
            norm = np.linalg.norm(word_features[i])
            assert abs(norm - 1.0) < 1e-6, f"Feature {i} not normalized: {norm}"
        
        # Check that entities with known words have meaningful features
        # Entity 0: ["New", "York", "City"] - all words known
        assert not np.allclose(word_features[0], 0)
    
    def test_generate_word_features_missing_words(self):
        """Test word feature generation with missing words in vocabulary."""
        # Use entity names with words not in vocabulary
        entity_names_missing = [(0, ["unknown", "words"])]
        
        word_features = self.feature_generator.generate_word_features(
            entity_names_missing, self.word_vectors, 1, self.vector_dim
        )
        
        # Should still produce a normalized vector (random)
        assert word_features.shape == (1, self.vector_dim)
        norm = np.linalg.norm(word_features[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_generate_word_features_empty_input(self):
        """Test word feature generation with empty input."""
        with pytest.raises(ValueError, match="Empty entity_names list provided"):
            self.feature_generator.generate_word_features([], self.word_vectors, self.node_size)
    
    def test_generate_word_features_empty_word_vectors(self):
        """Test word feature generation with empty word vectors."""
        # Should work but use random vectors for all entities
        word_features = self.feature_generator.generate_word_features(
            self.entity_names, {}, self.node_size, self.vector_dim
        )
        
        # Should still produce normalized vectors
        assert word_features.shape == (self.node_size, self.vector_dim)
        for i in range(word_features.shape[0]):
            norm = np.linalg.norm(word_features[i])
            assert abs(norm - 1.0) < 1e-6
    
    def test_generate_char_features_basic(self):
        """Test basic character feature generation."""
        # First generate bigram dictionary
        bigram_dict = self.feature_generator.generate_bigram_dictionary(self.entity_names)
        
        char_features = self.feature_generator.generate_char_features(
            self.entity_names, bigram_dict, self.node_size
        )
        
        # Check output shape
        assert char_features.shape == (self.node_size, len(bigram_dict))
        
        # Check that features are L2 normalized
        for i in range(char_features.shape[0]):
            norm = np.linalg.norm(char_features[i])
            assert abs(norm - 1.0) < 1e-6, f"Feature {i} not normalized: {norm}"
        
        # Check that entities have meaningful character features
        assert not np.allclose(char_features[0], 0)
    
    def test_generate_char_features_empty_bigram_dict(self):
        """Test character feature generation with empty bigram dictionary."""
        with pytest.raises(ValueError, match="Empty bigram_dict provided"):
            self.feature_generator.generate_char_features(self.entity_names, {}, self.node_size)
    
    def test_generate_char_features_empty_input(self):
        """Test character feature generation with empty input."""
        bigram_dict = {'ab': 0, 'cd': 1}
        with pytest.raises(ValueError, match="Empty entity_names list provided"):
            self.feature_generator.generate_char_features([], bigram_dict, self.node_size)
    
    def test_generate_hybrid_features_word_level(self):
        """Test hybrid feature generation in word-level mode."""
        features = self.feature_generator.generate_hybrid_features(
            self.entity_names, self.word_vectors, self.node_size, mode='word-level', vector_dim=self.vector_dim
        )
        
        # Should be same as word features
        assert features.shape == (self.node_size, self.vector_dim)
        
        # Check normalization
        for i in range(features.shape[0]):
            norm = np.linalg.norm(features[i])
            assert abs(norm - 1.0) < 1e-6
    
    def test_generate_hybrid_features_char_level(self):
        """Test hybrid feature generation in char-level mode."""
        features = self.feature_generator.generate_hybrid_features(
            self.entity_names, self.word_vectors, self.node_size, mode='char-level'
        )
        
        # Should have character feature dimensions
        assert features.shape[0] == self.node_size
        assert features.shape[1] > 0  # Should have some bigram dimensions
        
        # Check normalization
        for i in range(features.shape[0]):
            norm = np.linalg.norm(features[i])
            assert abs(norm - 1.0) < 1e-6
    
    def test_generate_hybrid_features_hybrid_level(self):
        """Test hybrid feature generation in hybrid-level mode."""
        features = self.feature_generator.generate_hybrid_features(
            self.entity_names, self.word_vectors, self.node_size, mode='hybrid-level', vector_dim=self.vector_dim
        )
        
        # Should concatenate word and character features
        assert features.shape[0] == self.node_size
        assert features.shape[1] > self.vector_dim  # Should be word_dim + char_dim
        
        # Check normalization
        for i in range(features.shape[0]):
            norm = np.linalg.norm(features[i])
            assert abs(norm - 1.0) < 1e-6
    
    def test_generate_hybrid_features_invalid_mode(self):
        """Test hybrid feature generation with invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode 'invalid-mode'"):
            self.feature_generator.generate_hybrid_features(
                self.entity_names, self.word_vectors, self.node_size, mode='invalid-mode'
            )
    
    def test_generate_hybrid_features_empty_input(self):
        """Test hybrid feature generation with empty input."""
        with pytest.raises(ValueError, match="Empty entity_names list provided"):
            self.feature_generator.generate_hybrid_features(
                [], self.word_vectors, self.node_size
            )
    
    def test_l2_normalize(self):
        """Test L2 normalization utility function."""
        # Create test matrix
        test_matrix = np.array([
            [3, 4],  # norm = 5
            [1, 0],  # norm = 1
            [0, 0],  # zero vector
        ])
        
        normalized = self.feature_generator._l2_normalize(test_matrix)
        
        # Check expected results
        expected = np.array([
            [3/5, 4/5],
            [1, 0],
            [0, 0],  # Zero vector stays zero
        ])
        
        np.testing.assert_allclose(normalized, expected, rtol=1e-6)
    
    def test_validate_entity_names_valid(self):
        """Test validation of correctly formatted entity names."""
        valid = self.feature_generator.validate_entity_names(self.entity_names)
        assert valid is True
    
    def test_validate_entity_names_invalid_format(self):
        """Test validation of incorrectly formatted entity names."""
        # Test various invalid formats
        invalid_cases = [
            "not_a_list",  # Not a list
            [(0, ["valid"], "extra")],  # Too many elements
            [("not_int", ["words"])],  # Non-int ID
            [(0, "not_list")],  # Words not a list
            [(0, [123, "mixed"])],  # Non-string words
        ]
        
        for invalid_input in invalid_cases:
            valid = self.feature_generator.validate_entity_names(invalid_input)
            assert valid is False
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible when numpy seed is set."""
        entity_names_no_words = [(0, ["unknown", "entity"])]
        
        # Generate features twice with same seed
        np.random.seed(42)
        features1 = self.feature_generator.generate_word_features(
            entity_names_no_words, {}, 1, 3
        )
        
        np.random.seed(42)
        features2 = self.feature_generator.generate_word_features(
            entity_names_no_words, {}, 1, 3
        )
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_bigram_case_insensitive(self):
        """Test that bigram generation is case insensitive."""
        entity_names_mixed_case = [
            (0, ["NEW", "york"]),
            (1, ["New", "YORK"])
        ]
        
        bigram_dict = self.feature_generator.generate_bigram_dictionary(entity_names_mixed_case)
        
        # Should have bigrams from lowercase versions
        assert 'ne' in bigram_dict  # from "new"
        assert 'yo' in bigram_dict  # from "york"
        # Should not have uppercase versions
        assert 'NE' not in bigram_dict
        assert 'YO' not in bigram_dict
    
    def test_feature_dimensions_consistency(self):
        """Test that feature dimensions are consistent across methods."""
        # Generate all types of features
        bigram_dict = self.feature_generator.generate_bigram_dictionary(self.entity_names)
        word_features = self.feature_generator.generate_word_features(
            self.entity_names, self.word_vectors, self.node_size, self.vector_dim
        )
        char_features = self.feature_generator.generate_char_features(
            self.entity_names, bigram_dict, self.node_size
        )
        hybrid_features = self.feature_generator.generate_hybrid_features(
            self.entity_names, self.word_vectors, self.node_size, 'hybrid-level', self.vector_dim
        )
        
        # Check dimensions
        assert word_features.shape[0] == self.node_size
        assert char_features.shape[0] == self.node_size
        assert hybrid_features.shape[0] == self.node_size
        
        # Hybrid should be concatenation of word + char
        assert hybrid_features.shape[1] == word_features.shape[1] + char_features.shape[1]
    
    def test_memory_efficiency_large_entity_set(self):
        """Test memory efficiency with a larger entity set."""
        # Create larger entity set
        large_entity_names = [(i, [f"entity_{i}", "name"]) for i in range(100)]
        
        # Should handle without issues
        bigram_dict = self.feature_generator.generate_bigram_dictionary(large_entity_names)
        features = self.feature_generator.generate_hybrid_features(
            large_entity_names, self.word_vectors, 100, 'hybrid-level'
        )
        
        assert features.shape[0] == 100
        assert isinstance(bigram_dict, dict)
        assert len(bigram_dict) > 0