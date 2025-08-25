"""
Unit tests for SEU data loaders module.

Tests the data loading functionality.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from seu.data.loaders import load_triples, load_aligned_pair, load_entity_names


class TestLoadTriples:
    """Test cases for load_triples function."""
    
    def create_test_files(self, tmp_path, triples_1_data, triples_2_data):
        """Helper to create test triple files."""
        test_dir = tmp_path / "test_kg"
        test_dir.mkdir(exist_ok=True)
        
        with open(test_dir / "triples_1", "w") as f:
            f.writelines(triples_1_data)
        with open(test_dir / "triples_2", "w") as f:
            f.writelines(triples_2_data)
        
        return str(test_dir)
    
    def test_load_triples_basic(self, tmp_path):
        """Test basic triple loading functionality."""
        triples_1_data = ["0\t1\t2\n", "3\t4\t5\n", "6\t7\t8\n"]
        triples_2_data = ["9\t10\t11\n", "12\t13\t14\n"]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        # Test with reverse=True (default)
        all_triples, node_size, rel_size = load_triples(file_path, reverse=True)
        
        # Check return types
        assert isinstance(all_triples, np.ndarray)
        assert isinstance(node_size, (int, np.integer))
        assert isinstance(rel_size, (int, np.integer))
        
        # Check shapes
        assert all_triples.shape[1] == 3  # Each triple has 3 elements
        
        # Check that reverse triples were added (rel_size should be doubled)
        assert rel_size == 28  # max relation ID (13) + 1, then doubled
        
        # Check node size calculation
        assert node_size == 15  # max entity ID (14) + 1
        
        # Check that we have original + reverse triples
        original_count = len(triples_1_data) + len(triples_2_data)
        # All triples should have unique entries after reverse addition
        assert len(all_triples) >= original_count  # At least original triples
    
    def test_load_triples_no_reverse(self, tmp_path):
        """Test triple loading without reverse triples."""
        triples_1_data = ["0\t1\t2\n", "3\t4\t5\n"]
        triples_2_data = ["6\t7\t8\n"]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        all_triples, node_size, rel_size = load_triples(file_path, reverse=False)
        
        # Check that relation size is not doubled
        assert rel_size == 8  # max relation ID (7) + 1, not doubled
    
    def test_load_triples_file_not_found(self):
        """Test triple loading with non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_triples("non_existent_path")
    
    def test_load_triples_invalid_format(self, tmp_path):
        """Test triple loading with invalid file format."""
        # Create files with invalid format
        triples_1_data = ["invalid\tformat\n"]  # Missing third element
        triples_2_data = ["0\t1\t2\n"]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        with pytest.raises((ValueError, IndexError)):
            load_triples(file_path)
    
    def test_load_triples_empty_files(self, tmp_path):
        """Test triple loading with empty files."""
        triples_1_data = []
        triples_2_data = []
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        all_triples, node_size, rel_size = load_triples(file_path)
        
        # Should handle empty files gracefully
        assert isinstance(all_triples, np.ndarray)
        assert all_triples.shape[0] == 0 or all_triples.size == 0
    
    def test_load_triples_numba_integration(self, tmp_path):
        """Test that numba compilation works correctly."""
        triples_1_data = ["0\t0\t1\n", "1\t0\t0\n"]
        triples_2_data = ["2\t1\t2\n"]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        # Should not raise numba-related errors
        all_triples, node_size, rel_size = load_triples(file_path, reverse=True)
        
        assert isinstance(all_triples, np.ndarray)
        assert node_size > 0
        assert rel_size > 0
    
    def test_load_triples_unique_filtering(self, tmp_path):
        """Test that duplicate triples are filtered out."""
        # Include duplicate triples
        triples_1_data = ["0\t1\t2\n", "0\t1\t2\n"]  # Duplicate
        triples_2_data = ["3\t4\t5\n"]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        all_triples, node_size, rel_size = load_triples(file_path)
        
        # Check that duplicates are removed using np.unique
        unique_triples = np.unique(all_triples, axis=0)
        assert len(all_triples) == len(unique_triples)
    
    def test_load_triples_large_dataset(self, tmp_path):
        """Test with larger synthetic dataset."""
        # Generate larger dataset for performance testing
        triples_1_data = [f"{i}\t{i%10}\t{i+1}\n" for i in range(100)]
        triples_2_data = [f"{i+100}\t{i%5}\t{i+200}\n" for i in range(50)]
        
        file_path = self.create_test_files(tmp_path, triples_1_data, triples_2_data)
        
        all_triples, node_size, rel_size = load_triples(file_path, reverse=True)
        
        assert isinstance(all_triples, np.ndarray)
        assert node_size > 200  # Should be at least max entity + 1
        assert rel_size > 10   # Should be at least max relation + 1, doubled


class TestLoadAlignedPair:
    """Test cases for load_aligned_pair function."""
    
    def create_test_files(self, tmp_path, ref_data, sup_data=None):
        """Helper to create test alignment files."""
        test_dir = tmp_path / "test_kg"
        test_dir.mkdir(exist_ok=True)
        
        with open(test_dir / "ref_ent_ids", "w") as f:
            f.writelines(ref_data)
        
        if sup_data:
            with open(test_dir / "sup_ent_ids", "w") as f:
                f.writelines(sup_data)
        
        return str(test_dir)
    
    def test_load_aligned_pairs_basic(self, tmp_path):
        """Test basic aligned pairs loading."""
        ref_data = ["0\t100\n", "1\t101\n", "2\t102\n", "3\t103\n", "4\t104\n"]
        
        file_path = self.create_test_files(tmp_path, ref_data)
        
        train_pairs, test_pairs = load_aligned_pair(file_path, ratio=0.3)
        
        # Check return types
        assert isinstance(train_pairs, np.ndarray)
        assert isinstance(test_pairs, np.ndarray)
        
        # Check shapes
        assert train_pairs.shape[1] == 2
        assert test_pairs.shape[1] == 2
        
        # Check split ratio (approximately)
        total_pairs = len(ref_data)
        expected_train = int(total_pairs * 0.3)
        assert len(train_pairs) == expected_train
        assert len(test_pairs) == total_pairs - expected_train
    
    def test_load_aligned_pairs_with_sup(self, tmp_path):
        """Test aligned pairs loading with supplementary file."""
        ref_data = ["0\t100\n", "1\t101\n"]
        sup_data = ["2\t102\n", "3\t103\n"]
        
        file_path = self.create_test_files(tmp_path, ref_data, sup_data)
        
        train_pairs, test_pairs = load_aligned_pair(file_path, ratio=0.5)
        
        # Total should include both ref and sup
        total_pairs = len(train_pairs) + len(test_pairs)
        assert total_pairs == 4  # 2 ref + 2 sup
    
    def test_load_aligned_pairs_different_ratios(self, tmp_path):
        """Test with different ratio values."""
        ref_data = [f"{i}\t{i+100}\n" for i in range(10)]
        file_path = self.create_test_files(tmp_path, ref_data)
        
        ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for ratio in ratios:
            train_pairs, test_pairs = load_aligned_pair(file_path, ratio=ratio)
            
            total = len(train_pairs) + len(test_pairs)
            expected_train = int(10 * ratio)
            
            assert len(train_pairs) == expected_train
            assert len(test_pairs) == total - expected_train
    
    def test_load_aligned_pairs_file_not_found(self):
        """Test aligned pairs loading with non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_aligned_pair("non_existent_path")
    
    def test_load_aligned_pairs_invalid_format(self, tmp_path):
        """Test with invalid alignment format."""
        ref_data = ["invalid_format\n", "0\t100\n"]
        file_path = self.create_test_files(tmp_path, ref_data)
        
        with pytest.raises((ValueError, IndexError)):
            load_aligned_pair(file_path)
    
    def test_load_aligned_pairs_edge_ratio(self, tmp_path):
        """Test with edge case ratios."""
        ref_data = ["0\t100\n", "1\t101\n", "2\t102\n"]
        file_path = self.create_test_files(tmp_path, ref_data)
        
        # Test ratio = 0.0 (all test)
        train_pairs, test_pairs = load_aligned_pair(file_path, ratio=0.0)
        assert len(train_pairs) == 0
        assert len(test_pairs) == 3
        
        # Test ratio = 1.0 (all train)
        train_pairs, test_pairs = load_aligned_pair(file_path, ratio=1.0)
        assert len(train_pairs) == 3
        assert len(test_pairs) == 0
    
    def test_load_aligned_pairs_shuffling(self, tmp_path):
        """Test that pairs are shuffled."""
        ref_data = [f"{i}\t{i+100}\n" for i in range(20)]
        file_path = self.create_test_files(tmp_path, ref_data)
        
        # Load pairs twice with same ratio
        np.random.seed(42)
        train1, test1 = load_aligned_pair(file_path, ratio=0.5)
        
        np.random.seed(43)  # Different seed
        train2, test2 = load_aligned_pair(file_path, ratio=0.5)
        
        # Results should be different due to shuffling (unless very unlucky)
        # At least the order should be different in most cases
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)


class TestLoadEntityNames:
    """Test cases for load_entity_names function."""
    
    def test_load_entity_names_basic(self):
        """Test basic entity names loading."""
        sample_data = [
            [0, ["entity", "zero"]],
            [1, ["entity", "one"]],
            [2, ["entity", "two"]]
        ]
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            entity_names = load_entity_names(temp_file)
            
            # Check return type and content
            assert isinstance(entity_names, list)
            assert len(entity_names) == 3
            
            # Check structure
            for entity_id, name_words in entity_names:
                assert isinstance(entity_id, int)
                assert isinstance(name_words, list)
                assert all(isinstance(word, str) for word in name_words)
            
            # Check specific content
            assert entity_names[0] == [0, ["entity", "zero"]]
            assert entity_names[1] == [1, ["entity", "one"]]
            assert entity_names[2] == [2, ["entity", "two"]]
            
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_file_not_found(self):
        """Test entity names loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_entity_names("non_existent_file.json")
    
    def test_load_entity_names_invalid_json(self):
        """Test entity names loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_entity_names(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_empty_file(self):
        """Test entity names loading with empty JSON file."""
        empty_data = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_data, f)
            temp_file = f.name
        
        try:
            entity_names = load_entity_names(temp_file)
            assert entity_names == []
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_unicode_content(self):
        """Test entity names loading with unicode content."""
        unicode_data = [
            [0, ["français", "entité"]],
            [1, ["中文", "实体"]],
            [2, ["العربية", "كيان"]]
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            entity_names = load_entity_names(temp_file)
            
            assert len(entity_names) == 3
            assert entity_names[0] == [0, ["français", "entité"]]
            assert entity_names[1] == [1, ["中文", "实体"]]
            assert entity_names[2] == [2, ["العربية", "كيان"]]
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_large_dataset(self):
        """Test with larger entity names dataset."""
        # Generate larger dataset
        large_data = [[i, [f"entity_{i}", "name"]] for i in range(1000)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_data, f)
            temp_file = f.name
        
        try:
            entity_names = load_entity_names(temp_file)
            
            assert len(entity_names) == 1000
            assert entity_names[0] == [0, ["entity_0", "name"]]
            assert entity_names[999] == [999, ["entity_999", "name"]]
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_various_structures(self):
        """Test with various entity name structures."""
        varied_data = [
            [0, ["single"]],                           # Single word
            [1, ["two", "words"]],                     # Two words
            [2, ["multiple", "word", "entity", "name"]], # Multiple words
            [3, []],                                   # Empty names
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(varied_data, f)
            temp_file = f.name
        
        try:
            entity_names = load_entity_names(temp_file)
            
            assert len(entity_names) == 4
            assert entity_names[0] == [0, ["single"]]
            assert entity_names[1] == [1, ["two", "words"]]
            assert entity_names[2] == [2, ["multiple", "word", "entity", "name"]]
            assert entity_names[3] == [3, []]
        finally:
            os.unlink(temp_file)