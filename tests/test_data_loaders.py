"""
Unit tests for SEU data loaders module.

Tests the KGDataLoader class and data loading functionality.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from seu.data.loaders import KGDataLoader


class TestKGDataLoader:
    """Test cases for KGDataLoader class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def sample_triples_data(self):
        """Sample triples data for testing."""
        return {
            'triples_1': ["0\t1\t2\n", "3\t4\t5\n", "6\t7\t8\n"],
            'triples_2': ["9\t10\t11\n", "12\t13\t14\n"]
        }
    
    @pytest.fixture
    def sample_aligned_pairs_data(self):
        """Sample aligned pairs data for testing."""
        return {
            'ref_ent_ids': ["0\t100\n", "1\t101\n", "2\t102\n", "3\t103\n", "4\t104\n"]
        }
    
    @pytest.fixture
    def sample_entity_names_data(self):
        """Sample entity names data for testing."""
        return [
            [0, ["entity", "zero"]],
            [1, ["entity", "one"]],
            [2, ["entity", "two"]]
        ]
    
    @pytest.fixture
    def kg_loader(self, temp_dir):
        """Create KGDataLoader instance with temporary directory."""
        return KGDataLoader(data_dir=str(temp_dir))
    
    def create_test_files(self, temp_dir, triples_data=None, aligned_data=None):
        """Helper to create test data files."""
        kg_dir = temp_dir / "test_kg"
        kg_dir.mkdir(exist_ok=True)
        
        if triples_data:
            # Create triples files
            with open(kg_dir / "triples_1", "w") as f:
                f.writelines(triples_data['triples_1'])
            with open(kg_dir / "triples_2", "w") as f:
                f.writelines(triples_data['triples_2'])
        
        if aligned_data:
            # Create aligned pairs file
            with open(kg_dir / "ref_ent_ids", "w") as f:
                f.writelines(aligned_data['ref_ent_ids'])
        
        return str(kg_dir)
    
    def test_init(self, temp_dir):
        """Test KGDataLoader initialization."""
        # Test with custom data directory
        loader = KGDataLoader(data_dir=str(temp_dir))
        assert loader.data_dir == temp_dir
        assert loader._triples is None
        assert loader._node_size is None
        assert loader._rel_size is None
        assert loader._aligned_pairs is None
        assert loader._entity_names is None
        
        # Test with default data directory
        loader_default = KGDataLoader()
        assert loader_default.data_dir == Path.cwd() / 'KGs'
    
    def test_load_triples_basic(self, kg_loader, temp_dir, sample_triples_data):
        """Test basic triple loading functionality."""
        kg_path = self.create_test_files(temp_dir, triples_data=sample_triples_data)
        
        # Test with reverse=True (default)
        all_triples, node_size, rel_size = kg_loader.load_triples(kg_path, reverse=True)
        
        # Check return types
        assert isinstance(all_triples, np.ndarray)
        assert isinstance(node_size, int)
        assert isinstance(rel_size, int)
        
        # Check shapes
        assert all_triples.shape[1] == 3  # Each triple has 3 elements
        
        # Check that reverse triples were added (rel_size should be doubled)
        original_triples = len(sample_triples_data['triples_1']) + len(sample_triples_data['triples_2'])
        assert rel_size == 28  # max relation ID (13) + 1, then doubled
        
        # Check node size calculation
        assert node_size == 15  # max entity ID (14) + 1
        
        # Verify data is stored for validation
        assert kg_loader._triples is not None
        assert kg_loader._node_size == node_size
        assert kg_loader._rel_size == rel_size
    
    def test_load_triples_no_reverse(self, kg_loader, temp_dir, sample_triples_data):
        """Test triple loading without reverse triples."""
        kg_path = self.create_test_files(temp_dir, triples_data=sample_triples_data)
        
        all_triples, node_size, rel_size = kg_loader.load_triples(kg_path, reverse=False)
        
        # Check that relation size is not doubled
        assert rel_size == 14  # max relation ID (13) + 1, not doubled
        assert kg_loader._rel_size == rel_size
    
    def test_load_triples_file_not_found(self, kg_loader):
        """Test triple loading with non-existent files."""
        with pytest.raises(FileNotFoundError):
            kg_loader.load_triples("non_existent_path")
    
    def test_load_triples_invalid_format(self, kg_loader, temp_dir):
        """Test triple loading with invalid file format."""
        kg_dir = temp_dir / "test_kg"
        kg_dir.mkdir(exist_ok=True)
        
        # Create invalid triples file
        with open(kg_dir / "triples_1", "w") as f:
            f.write("invalid\tformat\n")  # Missing third element
        with open(kg_dir / "triples_2", "w") as f:
            f.write("0\t1\t2\n")
        
        with pytest.raises(ValueError):
            kg_loader.load_triples(str(kg_dir))
    
    def test_load_aligned_pairs_basic(self, kg_loader, temp_dir, sample_aligned_pairs_data):
        """Test basic aligned pairs loading."""
        kg_path = self.create_test_files(temp_dir, aligned_data=sample_aligned_pairs_data)
        
        train_pairs, test_pairs = kg_loader.load_aligned_pairs(kg_path, ratio=0.3)
        
        # Check return types
        assert isinstance(train_pairs, np.ndarray)
        assert isinstance(test_pairs, np.ndarray)
        
        # Check shapes
        assert train_pairs.shape[1] == 2
        assert test_pairs.shape[1] == 2
        
        # Check split ratio (approximately)
        total_pairs = len(sample_aligned_pairs_data['ref_ent_ids'])
        expected_train = int(total_pairs * 0.3)
        assert len(train_pairs) == expected_train
        assert len(test_pairs) == total_pairs - expected_train
        
        # Verify data is stored for validation
        assert kg_loader._aligned_pairs is not None
        assert len(kg_loader._aligned_pairs) == total_pairs
    
    def test_load_aligned_pairs_with_sup(self, kg_loader, temp_dir):
        """Test aligned pairs loading with supplementary file."""
        kg_dir = temp_dir / "test_kg"
        kg_dir.mkdir(exist_ok=True)
        
        # Create ref_ent_ids and sup_ent_ids files
        with open(kg_dir / "ref_ent_ids", "w") as f:
            f.write("0\t100\n1\t101\n")
        with open(kg_dir / "sup_ent_ids", "w") as f:
            f.write("2\t102\n3\t103\n")
        
        train_pairs, test_pairs = kg_loader.load_aligned_pairs(str(kg_dir), ratio=0.5)
        
        # Total should include both ref and sup
        total_pairs = len(train_pairs) + len(test_pairs)
        assert total_pairs == 4  # 2 ref + 2 sup
    
    def test_load_aligned_pairs_file_not_found(self, kg_loader):
        """Test aligned pairs loading with non-existent files."""
        with pytest.raises(FileNotFoundError):
            kg_loader.load_aligned_pairs("non_existent_path")
    
    def test_load_entity_names_basic(self, kg_loader, sample_entity_names_data):
        """Test basic entity names loading."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_names_data, f)
            temp_file = f.name
        
        try:
            entity_names = kg_loader.load_entity_names(temp_file)
            
            # Check return type and content
            assert isinstance(entity_names, list)
            assert len(entity_names) == 3
            
            # Check structure
            for entity_id, name_words in entity_names:
                assert isinstance(entity_id, int)
                assert isinstance(name_words, list)
                assert all(isinstance(word, str) for word in name_words)
            
            # Verify data is stored for validation
            assert kg_loader._entity_names is not None
            assert len(kg_loader._entity_names) == 3
            
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_file_not_found(self, kg_loader):
        """Test entity names loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            kg_loader.load_entity_names("non_existent_file.json")
    
    def test_load_entity_names_invalid_json(self, kg_loader):
        """Test entity names loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                kg_loader.load_entity_names(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_entity_names_invalid_format(self, kg_loader):
        """Test entity names loading with invalid format."""
        invalid_data = [
            [0, ["valid", "entry"]],
            ["invalid_id", ["name"]],  # Invalid ID type
            [2, "invalid_name_format"]  # Invalid name format
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                kg_loader.load_entity_names(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_data_integrity_no_data(self, kg_loader):
        """Test data integrity validation with no data loaded."""
        result = kg_loader.validate_data_integrity()
        assert result is False
    
    def test_validate_data_integrity_triples_only(self, kg_loader, temp_dir, sample_triples_data):
        """Test data integrity validation with only triples loaded."""
        kg_path = self.create_test_files(temp_dir, triples_data=sample_triples_data)
        kg_loader.load_triples(kg_path)
        
        result = kg_loader.validate_data_integrity()
        assert result is True
    
    def test_validate_data_integrity_invalid_triples(self, kg_loader):
        """Test data integrity validation with invalid triples."""
        # Manually set invalid triples data
        kg_loader._triples = np.array([[-1, 0, 1], [2, 3, 4]])  # Negative ID
        kg_loader._node_size = 5
        kg_loader._rel_size = 4
        
        result = kg_loader.validate_data_integrity()
        assert result is False
    
    def test_validate_data_integrity_all_data(self, kg_loader, temp_dir, sample_triples_data, 
                                           sample_aligned_pairs_data, sample_entity_names_data):
        """Test data integrity validation with all data types loaded."""
        # Load triples
        kg_path = self.create_test_files(temp_dir, triples_data=sample_triples_data, 
                                       aligned_data=sample_aligned_pairs_data)
        kg_loader.load_triples(kg_path)
        kg_loader.load_aligned_pairs(kg_path)
        
        # Load entity names
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_entity_names_data, f)
            temp_file = f.name
        
        try:
            kg_loader.load_entity_names(temp_file)
            result = kg_loader.validate_data_integrity()
            assert result is True
        finally:
            os.unlink(temp_file)
    
    def test_validate_data_integrity_inconsistent_sizes(self, kg_loader):
        """Test data integrity validation with inconsistent sizes."""
        # Set inconsistent data
        kg_loader._triples = np.array([[0, 1, 2], [3, 4, 5]])
        kg_loader._node_size = 3  # Too small for the data
        kg_loader._rel_size = 6
        
        result = kg_loader.validate_data_integrity()
        assert result is False
    
    def test_validate_data_integrity_invalid_aligned_pairs(self, kg_loader):
        """Test data integrity validation with invalid aligned pairs."""
        kg_loader._aligned_pairs = np.array([[-1, 0], [1, 2]])  # Negative ID
        
        result = kg_loader.validate_data_integrity()
        assert result is False
    
    def test_validate_data_integrity_invalid_entity_names(self, kg_loader):
        """Test data integrity validation with invalid entity names."""
        kg_loader._entity_names = [(-1, ["invalid", "id"])]  # Negative ID
        
        result = kg_loader.validate_data_integrity()
        assert result is False
    
    @patch('logging.getLogger')
    def test_logging(self, mock_get_logger, kg_loader, temp_dir, sample_triples_data):
        """Test that appropriate log messages are generated."""
        mock_logger = mock_get_logger.return_value
        
        kg_path = self.create_test_files(temp_dir, triples_data=sample_triples_data)
        kg_loader.load_triples(kg_path)
        
        # Check that info logs were called
        assert mock_logger.info.call_count >= 2  # At least 2 info messages
    
    def test_numpy_random_seed_effect(self, kg_loader, temp_dir, sample_aligned_pairs_data):
        """Test that numpy random seed affects shuffling in aligned pairs."""
        kg_path = self.create_test_files(temp_dir, aligned_data=sample_aligned_pairs_data)
        
        # Set seed and load pairs
        np.random.seed(42)
        train1, test1 = kg_loader.load_aligned_pairs(kg_path, ratio=0.3)
        
        # Reset loader and set same seed
        kg_loader._aligned_pairs = None
        np.random.seed(42)
        train2, test2 = kg_loader.load_aligned_pairs(kg_path, ratio=0.3)
        
        # Results should be identical
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)