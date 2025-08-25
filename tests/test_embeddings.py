"""
Unit tests for SEU embeddings module.

Tests the EmbeddingLoader class and embedding-related functionality.
"""

import pytest
import numpy as np
import os
import tempfile
import zipfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from seu.core.embeddings import EmbeddingLoader, download_and_extract_file
from tests.fixtures.mock_embeddings import (
    get_mock_word_vectors,
    get_mock_glove_file_content,
    get_mock_embedding_file_path,
    create_mock_zip_file
)


class TestEmbeddingLoader:
    """Test cases for EmbeddingLoader class."""
    
    def test_init_default_cache_dir(self):
        """Test EmbeddingLoader initialization with default cache directory."""
        loader = EmbeddingLoader()
        
        assert loader.cache_dir == Path.home() / '.seu_cache'
        assert loader.cache_dir.exists()
        assert loader._word_vectors is None
        assert loader._embedding_dimension is None
    
    def test_init_custom_cache_dir(self, tmp_path):
        """Test EmbeddingLoader initialization with custom cache directory."""
        custom_cache = tmp_path / "custom_cache"
        loader = EmbeddingLoader(cache_dir=str(custom_cache))
        
        assert loader.cache_dir == custom_cache
        assert loader.cache_dir.exists()
    
    @patch('seu.core.embeddings.download_and_extract_file')
    def test_download_glove_embeddings_success(self, mock_download):
        """Test successful GloVe embedding download."""
        loader = EmbeddingLoader()
        mock_download.return_value = "/path/to/glove.6B.300d.txt"
        
        result = loader.download_glove_embeddings(
            url="http://example.com/glove.zip",
            dimension=300,
            dest_path="/path/to/dest"
        )
        
        assert result == "/path/to/glove.6B.300d.txt"
        mock_download.assert_called_once_with(
            url="http://example.com/glove.zip",
            inner_path="glove.6B.300d.txt",
            dest_path="/path/to/dest"
        )
    
    def test_download_glove_embeddings_invalid_dimension(self):
        """Test GloVe download with invalid dimension."""
        loader = EmbeddingLoader()
        
        with pytest.raises(ValueError, match="Unsupported dimension: 42"):
            loader.download_glove_embeddings("http://example.com/glove.zip", 42, "/path/to/dest")
    
    def test_download_glove_embeddings_empty_url(self):
        """Test GloVe download with empty URL."""
        loader = EmbeddingLoader()
        
        with pytest.raises(ValueError, match="URL cannot be empty"):
            loader.download_glove_embeddings("", 300, "/path/to/dest")
    
    @patch('seu.core.embeddings.download_and_extract_file')
    def test_download_glove_embeddings_download_failure(self, mock_download):
        """Test handling of download failure."""
        loader = EmbeddingLoader()
        mock_download.side_effect = RuntimeError("Download failed")
        
        with pytest.raises(RuntimeError, match="Download failed"):
            loader.download_glove_embeddings("http://example.com/glove.zip", 300, "/path/to/dest")
    
    def test_load_word_vectors_success(self, tmp_path):
        """Test successful word vector loading."""
        loader = EmbeddingLoader()
        file_path = get_mock_embedding_file_path(tmp_path, dimension=50)
        
        word_vectors = loader.load_word_vectors(file_path)
        
        assert isinstance(word_vectors, dict)
        assert len(word_vectors) > 0
        assert loader._embedding_dimension == 50
        assert loader._word_vectors is word_vectors
        
        # Check that vectors are numpy arrays with correct dimension
        for word, vector in word_vectors.items():
            assert isinstance(vector, np.ndarray)
            assert vector.shape == (50,)
    
    def test_load_word_vectors_file_not_found(self):
        """Test word vector loading with non-existent file."""
        loader = EmbeddingLoader()
        
        with pytest.raises(FileNotFoundError, match="Embedding file not found"):
            loader.load_word_vectors("/nonexistent/file.txt")
    
    def test_load_word_vectors_directory_path(self, tmp_path):
        """Test word vector loading with directory path instead of file."""
        loader = EmbeddingLoader()
        
        with pytest.raises(ValueError, match="Path is not a file"):
            loader.load_word_vectors(str(tmp_path))
    
    def test_load_word_vectors_malformed_file(self, tmp_path):
        """Test word vector loading with malformed file content."""
        loader = EmbeddingLoader()
        malformed_file = tmp_path / "malformed.txt"
        malformed_file.write_text("word_without_vectors\n\n")
        
        with pytest.raises(RuntimeError, match="No valid word vectors found"):
            loader.load_word_vectors(str(malformed_file))
    
    def test_load_word_vectors_inconsistent_dimensions(self, tmp_path):
        """Test word vector loading with inconsistent dimensions."""
        loader = EmbeddingLoader()
        inconsistent_file = tmp_path / "inconsistent.txt"
        content = "word1 1.0 2.0 3.0\nword2 4.0 5.0\nword3 6.0 7.0 8.0\n"
        inconsistent_file.write_text(content)
        
        word_vectors = loader.load_word_vectors(str(inconsistent_file))
        
        # Should only load vectors with consistent dimension (3 in this case)
        assert len(word_vectors) == 2  # word1 and word3
        assert "word2" not in word_vectors
    
    def test_get_embedding_dimension_success(self, tmp_path):
        """Test getting embedding dimension after loading vectors."""
        loader = EmbeddingLoader()
        file_path = get_mock_embedding_file_path(tmp_path, dimension=100)
        loader.load_word_vectors(file_path)
        
        assert loader.get_embedding_dimension() == 100
    
    def test_get_embedding_dimension_no_vectors_loaded(self):
        """Test getting embedding dimension without loaded vectors."""
        loader = EmbeddingLoader()
        
        with pytest.raises(RuntimeError, match="No embeddings loaded"):
            loader.get_embedding_dimension()
    
    def test_get_word_vector_success(self, tmp_path):
        """Test getting vector for a specific word."""
        loader = EmbeddingLoader()
        file_path = get_mock_embedding_file_path(tmp_path)
        loader.load_word_vectors(file_path)
        
        # Should be able to get vectors for words from controlled vocabulary
        vector = loader.get_word_vector("the")
        assert vector is not None
        assert isinstance(vector, np.ndarray)
        
        # Should return None for non-existent word
        vector = loader.get_word_vector("nonexistent_word_xyz")
        assert vector is None
    
    def test_get_word_vector_no_vectors_loaded(self):
        """Test getting word vector without loaded vectors."""
        loader = EmbeddingLoader()
        
        with pytest.raises(RuntimeError, match="No embeddings loaded"):
            loader.get_word_vector("test")
    
    def test_get_vocabulary_size(self, tmp_path):
        """Test getting vocabulary size."""
        loader = EmbeddingLoader()
        file_path = get_mock_embedding_file_path(tmp_path, dimension=50)
        word_vectors = loader.load_word_vectors(file_path)
        
        assert loader.get_vocabulary_size() == len(word_vectors)
    
    def test_get_vocabulary_size_no_vectors_loaded(self):
        """Test getting vocabulary size without loaded vectors."""
        loader = EmbeddingLoader()
        
        with pytest.raises(RuntimeError, match="No embeddings loaded"):
            loader.get_vocabulary_size()
    
    def test_has_word(self, tmp_path):
        """Test checking if word exists in vocabulary."""
        loader = EmbeddingLoader()
        file_path = get_mock_embedding_file_path(tmp_path)
        loader.load_word_vectors(file_path)
        
        assert loader.has_word("the") is True
        assert loader.has_word("nonexistent_word_xyz") is False
    
    def test_has_word_no_vectors_loaded(self):
        """Test checking word existence without loaded vectors."""
        loader = EmbeddingLoader()
        
        with pytest.raises(RuntimeError, match="No embeddings loaded"):
            loader.has_word("test")


class TestDownloadAndExtractFile:
    """Test cases for download_and_extract_file function."""
    
    @patch('seu.core.embeddings.requests')
    def test_download_success_requests(self, mock_requests, tmp_path):
        """Test successful file download and extraction using requests."""
        # Create mock zip content
        zip_path = tmp_path / "test.zip"
        inner_filename = "test.txt"
        content = "test content"
        create_mock_zip_file(str(zip_path), inner_filename, content)
        
        # Mock requests response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_path.read_bytes()]
        mock_requests.get.return_value.__enter__.return_value = mock_response
        
        dest_path = tmp_path / "extracted.txt"
        result = download_and_extract_file(
            url="http://example.com/test.zip",
            inner_path=inner_filename,
            dest_path=str(dest_path)
        )
        
        assert result == str(dest_path)
        assert dest_path.read_text() == content
    
    @patch('seu.core.embeddings.HAS_REQUESTS', False)
    @patch('urllib.request')
    def test_download_success_urllib(self, mock_urllib, tmp_path):
        """Test successful file download using urllib fallback."""
        # Create mock zip content
        zip_path = tmp_path / "test.zip"
        inner_filename = "test.txt"
        content = "test content"
        create_mock_zip_file(str(zip_path), inner_filename, content)
        
        # Mock urllib response
        mock_response = MagicMock()
        mock_response.read.side_effect = [zip_path.read_bytes(), b'']
        mock_urllib.urlopen.return_value.__enter__.return_value = mock_response
        
        dest_path = tmp_path / "extracted.txt"
        result = download_and_extract_file(
            url="http://example.com/test.zip",
            inner_path=inner_filename,
            dest_path=str(dest_path)
        )
        
        assert result == str(dest_path)
        assert dest_path.read_text() == content
    
    def test_download_inner_file_not_found(self, tmp_path):
        """Test handling when inner file is not found in zip."""
        # Create zip without the requested file
        zip_path = tmp_path / "test.zip"
        create_mock_zip_file(str(zip_path), "other.txt", "content")
        
        with patch('seu.core.embeddings.requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [zip_path.read_bytes()]
            mock_requests.get.return_value.__enter__.return_value = mock_response
            
            with pytest.raises(FileNotFoundError, match="'nonexistent.txt' not found in zip"):
                download_and_extract_file(
                    url="http://example.com/test.zip",
                    inner_path="nonexistent.txt",
                    dest_path=str(tmp_path / "dest.txt")
                )
    
    def test_download_basename_matching(self, tmp_path):
        """Test that basename matching works for inner files."""
        # Create zip with file in subdirectory
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(str(zip_path), 'w') as zf:
            zf.writestr("subdir/test.txt", "content")
        
        with patch('seu.core.embeddings.requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [zip_path.read_bytes()]
            mock_requests.get.return_value.__enter__.return_value = mock_response
            
            dest_path = tmp_path / "extracted.txt"
            result = download_and_extract_file(
                url="http://example.com/test.zip",
                inner_path="test.txt",  # Just basename, not full path
                dest_path=str(dest_path)
            )
            
            assert result == str(dest_path)
            assert dest_path.read_text() == "content"
    
    @patch('seu.core.embeddings.requests')
    def test_download_to_directory(self, mock_requests, tmp_path):
        """Test extraction to directory path."""
        # Create mock zip content
        zip_path = tmp_path / "test.zip"
        inner_filename = "test.txt"
        content = "test content"
        create_mock_zip_file(str(zip_path), inner_filename, content)
        
        # Mock requests response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_path.read_bytes()]
        mock_requests.get.return_value.__enter__.return_value = mock_response
        
        dest_dir = tmp_path / "output"
        dest_dir.mkdir()
        
        result = download_and_extract_file(
            url="http://example.com/test.zip",
            inner_path=inner_filename,
            dest_path=str(dest_dir)
        )
        
        expected_path = dest_dir / inner_filename
        assert result == str(expected_path)
        assert expected_path.read_text() == content
    
    @patch('seu.core.embeddings.requests')
    def test_download_return_bytes(self, mock_requests, tmp_path):
        """Test returning bytes when dest_path is None."""
        # Create mock zip content
        zip_path = tmp_path / "test.zip"
        inner_filename = "test.txt"
        content = "test content"
        create_mock_zip_file(str(zip_path), inner_filename, content)
        
        # Mock requests response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_path.read_bytes()]
        mock_requests.get.return_value.__enter__.return_value = mock_response
        
        result = download_and_extract_file(
            url="http://example.com/test.zip",
            inner_path=inner_filename,
            dest_path=None
        )
        
        assert result == content.encode('utf-8')
    
    @patch('seu.core.embeddings.requests')
    def test_download_http_error(self, mock_requests):
        """Test handling of HTTP errors during download."""
        mock_requests.get.return_value.__enter__.return_value.raise_for_status.side_effect = \
            Exception("HTTP 404 Not Found")
        
        with pytest.raises(Exception, match="HTTP 404 Not Found"):
            download_and_extract_file(
                url="http://example.com/nonexistent.zip",
                inner_path="test.txt",
                dest_path="/tmp/test.txt"
            )
    
    def test_download_ambiguous_basename(self, tmp_path):
        """Test handling of ambiguous basename matches."""
        # Create zip with multiple files having same basename
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(str(zip_path), 'w') as zf:
            zf.writestr("dir1/test.txt", "content1")
            zf.writestr("dir2/test.txt", "content2")
        
        with patch('seu.core.embeddings.requests') as mock_requests:
            mock_response = MagicMock()
            mock_response.iter_content.return_value = [zip_path.read_bytes()]
            mock_requests.get.return_value.__enter__.return_value = mock_response
            
            with pytest.raises(FileNotFoundError, match="Ambiguous inner filename"):
                download_and_extract_file(
                    url="http://example.com/test.zip",
                    inner_path="test.txt",
                    dest_path=str(tmp_path / "dest.txt")
                )