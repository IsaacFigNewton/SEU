"""
Unit tests for SEU embeddings module.

Tests the embedding loading functionality.
"""

import pytest
import numpy as np
import os
import tempfile
import zipfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from seu.core.embeddings import load_word_vectors, download_and_extract_file


class TestLoadWordVectors:
    """Test cases for load_word_vectors function."""
    
    def create_test_file(self, tmp_path, content):
        """Helper to create test embedding file."""
        test_file = tmp_path / "embeddings.txt"
        test_file.write_text(content)
        return str(test_file)
    
    def test_load_word_vectors_success(self, tmp_path):
        """Test successful word vector loading."""
        content = "hello 0.1 0.2 0.3\nworld 0.4 0.5 0.6\ntest 0.7 0.8 0.9\n"
        file_path = self.create_test_file(tmp_path, content)
        
        word_vectors = load_word_vectors(file_path)
        
        assert isinstance(word_vectors, dict)
        assert len(word_vectors) == 3
        assert "hello" in word_vectors
        assert "world" in word_vectors
        assert "test" in word_vectors
        
        # Check vector values
        np.testing.assert_array_almost_equal(word_vectors["hello"], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(word_vectors["world"], [0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(word_vectors["test"], [0.7, 0.8, 0.9])
    
    def test_load_word_vectors_file_not_found(self):
        """Test word vector loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_word_vectors("/nonexistent/file.txt")
    
    def test_load_word_vectors_empty_file(self, tmp_path):
        """Test word vector loading with empty file."""
        file_path = self.create_test_file(tmp_path, "")
        
        word_vectors = load_word_vectors(file_path)
        assert word_vectors == {}
    
    def test_load_word_vectors_malformed_lines(self, tmp_path):
        """Test word vector loading with some malformed lines."""
        content = "hello 0.1 0.2 0.3\nmalformed_line\nworld 0.4 0.5 0.6\n"
        file_path = self.create_test_file(tmp_path, content)
        
        # The current implementation creates empty arrays for malformed lines
        word_vectors = load_word_vectors(file_path)
        assert len(word_vectors) == 3  # All lines are processed
        assert "hello" in word_vectors
        assert "world" in word_vectors
        assert "malformed_line" in word_vectors
        
        # Check that proper vectors have values
        assert len(word_vectors["hello"]) == 3
        assert len(word_vectors["world"]) == 3
        # Malformed line gets empty array
        assert len(word_vectors["malformed_line"]) == 0
    
    def test_load_word_vectors_utf8_encoding(self, tmp_path):
        """Test word vector loading with UTF-8 characters."""
        content = "français 0.1 0.2 0.3\n中文 0.4 0.5 0.6\n"
        test_file = tmp_path / "embeddings.txt"
        # Write with explicit UTF-8 encoding to handle Unicode characters
        test_file.write_text(content, encoding='utf-8')
        file_path = str(test_file)
        
        word_vectors = load_word_vectors(file_path)
        assert len(word_vectors) == 2
        assert "français" in word_vectors
        assert "中文" in word_vectors


class TestDownloadAndExtractFile:
    """Test cases for download_and_extract_file function."""
    
    def create_mock_zip_file(self, zip_path, inner_filename, content):
        """Helper to create mock zip file."""
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr(inner_filename, content)
    
    @patch('seu.core.embeddings.requests')
    def test_download_success_requests(self, mock_requests, tmp_path):
        """Test successful file download and extraction using requests."""
        # Create mock zip content
        zip_path = tmp_path / "test.zip"
        inner_filename = "test.txt"
        content = "test content"
        self.create_mock_zip_file(str(zip_path), inner_filename, content)
        
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
        self.create_mock_zip_file(str(zip_path), inner_filename, content)
        
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
        self.create_mock_zip_file(str(zip_path), "other.txt", "content")
        
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
        self.create_mock_zip_file(str(zip_path), inner_filename, content)
        
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
        self.create_mock_zip_file(str(zip_path), inner_filename, content)
        
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