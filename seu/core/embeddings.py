"""
SEU Embeddings Module

This module handles GloVe embedding download and loading functionality.
It provides the EmbeddingLoader class for managing word embeddings used
in entity alignment.

Classes:
    EmbeddingLoader: Main class for downloading and loading embeddings
"""

from typing import Dict, Optional, Tuple, Any
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

# Try to import requests, otherwise use urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Always import urllib.request for fallback
import urllib.request

# Set up logging
logger = logging.getLogger(__name__)


def load_word_vectors(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load word vectors from file.
    
    Simplified version of the embedding loading functionality.
    
    Args:
        file_path: Path to word vectors file
        
    Returns:
        Dictionary mapping words to vectors
    """
    word_vecs = {}
    with open(file_path, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    return word_vecs


def download_and_extract_file(url: str, inner_path: str, dest_path: Optional[str] = None, chunk_size: int = 8192):
    """
    Download a zip from `url`, open it, and extract the file `inner_path` (path inside the zip).
    
    This function is extracted directly from the main.ipynb notebook (cell 2) to maintain
    exact compatibility with the original implementation.
    
    Args:
        url: URL to download zip from
        inner_path: Path of file inside the zip to extract
        dest_path: If None, returns the file bytes. If a directory, writes the inner file
                  into that directory using its basename. If a file path, writes to that
                  exact file path.
        chunk_size: Size of chunks for downloading
        
    Returns:
        The bytes if dest_path is None, otherwise returns the path to the written file.

    Raises:
        FileNotFoundError: If inner_path not found in the zip
        RuntimeError: For download / extraction problems
    """
    # 1) Download to a temporary file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)  # we'll open it normally
    try:
        logger.info(f"Downloading {url} to temporary file {tmp_path}")
        if HAS_REQUESTS:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
        else:
            # urllib fallback
            with urllib.request.urlopen(url, timeout=30) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

        # 2) Open the zip and extract single file safely
        with zipfile.ZipFile(tmp_path, "r") as z:
            # normalize names inside zip
            namelist = z.namelist()
            if inner_path not in namelist:
                # allow matching by basename if exact path not found
                matches = [n for n in namelist if os.path.basename(n) == os.path.basename(inner_path)]
                if len(matches) == 1:
                    inner_path = matches[0]
                elif len(matches) > 1:
                    raise FileNotFoundError(
                        f"Ambiguous inner filename '{inner_path}'. Multiple matches: {matches}"
                    )
                else:
                    raise FileNotFoundError(f"'{inner_path}' not found in zip (checked {len(namelist)} entries).")

            # read bytes from the zip (avoid zip-slip because we do not extract paths)
            with z.open(inner_path, "r") as inner_file:
                data = inner_file.read()

            if dest_path is None:
                return data

            # decide final output path
            if os.path.isdir(dest_path):
                out_path = os.path.join(dest_path, os.path.basename(inner_path))
            else:
                # if dest_path endswith a path separator treat as dir
                if dest_path.endswith(os.path.sep):
                    os.makedirs(dest_path, exist_ok=True)
                    out_path = os.path.join(dest_path, os.path.basename(inner_path))
                else:
                    out_path = dest_path
                    out_dir = os.path.dirname(out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)

            # write file atomically
            tmp_out_fd, tmp_out_path = tempfile.mkstemp(dir=os.path.dirname(out_path))
            try:
                with os.fdopen(tmp_out_fd, "wb") as wf:
                    wf.write(data)
                os.replace(tmp_out_path, out_path)
                logger.info(f"Successfully extracted {inner_path} to {out_path}")
            finally:
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)

            return out_path

    finally:
        # clean up downloaded zip
        if os.path.exists(tmp_path):
            os.remove(tmp_path)