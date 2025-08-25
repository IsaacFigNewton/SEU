"""
SEU Data Loaders Module

This module contains utilities for loading knowledge graph datasets.
Functions extracted from utils.py.
"""

import numba as nb
import numpy as np
import os
import json
from typing import List, Tuple


def load_triples(file_path: str, reverse: bool = True) -> Tuple[np.ndarray, int, int]:
    """
    Load triples from files and optionally add reverse triples.
    
    Extracted from utils.py.
    """
    @nb.njit
    def reverse_triples(triples, rel_size):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i, 0] = triples[i, 2]
            reversed_triples[i, 2] = triples[i, 0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples
    
    with open(file_path + "/triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "/triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n", "").split("\t") for line in triples1 + triples2 if line.strip()]).astype(np.int64)
    
    if len(triples) == 0:
        return np.array([]), 0, 0
        
    node_size = max([np.max(triples[:, 0]), np.max(triples[:, 2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1
    
    all_triples = np.concatenate([triples, reverse_triples(triples, rel_size)], axis=0)
    all_triples = np.unique(all_triples, axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size


def load_aligned_pair(file_path: str, ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load aligned entity pairs from files.
    
    Extracted from utils.py.
    """
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "/ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "/ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "/sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup
        
    aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]


def load_entity_names(file_path: str) -> List[Tuple[int, List[str]]]:
    """
    Load entity names from JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)