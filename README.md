# SEU - Entity Alignment Package

**From Alignment to Assignment: Frustratingly Simple but Effective Unsupervised Entity Alignment without Neural Networks**

A Python package implementing unsupervised entity alignment between knowledge graphs using word embeddings and graph-based features.

## Package Structure

This repository has been refactored from the original Jupyter notebook into a professional Python package with the following structure:

```
seu/
├── core/           # Core functionality
│   ├── embeddings.py    # GloVe embedding download and loading
│   ├── features.py      # Word-level and character-level feature generation  
│   ├── graph.py         # Graph operations and sparse matrix construction
│   ├── solvers.py       # Hungarian algorithm and Sinkhorn operation solvers
│   └── evaluation.py    # Testing metrics (hits@1, hits@10, MRR)
├── data/           # Data loading utilities
│   └── loaders.py       # KG data loading utilities
├── utils/          # General utilities
│   └── helpers.py       # Helper functions and logging
├── config/         # Configuration management
│   └── settings.py      # Settings and constants
└── cli/            # Command line interface
    └── main.py          # CLI for running experiments

tests/              # Comprehensive test suite
├── unit/           # Unit tests for individual modules
├── integration/    # Integration tests for component interactions  
└── fixtures/       # Test fixtures and sample data
```

## Installation

Install the package in development mode:

```bash
pip install -e .
```

Install with all development dependencies:

```bash
pip install -e ".[dev]"
```

## Original Data and Requirements

### Datasets

* ent_ids_1: ids for entities in source KG
* ent_ids_2: ids for entities in target KG  
* ref_ent_ids: entity links encoded by ids
* triples_1: relation triples encoded by ids in source KG
* triples_2: relation triples encoded by ids in target KG

### Word Vectors

Download the GloVe vectors:
- URL: http://nlp.stanford.edu/data/glove.6B.zip
- Use: "glove.6B.300d.txt" as the word vectors

### Environment

- Python >= 3.8
- tensorflow >= 2.4.1
- numpy >= 1.19.0
- scipy >= 1.6.0
- numba >= 0.53.0
- tqdm >= 4.60.0
- requests >= 2.25.0

## Usage

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_embeddings.py

# Run with coverage
pytest --cov=seu --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/
```

### Original Notebook
The original implementation is still available in `notebooks/main.ipynb` for reference.
