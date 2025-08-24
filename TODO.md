# SEU Package Refactoring Plan

## Project Analysis
This repository contains a Jupyter notebook implementation of "From Alignment to Assignment: Frustratingly Simple but Effective Unsupervised Entity Alignment without Neural Networks". The current codebase needs refactoring into a proper Python package structure with comprehensive testing.

## Current Structure Assessment
- **main.ipynb**: Contains entire workflow (embedding download, feature generation, graph operations, solving methods)
- **utils.py**: Basic utility functions for loading triples, aligned pairs, and evaluation
- **Data**: Multiple KG datasets (dbp_ja_en, dbp_fr_en, dbp_zh_en, srprs_de_en, srprs_fr_en)
- **Dependencies**: tensorflow, numpy, scipy, tqdm, numba, json, os, tempfile, zipfile, requests

## Proposed Package Structure

```
seu/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── embeddings.py      # GloVe embedding download and loading
│   ├── features.py        # Word-level and character-level feature generation
│   ├── graph.py          # Graph operations and sparse matrix construction
│   ├── solvers.py        # Hungarian algorithm and Sinkhorn operation solvers
│   └── evaluation.py     # Testing metrics (hits@1, hits@10, MRR)
├── data/
│   ├── __init__.py
│   └── loaders.py        # KG data loading utilities (from utils.py)
├── utils/
│   ├── __init__.py
│   └── helpers.py        # General utility functions
├── config/
│   ├── __init__.py
│   └── settings.py       # Configuration management and constants
└── cli/
    ├── __init__.py
    └── main.py           # Command line interface for running experiments
```

## Test Structure

```
tests/
├── __init__.py
├── unit/
│   ├── test_embeddings.py
│   ├── test_features.py
│   ├── test_graph.py
│   ├── test_solvers.py
│   ├── test_evaluation.py
│   ├── test_data_loaders.py
│   └── test_helpers.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_pipeline.py
├── fixtures/
│   ├── __init__.py
│   ├── sample_data.py    # Small test datasets
│   ├── mock_embeddings.py # Mock embedding data
│   └── expected_results.py # Expected test outcomes
└── conftest.py           # pytest configuration and shared fixtures
```

## Detailed Refactoring Steps

### Phase 1: Package Setup and Structure
1. Create package directory structure
2. Create pyproject.toml for package configuration, complete with required dependencies
4. Set up __init__.py files with proper imports
5. Create .gitignore for Python packages
6. Set up pre-commit hooks for code quality

### Phase 2: Core Module Extraction

#### 2.1 Embeddings Module (seu/core/embeddings.py)
- Extract `download_and_extract_file` function from notebook cell 2
- Create `EmbeddingLoader` class with methods:
  - `download_glove_embeddings(url, dimension, dest_path)`
  - `load_word_vectors(file_path)` 
  - `get_embedding_dimension()`
- Add proper error handling and logging
- Support multiple embedding formats (GloVe, Word2Vec, FastText)

#### 2.2 Features Module (seu/core/features.py)
- Extract feature generation logic from notebook cells 8-11
- Create `FeatureGenerator` class with methods:
  - `generate_bigram_dictionary(entity_names)`
  - `generate_word_features(entity_names, word_vectors)`
  - `generate_char_features(entity_names, bigram_dict)`
  - `generate_hybrid_features(entity_names, word_vectors, mode='hybrid-level')`
- Add normalization and validation functions

#### 2.3 Graph Module (seu/core/graph.py)  
- Extract sparse matrix construction from notebook cell 10
- Create `GraphBuilder` class with methods:
  - `build_sparse_adjacency_matrix(triples, node_size)`
  - `calculate_relation_weights(triples)`
  - `propagate_features(feature_matrix, adj_matrix, depth)`
- Optimize memory usage for large graphs

#### 2.4 Solvers Module (seu/core/solvers.py)
- Extract Hungarian and Sinkhorn solvers from cells 13-14
- Create `EntityAlignmentSolver` base class
- Implement `HungarianSolver` and `SinkhornSolver` subclasses
- Add configurable parameters (temperature, iterations)
- Support batch processing for large datasets

#### 2.5 Evaluation Module (seu/core/evaluation.py)
- Extract and enhance test function from utils.py
- Create `Evaluator` class with methods:
  - `calculate_hits_at_k(similarities, k=[1, 5, 10])`
  - `calculate_mrr(similarities)`
  - `calculate_precision_recall(similarities)`
  - `generate_evaluation_report(results)`

### Phase 3: Data and Utilities

#### 3.1 Data Loaders (seu/data/loaders.py)
- Refactor functions from utils.py
- Create `KGDataLoader` class with methods:
  - `load_triples(file_path, reverse=True)`
  - `load_aligned_pairs(file_path, ratio=0.3)`
  - `load_entity_names(file_path)`
  - `validate_data_integrity()`

#### 3.2 Configuration (seu/config/settings.py)
- Define constants (default URLs, file paths, hyperparameters)
- Create configuration dataclasses
- Support config file loading (YAML/JSON)

#### 3.3 Utilities (seu/utils/helpers.py)
- Common helper functions
- Logging configuration
- File I/O utilities

### Phase 4: CLI Interface (seu/cli/main.py)
- Create command-line interface using argparse or click
- Support for different datasets and configurations
- Progress bars and logging output
- Results export functionality

### Phase 5: Comprehensive Testing

#### 5.1 Unit Tests
**tests/test_embeddings.py**:
- Test embedding download with mocked HTTP requests
- Test word vector loading with sample files
- Test error handling for corrupted/missing files
- Test different embedding formats

**tests/test_features.py**:
- Test bigram dictionary generation
- Test word-level feature extraction with known embeddings
- Test character-level feature extraction
- Test hybrid feature combination
- Test normalization correctness

**tests/test_graph.py**:
- Test sparse matrix construction with small graphs
- Test feature propagation with known inputs/outputs
- Test relation weight calculations
- Test memory efficiency with large synthetic graphs

**tests/test_solvers.py**:
- Test Hungarian algorithm with small similarity matrices
- Test Sinkhorn operations with known convergence
- Test batch processing functionality
- Compare solver outputs for consistency

**tests/test_evaluation.py**:
- Test hits@k calculations with known rankings
- Test MRR calculations
- Test edge cases (perfect/worst alignments)
- Validate against original notebook results

**tests/test_data_loaders.py**:
- Test triple loading with sample KG data
- Test entity pair loading
- Test data validation functions
- Test error handling for malformed data

#### 5.2 Integration Tests
**tests/test_end_to_end.py**:
- Full pipeline test with small dataset
- Compare results with original notebook
- Test different configuration combinations
- Performance regression tests

**tests/test_pipeline.py**:
- Test component interactions
- Test data flow between modules
- Test error propagation

#### 5.3 Test Fixtures and Mocking
**tests/fixtures/sample_data.py**:
- Small synthetic KG datasets
- Known entity alignments for validation
- Various edge cases

**tests/fixtures/mock_embeddings.py**:
- Small word embedding matrices
- Controlled vocabulary for testing

**tests/conftest.py**:
- Shared pytest fixtures
- Test configuration
- Temporary file management

### Phase 6: Documentation and Quality

#### 6.1 Documentation
- API documentation with sphinx
- Usage examples and tutorials
- Performance benchmarks
- Migration guide from notebook

#### 6.2 Code Quality
- Type hints throughout codebase
- Docstrings for all public methods
- Code formatting with black
- Linting with flake8/pylint
- Static type checking with mypy

#### 6.3 CI/CD Setup
- GitHub Actions for testing
- Test coverage reporting
- Automated releases
- Multi-python version testing

### Phase 7: Performance and Optimization

#### 7.1 Memory Optimization
- Lazy loading of embeddings
- Chunked processing for large datasets
- Memory profiling and optimization

#### 7.2 Computational Optimization
- Leverage numba JIT compilation
- GPU support with TensorFlow/PyTorch
- Parallel processing where applicable

### Phase 8: Compatibility and Testing Framework

#### 8.1 Test Framework Compatibility
**Both unittest and pytest support**:
- Use pytest as primary framework
- Ensure unittest.TestCase compatibility where needed
- Create custom test base classes if required
- Support for both discovery methods

**Test execution commands**:
```bash
# pytest execution
pytest tests/
pytest tests/unit/test_embeddings.py -v
pytest tests/ --cov=seu --cov-report=html

# unittest execution  
python -m unittest discover tests/
python -m unittest tests.unit.test_embeddings -v
```

#### 8.2 Test Categories
- **Unit tests**: Individual function/method testing
- **Integration tests**: Component interaction testing
- **Regression tests**: Ensure results match original implementation
- **Performance tests**: Speed and memory usage benchmarks
- **Property-based tests**: Using hypothesis for edge case discovery

### Phase 9: Validation and Migration

#### 9.1 Results Validation
- Compare package results with original notebook
- Validate on all provided datasets
- Statistical significance testing
- Performance comparison

#### 9.2 Migration Support
- Provide notebook-to-package migration script
- Backward compatibility layer if needed
- Clear migration documentation

## Success Criteria
1. All functionality from main.ipynb successfully extracted and modularized
2. 100% test coverage for core functionality
3. Results identical to original implementation
4. Performance within 5% of original (preferably improved)
5. Support for both unittest and pytest frameworks
6. Clean, documented, and maintainable codebase
7. Successful CI/CD pipeline
8. Comprehensive documentation

## Estimated Timeline
- **Phase 1-2**: 2-3 days (Structure and core modules)
- **Phase 3-4**: 1-2 days (Data utilities and CLI)
- **Phase 5**: 3-4 days (Comprehensive testing)
- **Phase 6-7**: 2-3 days (Documentation and optimization)
- **Phase 8-9**: 1-2 days (Compatibility and validation)

**Total Estimated Time**: 9-14 days

## Dependencies to Add
```
# Core dependencies
numpy>=1.19.0
scipy>=1.6.0
tensorflow>=2.4.1
numba>=0.53.0
tqdm>=4.60.0
requests>=2.25.0

# Development dependencies  
pytest>=6.2.0
pytest-cov>=2.11.0
pytest-mock>=3.6.0
hypothesis>=6.10.0
black>=21.0.0
flake8>=3.9.0
mypy>=0.812
sphinx>=4.0.0

# CLI dependencies
click>=8.0.0
pyyaml>=5.4.0
```

This plan provides a comprehensive roadmap for converting the Jupyter notebook into a professional-grade Python package while maintaining full compatibility with both unittest and pytest frameworks.