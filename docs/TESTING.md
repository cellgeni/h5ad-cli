# H5AD Testing Documentation

## Quick Start

Install development dependencies:
```bash
uv sync --extra dev
```

Run all tests:
```bash
uv run pytest
```

Run specific test file:
```bash
uv run pytest tests/test_subset.py
```

Run specific test class:
```bash
uv run pytest tests/test_subset.py::TestSubsetH5ad
```

Run specific test function:
```bash
uv run pytest tests/test_subset.py::TestSubsetH5ad::test_subset_h5ad_obs_only
```

Run with coverage:
```bash
uv run pytest --cov=h5ad --cov-report=html
```

Run verbose:
```bash
uv run pytest -v
```

Run and show print statements:
```bash
uv run pytest -s
```

Run with short tracebacks:
```bash
uv run pytest --tb=short
```

## Test Structure

- **`conftest.py`** - Pytest fixtures for test h5ad files (dense, CSR sparse, CSC sparse, categorical)
- **`test_info_read.py`** - Tests for info.py and read.py utility functions (58 tests total)
  - Axis length calculation
  - String decoding (bytes/unicode/object arrays)
  - Categorical column reading with caching
  - Column chunk reading
- **`test_subset.py`** - Tests for subsetting operations
  - Name file reading and parsing
  - Index matching with chunked processing
  - Axis group subsetting (including categorical columns)
  - Dense matrix subsetting (chunked)
  - Sparse matrix subsetting (CSR and CSC formats)
  - Integration tests for complete subset workflows
- **`test_cli.py`** - CLI command integration tests
  - `info` command (file structure inspection)
  - `table` command (CSV export with filters)
  - `subset` command (file subsetting)
  - Error handling and validation

## Test Coverage

The test suite provides comprehensive coverage:

- ✅ **CLI Commands**: All commands (info, table, subset) with various options
- ✅ **Dense Matrices**: Chunked reading and subsetting
- ✅ **Sparse Matrices**: CSR and CSC format support
- ✅ **Categorical Data**: Proper handling of categorical columns
- ✅ **Chunked Processing**: Memory-efficient operations on large datasets
- ✅ **Error Handling**: Invalid inputs, missing files, malformed data
- ✅ **Edge Cases**: Empty results, missing names, single row/column

**Current Status**: 58 tests, all passing ✅

## Test Fixtures

### Available Fixtures

- `temp_dir`: Temporary directory for test files
- `sample_h5ad_file`: Basic h5ad file with dense matrix
- `sample_sparse_csr_h5ad`: H5ad file with CSR sparse matrix
- `sample_sparse_csc_h5ad`: H5ad file with CSC sparse matrix
- `sample_categorical_h5ad`: H5ad file with categorical columns

## Coverage Reports

Generate terminal coverage report:
```bash
uv run pytest --cov=h5ad --cov-report=term-missing
```

Generate HTML coverage report:
```bash
uv run pytest --cov=h5ad --cov-report=html
```

View HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

Coverage configuration is in `pyproject.toml` under `[tool.coverage.*]`.
