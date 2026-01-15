"""Pytest configuration and fixtures for h5ad tests."""

from pathlib import Path
import tempfile
import pytest
import h5py
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_h5ad_file(temp_dir):
    """Create a sample h5ad file for testing."""
    file_path = temp_dir / "test.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs (observations/cells)
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Add some metadata
        obs.create_dataset(
            "cell_type",
            data=np.array(["TypeA", "TypeB", "TypeA", "TypeC", "TypeB"], dtype="S"),
        )
        obs.create_dataset(
            "n_counts", data=np.array([100, 200, 150, 300, 250], dtype=np.int32)
        )

        # Create var (variables/genes)
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3", "gene_4"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))
        var.create_dataset(
            "highly_variable", data=np.array([True, False, True, False], dtype=bool)
        )

        # Create dense X matrix (5 cells x 4 genes)
        X = np.array(
            [
                [1.0, 0.0, 2.5, 0.0],
                [0.0, 3.2, 0.0, 1.1],
                [2.1, 0.0, 1.8, 0.0],
                [0.0, 4.5, 0.0, 2.3],
                [1.5, 0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        f.create_dataset("X", data=X)

        # Create uns (unstructured annotations)
        uns = f.create_group("uns")
        uns.create_dataset("description", data=np.array(["Test dataset"], dtype="S"))

    return file_path


@pytest.fixture
def sample_sparse_csr_h5ad(temp_dir):
    """Create a sample h5ad file with CSR sparse matrix."""
    file_path = temp_dir / "test_csr.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create CSR sparse matrix
        # Matrix (4x3):
        # [[1.0, 0.0, 2.0],
        #  [0.0, 0.0, 0.0],
        #  [3.0, 4.0, 0.0],
        #  [0.0, 5.0, 6.0]]
        X = f.create_group("X")
        X.attrs["encoding-type"] = "csr_matrix"
        X.attrs["shape"] = (4, 3)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        indices = np.array([0, 2, 0, 1, 1, 2], dtype=np.int32)
        indptr = np.array([0, 2, 2, 4, 6], dtype=np.int32)

        X.create_dataset("data", data=data)
        X.create_dataset("indices", data=indices)
        X.create_dataset("indptr", data=indptr)

    return file_path


@pytest.fixture
def sample_sparse_csc_h5ad(temp_dir):
    """Create a sample h5ad file with CSC sparse matrix."""
    file_path = temp_dir / "test_csc.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2", "gene_3", "gene_4"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create CSC sparse matrix (same logical matrix as CSR, transposed)
        # Matrix (3x4):
        # [[1.0, 0.0, 3.0, 0.0],
        #  [0.0, 0.0, 4.0, 5.0],
        #  [2.0, 0.0, 0.0, 6.0]]
        X = f.create_group("X")
        X.attrs["encoding-type"] = "csc_matrix"
        X.attrs["shape"] = (3, 4)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        indices = np.array([0, 2, 2, 2, 1, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 1, 3, 5, 7], dtype=np.int32)

        X.create_dataset("data", data=data)
        X.create_dataset("indices", data=indices)
        X.create_dataset("indptr", data=indptr)

    return file_path


@pytest.fixture
def sample_categorical_h5ad(temp_dir):
    """Create a sample h5ad file with categorical columns."""
    file_path = temp_dir / "test_categorical.h5ad"

    with h5py.File(file_path, "w") as f:
        # Create obs with categorical column
        obs = f.create_group("obs")
        obs.attrs["_index"] = "obs_names"
        obs_names = ["cell_1", "cell_2", "cell_3", "cell_4"]
        obs.create_dataset("obs_names", data=np.array(obs_names, dtype="S"))

        # Create categorical column
        cell_type_group = obs.create_group("cell_type")
        cell_type_group.attrs["encoding-type"] = "categorical"
        categories = np.array(["TypeA", "TypeB", "TypeC"], dtype="S")
        codes = np.array([0, 1, 0, 2], dtype=np.int8)
        cell_type_group.create_dataset("categories", data=categories)
        cell_type_group.create_dataset("codes", data=codes)

        # Create var
        var = f.create_group("var")
        var.attrs["_index"] = "var_names"
        var_names = ["gene_1", "gene_2"]
        var.create_dataset("var_names", data=np.array(var_names, dtype="S"))

        # Create X matrix
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        f.create_dataset("X", data=X)

    return file_path
