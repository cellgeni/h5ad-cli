"""Tests for subset.py module functions."""

import pytest
import h5py
import numpy as np
from pathlib import Path
from h5ad.commands.subset import (
    _read_name_file,
    indices_from_name_set,
    subset_axis_group,
    subset_dense_matrix,
    subset_sparse_matrix_group,
    subset_h5ad,
)
from rich.console import Console


class TestReadNameFile:
    """Tests for _read_name_file function."""

    def test_read_name_file(self, temp_dir):
        """Test reading names from file."""
        file_path = temp_dir / "names.txt"
        file_path.write_text("name1\nname2\nname3\n")

        names = _read_name_file(file_path)
        assert names == {"name1", "name2", "name3"}

    def test_read_name_file_with_blanks(self, temp_dir):
        """Test reading names with blank lines."""
        file_path = temp_dir / "names.txt"
        file_path.write_text("name1\n\nname2\n  \nname3\n")

        names = _read_name_file(file_path)
        assert names == {"name1", "name2", "name3"}

    def test_read_name_file_empty(self, temp_dir):
        """Test reading empty file."""
        file_path = temp_dir / "names.txt"
        file_path.write_text("")

        names = _read_name_file(file_path)
        assert names == set()


class TestIndicesFromNameSet:
    """Tests for indices_from_name_set function."""

    def test_indices_from_name_set_all_found(self, sample_h5ad_file):
        """Test finding all names."""
        with h5py.File(sample_h5ad_file, "r") as f:
            names_ds = f["obs"]["obs_names"]
            keep = {"cell_1", "cell_3", "cell_5"}

            indices, missing = indices_from_name_set(names_ds, keep)
            assert len(indices) == 3
            assert set(indices) == {0, 2, 4}
            assert len(missing) == 0

    def test_indices_from_name_set_some_missing(self, sample_h5ad_file):
        """Test with some missing names."""
        with h5py.File(sample_h5ad_file, "r") as f:
            names_ds = f["obs"]["obs_names"]
            keep = {"cell_1", "cell_99", "cell_3"}

            indices, missing = indices_from_name_set(names_ds, keep)
            assert len(indices) == 2
            assert set(indices) == {0, 2}
            assert missing == {"cell_99"}

    def test_indices_from_name_set_none_found(self, sample_h5ad_file):
        """Test with no names found."""
        with h5py.File(sample_h5ad_file, "r") as f:
            names_ds = f["obs"]["obs_names"]
            keep = {"nonexistent_1", "nonexistent_2"}

            indices, missing = indices_from_name_set(names_ds, keep)
            assert len(indices) == 0
            assert missing == {"nonexistent_1", "nonexistent_2"}

    def test_indices_from_name_set_chunked(self, temp_dir):
        """Test chunked processing with many names."""
        file_path = temp_dir / "large.h5ad"

        # Create file with many names
        with h5py.File(file_path, "w") as f:
            obs = f.create_group("obs")
            obs.attrs["_index"] = "obs_names"
            names = [f"cell_{i}" for i in range(1000)]
            obs.create_dataset("obs_names", data=np.array(names, dtype="S"))

        with h5py.File(file_path, "r") as f:
            names_ds = f["obs"]["obs_names"]
            keep = {f"cell_{i}" for i in [0, 100, 500, 999]}

            indices, missing = indices_from_name_set(names_ds, keep, chunk_size=100)
            assert len(indices) == 4
            assert set(indices) == {0, 100, 500, 999}
            assert len(missing) == 0


class TestSubsetAxisGroup:
    """Tests for subset_axis_group function."""

    def test_subset_axis_group_with_indices(self, sample_h5ad_file, temp_dir):
        """Test subsetting axis group with indices."""
        output = temp_dir / "subset.h5ad"

        with h5py.File(sample_h5ad_file, "r") as src, h5py.File(output, "w") as dst:
            indices = np.array([0, 2, 4], dtype=np.int64)
            obs_dst = dst.create_group("obs")

            subset_axis_group(src["obs"], obs_dst, indices)

            # Check that obs_names were subsetted
            assert obs_dst["obs_names"].shape[0] == 3
            # Check that other datasets were subsetted
            assert obs_dst["cell_type"].shape[0] == 3
            assert obs_dst["n_counts"].shape[0] == 3

    def test_subset_axis_group_no_indices(self, sample_h5ad_file, temp_dir):
        """Test copying axis group without subsetting."""
        output = temp_dir / "subset.h5ad"

        with h5py.File(sample_h5ad_file, "r") as src, h5py.File(output, "w") as dst:
            obs_dst = dst.create_group("obs")

            subset_axis_group(src["obs"], obs_dst, None)

            # Check that all data was copied
            assert obs_dst["obs_names"].shape[0] == 5
            assert obs_dst["cell_type"].shape[0] == 5

    def test_subset_axis_group_categorical(self, sample_categorical_h5ad, temp_dir):
        """Test subsetting axis group with categorical column."""
        output = temp_dir / "subset.h5ad"

        with (
            h5py.File(sample_categorical_h5ad, "r") as src,
            h5py.File(output, "w") as dst,
        ):
            indices = np.array([0, 2], dtype=np.int64)
            obs_dst = dst.create_group("obs")

            subset_axis_group(src["obs"], obs_dst, indices)

            # Check categorical structure is preserved
            assert "cell_type" in obs_dst
            assert "categories" in obs_dst["cell_type"]
            assert "codes" in obs_dst["cell_type"]
            # Categories should be copied as-is
            assert obs_dst["cell_type"]["categories"].shape[0] == 3
            # Codes should be subsetted
            assert obs_dst["cell_type"]["codes"].shape[0] == 2


class TestSubsetDenseMatrix:
    """Tests for subset_dense_matrix function."""

    def test_subset_dense_matrix_both_axes(self, sample_h5ad_file, temp_dir):
        """Test subsetting dense matrix on both axes."""
        output = temp_dir / "subset.h5ad"

        with h5py.File(sample_h5ad_file, "r") as src, h5py.File(output, "w") as dst:
            obs_idx = np.array([0, 2, 4], dtype=np.int64)  # 3 cells
            var_idx = np.array([0, 2], dtype=np.int64)  # 2 genes

            subset_dense_matrix(src["X"], dst, "X", obs_idx, var_idx, chunk_rows=2)

            assert dst["X"].shape == (3, 2)
            # Check some values
            expected = src["X"][obs_idx, :][:, var_idx]
            np.testing.assert_array_equal(dst["X"][...], expected)

    def test_subset_dense_matrix_obs_only(self, sample_h5ad_file, temp_dir):
        """Test subsetting dense matrix on obs axis only."""
        output = temp_dir / "subset.h5ad"

        with h5py.File(sample_h5ad_file, "r") as src, h5py.File(output, "w") as dst:
            obs_idx = np.array([1, 3], dtype=np.int64)

            subset_dense_matrix(src["X"], dst, "X", obs_idx, None, chunk_rows=1)

            assert dst["X"].shape == (2, 4)

    def test_subset_dense_matrix_var_only(self, sample_h5ad_file, temp_dir):
        """Test subsetting dense matrix on var axis only."""
        output = temp_dir / "subset.h5ad"

        with h5py.File(sample_h5ad_file, "r") as src, h5py.File(output, "w") as dst:
            var_idx = np.array([1, 2], dtype=np.int64)

            subset_dense_matrix(src["X"], dst, "X", None, var_idx, chunk_rows=2)

            assert dst["X"].shape == (5, 2)


class TestSubsetSparseMatrixGroup:
    """Tests for subset_sparse_matrix_group function."""

    def test_subset_sparse_csr_both_axes(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subsetting CSR sparse matrix on both axes."""
        output = temp_dir / "subset.h5ad"

        with (
            h5py.File(sample_sparse_csr_h5ad, "r") as src,
            h5py.File(output, "w") as dst,
        ):
            obs_idx = np.array([0, 2, 3], dtype=np.int64)  # rows 0, 2, 3
            var_idx = np.array([0, 2], dtype=np.int64)  # cols 0, 2

            subset_sparse_matrix_group(src["X"], dst, "X", obs_idx, var_idx)

            assert dst["X"].attrs["shape"][0] == 3
            assert dst["X"].attrs["shape"][1] == 2
            # h5py stores encoding-type as string
            encoding = dst["X"].attrs["encoding-type"]
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            assert encoding == "csr_matrix"

            # Check data structure
            assert "data" in dst["X"]
            assert "indices" in dst["X"]
            assert "indptr" in dst["X"]

            # indptr should have length = n_rows + 1
            assert dst["X"]["indptr"].shape[0] == 4

    def test_subset_sparse_csr_obs_only(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subsetting CSR sparse matrix on obs axis only."""
        output = temp_dir / "subset.h5ad"

        with (
            h5py.File(sample_sparse_csr_h5ad, "r") as src,
            h5py.File(output, "w") as dst,
        ):
            obs_idx = np.array([0, 3], dtype=np.int64)

            subset_sparse_matrix_group(src["X"], dst, "X", obs_idx, None)

            assert dst["X"].attrs["shape"][0] == 2
            assert dst["X"].attrs["shape"][1] == 3

    def test_subset_sparse_csc_both_axes(self, sample_sparse_csc_h5ad, temp_dir):
        """Test subsetting CSC sparse matrix on both axes."""
        output = temp_dir / "subset.h5ad"

        with (
            h5py.File(sample_sparse_csc_h5ad, "r") as src,
            h5py.File(output, "w") as dst,
        ):
            obs_idx = np.array([0, 2], dtype=np.int64)
            var_idx = np.array([0, 2, 3], dtype=np.int64)

            subset_sparse_matrix_group(src["X"], dst, "X", obs_idx, var_idx)

            assert dst["X"].attrs["shape"][0] == 2
            assert dst["X"].attrs["shape"][1] == 3
            # h5py stores encoding-type as string
            encoding = dst["X"].attrs["encoding-type"]
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            assert encoding == "csc_matrix"

    def test_subset_sparse_empty_result(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subsetting sparse matrix resulting in empty matrix."""
        output = temp_dir / "subset.h5ad"

        with (
            h5py.File(sample_sparse_csr_h5ad, "r") as src,
            h5py.File(output, "w") as dst,
        ):
            # Row 1 has no non-zero elements
            obs_idx = np.array([1], dtype=np.int64)
            var_idx = np.array([0, 1, 2], dtype=np.int64)

            subset_sparse_matrix_group(src["X"], dst, "X", obs_idx, var_idx)

            assert dst["X"]["data"].shape[0] == 0
            assert dst["X"]["indices"].shape[0] == 0


class TestSubsetH5ad:
    """Integration tests for subset_h5ad function."""

    def test_subset_h5ad_obs_only(self, sample_h5ad_file, temp_dir):
        """Test subsetting h5ad file by obs only."""
        obs_file = temp_dir / "obs_names.txt"
        obs_file.write_text("cell_1\ncell_3\ncell_5\n")

        output = temp_dir / "subset.h5ad"
        console = Console(stderr=True)

        subset_h5ad(
            file=sample_h5ad_file,
            output=output,
            obs_file=obs_file,
            var_file=None,
            chunk_rows=1024,
            console=console,
        )

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["obs"]["obs_names"].shape[0] == 3
            assert f["var"]["var_names"].shape[0] == 4  # All vars kept
            assert f["X"].shape == (3, 4)

    def test_subset_h5ad_var_only(self, sample_h5ad_file, temp_dir):
        """Test subsetting h5ad file by var only."""
        var_file = temp_dir / "var_names.txt"
        var_file.write_text("gene_1\ngene_3\n")

        output = temp_dir / "subset.h5ad"
        console = Console(stderr=True)

        subset_h5ad(
            file=sample_h5ad_file,
            output=output,
            obs_file=None,
            var_file=var_file,
            chunk_rows=1024,
            console=console,
        )

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["obs"]["obs_names"].shape[0] == 5  # All obs kept
            assert f["var"]["var_names"].shape[0] == 2
            assert f["X"].shape == (5, 2)

    def test_subset_h5ad_both(self, sample_h5ad_file, temp_dir):
        """Test subsetting h5ad file by both obs and var."""
        obs_file = temp_dir / "obs_names.txt"
        obs_file.write_text("cell_2\ncell_4\n")

        var_file = temp_dir / "var_names.txt"
        var_file.write_text("gene_2\ngene_4\n")

        output = temp_dir / "subset.h5ad"
        console = Console(stderr=True)

        subset_h5ad(
            file=sample_h5ad_file,
            output=output,
            obs_file=obs_file,
            var_file=var_file,
            chunk_rows=1024,
            console=console,
        )

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["obs"]["obs_names"].shape[0] == 2
            assert f["var"]["var_names"].shape[0] == 2
            assert f["X"].shape == (2, 2)
            # Check uns was copied
            assert "uns" in f

    def test_subset_h5ad_sparse_csr(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subsetting h5ad file with CSR sparse matrix."""
        obs_file = temp_dir / "obs_names.txt"
        obs_file.write_text("cell_1\ncell_3\n")

        output = temp_dir / "subset.h5ad"
        console = Console(stderr=True)

        subset_h5ad(
            file=sample_sparse_csr_h5ad,
            output=output,
            obs_file=obs_file,
            var_file=None,
            chunk_rows=1024,
            console=console,
        )

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["obs"]["obs_names"].shape[0] == 2
            # h5py stores encoding-type as string
            encoding = f["X"].attrs["encoding-type"]
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            assert encoding == "csr_matrix"

    def test_subset_h5ad_sparse_csc(self, sample_sparse_csc_h5ad, temp_dir):
        """Test subsetting h5ad file with CSC sparse matrix."""
        var_file = temp_dir / "var_names.txt"
        var_file.write_text("gene_1\ngene_3\n")

        output = temp_dir / "subset.h5ad"
        console = Console(stderr=True)

        subset_h5ad(
            file=sample_sparse_csc_h5ad,
            output=output,
            obs_file=None,
            var_file=var_file,
            chunk_rows=1024,
            console=console,
        )

        assert output.exists()

        with h5py.File(output, "r") as f:
            assert f["var"]["var_names"].shape[0] == 2
            # h5py stores encoding-type as string
            encoding = f["X"].attrs["encoding-type"]
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            assert encoding == "csc_matrix"
