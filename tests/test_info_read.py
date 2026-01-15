"""Tests for info.py and read.py modules."""

import pytest
import h5py
import numpy as np
from h5ad.info import axis_len, get_axis_group
from h5ad.read import decode_str_array, read_categorical_column, col_chunk_as_strings


class TestAxisLen:
    """Tests for axis_len function."""

    def test_axis_len_obs(self, sample_h5ad_file):
        """Test getting length of obs axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            length = axis_len(f, "obs")
            assert length == 5

    def test_axis_len_var(self, sample_h5ad_file):
        """Test getting length of var axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            length = axis_len(f, "var")
            assert length == 4

    def test_axis_len_nonexistent(self, sample_h5ad_file):
        """Test getting length of non-existent axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            length = axis_len(f, "nonexistent")
            assert length is None


class TestGetAxisGroup:
    """Tests for get_axis_group function."""

    def test_get_axis_group_obs(self, sample_h5ad_file):
        """Test getting obs axis group."""
        with h5py.File(sample_h5ad_file, "r") as f:
            group, length, index_name = get_axis_group(f, "obs")
            assert isinstance(group, h5py.Group)
            assert length == 5
            assert index_name == "obs_names"

    def test_get_axis_group_var(self, sample_h5ad_file):
        """Test getting var axis group."""
        with h5py.File(sample_h5ad_file, "r") as f:
            group, length, index_name = get_axis_group(f, "var")
            assert isinstance(group, h5py.Group)
            assert length == 4
            assert index_name == "var_names"

    def test_get_axis_group_invalid(self, sample_h5ad_file):
        """Test getting invalid axis."""
        with h5py.File(sample_h5ad_file, "r") as f:
            with pytest.raises(ValueError, match="axis must be 'obs' or 'var'"):
                get_axis_group(f, "invalid")

    def test_get_axis_group_missing(self, temp_dir):
        """Test getting missing axis."""
        file_path = temp_dir / "empty.h5ad"
        with h5py.File(file_path, "w") as f:
            f.create_group("obs")

        with h5py.File(file_path, "r") as f:
            with pytest.raises(KeyError, match="'var' not found"):
                get_axis_group(f, "var")


class TestDecodeStrArray:
    """Tests for decode_str_array function."""

    def test_decode_bytes_array(self):
        """Test decoding bytes array."""
        arr = np.array([b"hello", b"world"], dtype="S")
        result = decode_str_array(arr)
        assert result.dtype.kind == "U"
        assert list(result) == ["hello", "world"]

    def test_decode_object_array(self):
        """Test decoding object array."""
        arr = np.array(["hello", "world"], dtype=object)
        result = decode_str_array(arr)
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_decode_unicode_array(self):
        """Test decoding unicode array."""
        arr = np.array(["hello", "world"], dtype="U10")
        result = decode_str_array(arr)
        assert result[0] == "hello"
        assert result[1] == "world"


class TestReadCategoricalColumn:
    """Tests for read_categorical_column function."""

    def test_read_categorical_column(self, sample_categorical_h5ad):
        """Test reading categorical column."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}
            result = read_categorical_column(col_group, 0, 4, cache)
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_read_categorical_column_slice(self, sample_categorical_h5ad):
        """Test reading categorical column slice."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}
            result = read_categorical_column(col_group, 1, 3, cache)
            assert result == ["TypeB", "TypeA"]

    def test_read_categorical_column_caching(self, sample_categorical_h5ad):
        """Test that categorical column uses cache."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            col_group = f["obs"]["cell_type"]
            cache = {}

            # First call should populate cache
            read_categorical_column(col_group, 0, 2, cache)
            assert len(cache) == 1

            # Second call should reuse cache
            read_categorical_column(col_group, 2, 4, cache)
            assert len(cache) == 1


class TestColChunkAsStrings:
    """Tests for col_chunk_as_strings function."""

    def test_col_chunk_dataset(self, sample_h5ad_file):
        """Test reading dataset column as strings."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 0, 3, cache)
            assert result == ["TypeA", "TypeB", "TypeA"]

    def test_col_chunk_numeric(self, sample_h5ad_file):
        """Test reading numeric column as strings."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "n_counts", 0, 3, cache)
            assert result == ["100", "200", "150"]

    def test_col_chunk_categorical(self, sample_categorical_h5ad):
        """Test reading categorical column as strings."""
        with h5py.File(sample_categorical_h5ad, "r") as f:
            cache = {}
            result = col_chunk_as_strings(f["obs"], "cell_type", 0, 4, cache)
            assert result == ["TypeA", "TypeB", "TypeA", "TypeC"]

    def test_col_chunk_unsupported(self, sample_h5ad_file):
        """Test reading unsupported column."""
        with h5py.File(sample_h5ad_file, "r") as f:
            cache = {}
            with pytest.raises(RuntimeError, match="Unsupported column"):
                col_chunk_as_strings(f["obs"], "nonexistent", 0, 5, cache)
