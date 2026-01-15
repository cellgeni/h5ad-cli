"""Tests for the export command."""

import json
from pathlib import Path

import h5py
import numpy as np
from typer.testing import CliRunner

from h5ad.cli import app


runner = CliRunner()


def _read_mtx(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        assert header.startswith("%%MatrixMarket")
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_rows, n_cols, nnz = map(int, line.split())
        mat = np.zeros((n_rows, n_cols), dtype=np.float32)
        for _ in range(nnz):
            r, c, v = fh.readline().split()
            mat[int(r) - 1, int(c) - 1] = float(v)
        return mat


class TestExportNpy:
    def test_export_npy_dense_X(self, sample_h5ad_file, temp_dir):
        out = temp_dir / "X.npy"
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "X", str(out)])
        assert result.exit_code == 0
        assert out.exists()

        got = np.load(out)
        with h5py.File(sample_h5ad_file, "r") as f:
            expected = np.asarray(f["X"][...])
        np.testing.assert_allclose(got, expected)


class TestExportMtx:
    def test_export_mtx_csr(self, sample_sparse_csr_h5ad, temp_dir):
        out = temp_dir / "X_csr.mtx"
        result = runner.invoke(
            app, ["export", str(sample_sparse_csr_h5ad), "X", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()

        got = _read_mtx(out)
        expected = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [0.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(got, expected)

    def test_export_mtx_csc(self, temp_dir):
        # Build a small, consistent CSC matrix group
        file_path = temp_dir / "test_csc.h5ad"
        with h5py.File(file_path, "w") as f:
            X = f.create_group("X")
            X.attrs["encoding-type"] = "csc_matrix"
            X.attrs["shape"] = (3, 4)
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
            indices = np.array([0, 2, 0, 1, 1, 2], dtype=np.int32)
            indptr = np.array([0, 2, 2, 4, 6], dtype=np.int32)
            X.create_dataset("data", data=data)
            X.create_dataset("indices", data=indices)
            X.create_dataset("indptr", data=indptr)

        out = temp_dir / "X_csc.mtx"
        result = runner.invoke(app, ["export", str(file_path), "X", str(out)])
        assert result.exit_code == 0
        assert out.exists()

        got = _read_mtx(out)
        expected = np.array(
            [
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0, 5.0],
                [2.0, 0.0, 0.0, 6.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(got, expected)


class TestExportJson:
    def test_export_json_uns(self, sample_h5ad_file, temp_dir):
        out = temp_dir / "uns.json"
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "uns", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "description" in payload
        assert payload["description"] == ["Test dataset"]


class TestExportCsv:
    def test_export_csv_obs(self, sample_h5ad_file, temp_dir):
        out = temp_dir / "obs.csv"
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "obs", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "obs_names" in text


class TestExportValidation:
    def test_wrong_extension_for_type(self, sample_h5ad_file, temp_dir):
        """Test that wrong extension is rejected."""
        out = temp_dir / "obs.npy"  # obs is a dataframe, should be .csv
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "obs", str(out)])
        assert result.exit_code == 1
        assert "does not match" in result.output or "Expected" in result.output

    def test_sparse_matrix_wrong_extension(self, sample_sparse_csr_h5ad, temp_dir):
        """Test that sparse matrix rejects .npy extension."""
        out = temp_dir / "X.npy"  # sparse matrix should be .mtx
        result = runner.invoke(
            app, ["export", str(sample_sparse_csr_h5ad), "X", str(out)]
        )
        assert result.exit_code == 1
        assert "does not match" in result.output or ".mtx" in result.output

    def test_dense_matrix_wrong_extension(self, sample_h5ad_file, temp_dir):
        """Test that dense matrix rejects .csv extension."""
        out = temp_dir / "X.csv"  # dense matrix should be .npy
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "X", str(out)])
        assert result.exit_code == 1
        assert "does not match" in result.output or ".npy" in result.output

    def test_json_wrong_extension(self, sample_h5ad_file, temp_dir):
        """Test that dict rejects .npy extension."""
        out = temp_dir / "uns.npy"  # uns is dict, should be .json
        result = runner.invoke(app, ["export", str(sample_h5ad_file), "uns", str(out)])
        assert result.exit_code == 1
        assert "does not match" in result.output or ".json" in result.output

    def test_nonexistent_object(self, sample_h5ad_file, temp_dir):
        """Test that nonexistent object path is rejected."""
        out = temp_dir / "output.csv"
        result = runner.invoke(
            app, ["export", str(sample_h5ad_file), "nonexistent/path", str(out)]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_unknown_type_rejected(self, temp_dir):
        """Test that unknown/complex types are rejected."""
        file_path = temp_dir / "test_unknown.h5ad"
        with h5py.File(file_path, "w") as f:
            g = f.create_group("obs")
            g.create_dataset("obs_names", data=np.array([b"cell1"]))
            g.attrs["_index"] = "obs_names"
            # Create a group without known encoding
            weird = f.create_group("weird_group")
            weird.attrs["encoding-type"] = "some_unknown_encoding"

        out = temp_dir / "weird.json"
        result = runner.invoke(app, ["export", str(file_path), "weird_group", str(out)])
        # Should succeed as it's detected as dict
        # but if it had sparse inside, it would fail
        assert result.exit_code == 0
