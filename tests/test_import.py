"""Tests for the import command."""

import json
from pathlib import Path

import h5py
import numpy as np
from typer.testing import CliRunner

from h5ad.cli import app


runner = CliRunner()


class TestImportCsv:
    def test_import_csv_obs_inplace(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into obs with --inplace."""
        csv_file = temp_dir / "new_obs.csv"
        csv_file.write_text(
            "cell_id,score,label\n"
            "cell_1,1.5,A\n"
            "cell_2,2.5,B\n"
            "cell_3,3.5,A\n"
            "cell_4,4.5,C\n"
            "cell_5,5.5,B\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 0
        assert "5 rows" in result.output
        assert "2 columns" in result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "obs" in f
            obs = f["obs"]
            assert "score" in obs
            assert "label" in obs

    def test_import_csv_obs_output(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into obs with output file."""
        csv_file = temp_dir / "new_obs.csv"
        csv_file.write_text(
            "cell_id,score,label\n"
            "cell_1,1.5,A\n"
            "cell_2,2.5,B\n"
            "cell_3,3.5,A\n"
            "cell_4,4.5,C\n"
            "cell_5,5.5,B\n"
        )
        output_file = temp_dir / "output.h5ad"

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "-o",
                str(output_file),
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output file has the new data
        with h5py.File(output_file, "r") as f:
            assert "obs" in f
            obs = f["obs"]
            assert "score" in obs

        # Verify source file is unchanged
        with h5py.File(sample_h5ad_file, "r") as f:
            obs = f["obs"]
            assert "score" not in obs

    def test_import_csv_var(self, sample_h5ad_file, temp_dir):
        """Test importing CSV into var."""
        csv_file = temp_dir / "new_var.csv"
        csv_file.write_text(
            "gene_id,mean,std\n"
            "gene_1,0.1,0.01\n"
            "gene_2,0.2,0.02\n"
            "gene_3,0.3,0.03\n"
            "gene_4,0.4,0.04\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "var",
                str(csv_file),
                "--inplace",
                "-i",
                "gene_id",
            ],
        )
        assert result.exit_code == 0
        assert "4 rows" in result.output

    def test_import_csv_dimension_mismatch(self, sample_h5ad_file, temp_dir):
        """Test that dimension mismatch is rejected."""
        csv_file = temp_dir / "wrong_obs.csv"
        csv_file.write_text("cell_id,score\ncell_1,1.0\ncell_2,2.0\n")

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "cell_id",
            ],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_csv_invalid_index_column(self, sample_h5ad_file, temp_dir):
        """Test that invalid index column is rejected."""
        csv_file = temp_dir / "obs.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "obs",
                str(csv_file),
                "--inplace",
                "-i",
                "nonexistent",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_import_csv_not_obs_var(self, sample_h5ad_file, temp_dir):
        """Test that CSV import is only allowed for obs/var."""
        csv_file = temp_dir / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "uns/data", str(csv_file), "--inplace"],
        )
        assert result.exit_code == 1
        assert "only supported for 'obs' or 'var'" in result.output

    def test_import_requires_output_or_inplace(self, sample_h5ad_file, temp_dir):
        """Test that either --output or --inplace is required."""
        csv_file = temp_dir / "obs.csv"
        csv_file.write_text("a,b\n1,2\n")

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "obs", str(csv_file)],
        )
        assert result.exit_code == 1
        assert "Output file is required" in result.output


class TestImportNpy:
    def test_import_npy_obsm(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into obsm."""
        npy_file = temp_dir / "pca.npy"
        arr = np.random.randn(5, 10).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "obsm/X_pca", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 0
        assert "5×10" in result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "obsm/X_pca" in f
            np.testing.assert_allclose(f["obsm/X_pca"][...], arr)

    def test_import_npy_varm(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into varm."""
        npy_file = temp_dir / "pcs.npy"
        arr = np.random.randn(4, 5).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "varm/PCs", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "varm/PCs" in f

    def test_import_npy_X(self, sample_h5ad_file, temp_dir):
        """Test importing NPY into X."""
        npy_file = temp_dir / "X.npy"
        arr = np.random.randn(5, 4).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "X", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "X" in f
            np.testing.assert_allclose(f["X"][...], arr)

    def test_import_npy_dimension_mismatch_obsm(self, sample_h5ad_file, temp_dir):
        """Test that obsm dimension mismatch is rejected."""
        npy_file = temp_dir / "bad_pca.npy"
        arr = np.random.randn(10, 5).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "obsm/X_pca", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()

    def test_import_npy_dimension_mismatch_X(self, sample_h5ad_file, temp_dir):
        """Test that X dimension mismatch is rejected."""
        npy_file = temp_dir / "bad_X.npy"
        arr = np.random.randn(5, 10).astype(np.float32)
        np.save(npy_file, arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "X", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()


class TestImportMtx:
    def test_import_mtx_X(self, sample_h5ad_file, temp_dir):
        """Test importing MTX into X."""
        mtx_file = temp_dir / "X.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "% test matrix\n"
            "5 4 5\n"
            "1 1 1.0\n"
            "2 2 2.0\n"
            "3 3 3.0\n"
            "4 4 4.0\n"
            "5 1 5.0\n"
        )

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "X", str(mtx_file), "--inplace"],
        )
        assert result.exit_code == 0
        assert "5×4" in result.output
        assert "5 non-zero" in result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "X" in f
            X = f["X"]
            enc = X.attrs.get("encoding-type")
            if isinstance(enc, bytes):
                enc = enc.decode("utf-8")
            assert enc == "csr_matrix"

    def test_import_mtx_layer(self, sample_h5ad_file, temp_dir):
        """Test importing MTX into layers."""
        mtx_file = temp_dir / "layer.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "5 4 3\n"
            "1 1 1.0\n"
            "3 2 2.0\n"
            "5 4 3.0\n"
        )

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "layers/counts",
                str(mtx_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "layers/counts" in f

    def test_import_mtx_dimension_mismatch(self, sample_h5ad_file, temp_dir):
        """Test that MTX dimension mismatch is rejected."""
        mtx_file = temp_dir / "bad.mtx"
        mtx_file.write_text(
            "%%MatrixMarket matrix coordinate real general\n" "10 4 1\n" "1 1 1.0\n"
        )

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "X", str(mtx_file), "--inplace"],
        )
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower()


class TestImportJson:
    def test_import_json_uns(self, sample_h5ad_file, temp_dir):
        """Test importing JSON into uns."""
        json_file = temp_dir / "metadata.json"
        json_file.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "colors": ["red", "green", "blue"],
                    "n_pcs": 50,
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "uns/metadata",
                str(json_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0
        assert "JSON data" in result.output

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "uns/metadata" in f
            assert "colors" in f["uns/metadata"]
            assert "n_pcs" in f["uns/metadata"]

    def test_import_json_nested(self, sample_h5ad_file, temp_dir):
        """Test importing nested JSON."""
        json_file = temp_dir / "config.json"
        json_file.write_text(
            json.dumps(
                {
                    "settings": {
                        "threshold": 0.5,
                        "enabled": True,
                    },
                    "labels": ["A", "B", "C"],
                }
            )
        )

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "uns/config",
                str(json_file),
                "--inplace",
            ],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            assert "uns/config/settings" in f
            assert "uns/config/labels" in f


class TestImportValidation:
    def test_unsupported_extension(self, sample_h5ad_file, temp_dir):
        """Test that unsupported extensions are rejected."""
        bad_file = temp_dir / "data.xlsx"
        bad_file.write_text("dummy")

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "obs", str(bad_file), "--inplace"],
        )
        assert result.exit_code == 1
        assert "Unsupported" in result.output

    def test_index_column_only_for_csv_obs_var(self, sample_h5ad_file, temp_dir):
        """Test that --index-column is only valid for CSV obs/var."""
        npy_file = temp_dir / "data.npy"
        np.save(npy_file, np.array([1, 2, 3]))

        result = runner.invoke(
            app,
            [
                "import",
                str(sample_h5ad_file),
                "uns/data",
                str(npy_file),
                "--inplace",
                "-i",
                "col",
            ],
        )
        assert result.exit_code == 1
        assert "only valid for CSV" in result.output

    def test_replace_existing_object(self, sample_h5ad_file, temp_dir):
        """Test that existing objects can be replaced."""
        with h5py.File(sample_h5ad_file, "r") as f:
            original_X = np.array(f["X"][...])

        npy_file = temp_dir / "new_X.npy"
        new_arr = np.ones((5, 4), dtype=np.float32) * 999
        np.save(npy_file, new_arr)

        result = runner.invoke(
            app,
            ["import", str(sample_h5ad_file), "X", str(npy_file), "--inplace"],
        )
        assert result.exit_code == 0

        with h5py.File(sample_h5ad_file, "r") as f:
            np.testing.assert_allclose(f["X"][...], new_arr)
            assert not np.allclose(f["X"][...], original_X)
