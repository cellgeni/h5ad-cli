"""Tests for CLI commands."""

import pytest
import csv
from pathlib import Path
from typer.testing import CliRunner
from h5ad.cli import app
from h5ad.commands.info import show_info
from h5ad.commands.table import export_table
from rich.console import Console


runner = CliRunner()


class TestInfoCommand:
    """Tests for info command."""

    def test_info_command_success(self, sample_h5ad_file):
        """Test info command on valid file."""
        result = runner.invoke(app, ["info", str(sample_h5ad_file)])
        assert result.exit_code == 0
        assert "5 × 4" in result.stdout

    def test_info_command_nonexistent_file(self):
        """Test info command on non-existent file."""
        result = runner.invoke(app, ["info", "nonexistent.h5ad"])
        assert result.exit_code != 0

    def test_info_function_direct(self, sample_h5ad_file):
        """Test show_info function directly."""
        console = Console(stderr=True)
        # Should not raise exception
        show_info(sample_h5ad_file, console)


class TestTableCommand:
    """Tests for table command."""

    def test_table_command_obs(self, sample_h5ad_file, temp_dir):
        """Test table command for obs axis."""
        output = temp_dir / "obs_table.csv"
        result = runner.invoke(
            app,
            ["table", str(sample_h5ad_file), "--axis", "obs", "--output", str(output)],
        )
        assert result.exit_code == 0
        assert output.exists()

        # Check CSV content
        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6  # header + 5 rows
            assert "obs_names" in rows[0]

    def test_table_command_var(self, sample_h5ad_file, temp_dir):
        """Test table command for var axis."""
        output = temp_dir / "var_table.csv"
        result = runner.invoke(
            app,
            ["table", str(sample_h5ad_file), "--axis", "var", "--output", str(output)],
        )
        assert result.exit_code == 0
        assert output.exists()

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 5  # header + 4 rows

    def test_table_command_columns_filter(self, sample_h5ad_file, temp_dir):
        """Test table command with column filter."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "table",
                str(sample_h5ad_file),
                "--axis",
                "obs",
                "--columns",
                "obs_names,cell_type",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            assert "obs_names" in header
            assert "cell_type" in header
            assert "n_counts" not in header

    def test_table_command_head(self, sample_h5ad_file, temp_dir):
        """Test table command with head limit."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "table",
                str(sample_h5ad_file),
                "--axis",
                "obs",
                "--head",
                "2",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 rows

    def test_table_command_invalid_axis(self, sample_h5ad_file):
        """Test table command with invalid axis."""
        result = runner.invoke(
            app, ["table", str(sample_h5ad_file), "--axis", "invalid"]
        )
        assert result.exit_code == 1
        # Check both stdout and stderr since Console uses stderr=True
        output = result.stdout + result.stderr
        assert "Invalid axis" in output

    def test_export_table_function(self, sample_h5ad_file, temp_dir):
        """Test export_table function directly."""
        output = temp_dir / "test_table.csv"
        console = Console(stderr=True)

        export_table(
            file=sample_h5ad_file,
            axis="obs",
            columns=None,
            out=output,
            chunk_rows=10,
            head=None,
            console=console,
        )

        assert output.exists()
        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 1


class TestSubsetCommand:
    """Tests for subset command."""

    def test_subset_command_obs(self, sample_h5ad_file, temp_dir):
        """Test subset command with obs file."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_3\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app, ["subset", str(sample_h5ad_file), str(output), "--obs", str(obs_file)]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_var(self, sample_h5ad_file, temp_dir):
        """Test subset command with var file."""
        var_file = temp_dir / "var.txt"
        var_file.write_text("gene_1\ngene_2\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app, ["subset", str(sample_h5ad_file), str(output), "--var", str(var_file)]
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_both(self, sample_h5ad_file, temp_dir):
        """Test subset command with both obs and var files."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_2\n")

        var_file = temp_dir / "var.txt"
        var_file.write_text("gene_1\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                str(output),
                "--obs",
                str(obs_file),
                "--var",
                str(var_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_no_filters(self, sample_h5ad_file, temp_dir):
        """Test subset command without any filters (should fail)."""
        output = temp_dir / "subset.h5ad"
        result = runner.invoke(app, ["subset", str(sample_h5ad_file), str(output)])
        assert result.exit_code == 1
        # Check both stdout and stderr since Console uses stderr=True
        output_text = result.stdout + result.stderr
        assert "At least one of --obs or --var must be provided" in output_text

    def test_subset_command_chunk_rows(self, sample_h5ad_file, temp_dir):
        """Test subset command with custom chunk size."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_2\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                str(output),
                "--obs",
                str(obs_file),
                "--chunk-rows",
                "512",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_sparse(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subset command on sparse matrix file."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_3\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_sparse_csr_h5ad),
                str(output),
                "--obs",
                str(obs_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_help(self):
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Streaming CLI" in result.stdout

    def test_info_help(self):
        """Test info command help."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show high-level information" in result.stdout

    def test_table_help(self):
        """Test table command help."""
        result = runner.invoke(app, ["table", "--help"])
        assert result.exit_code == 0
        assert "Export a table" in result.stdout

    def test_subset_help(self):
        """Test subset command help."""
        result = runner.invoke(app, ["subset", "--help"])
        assert result.exit_code == 0
        assert "Subset an h5ad" in result.stdout
