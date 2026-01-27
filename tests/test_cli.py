"""Tests for CLI commands."""

import pytest
import csv
from pathlib import Path
from typer.testing import CliRunner
from h5ad.cli import app
from h5ad.commands.info import show_info
from h5ad.commands.export import export_table
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

    def test_info_tree_flag(self, sample_h5ad_file):
        """Test info command with --tree flag."""
        result = runner.invoke(app, ["info", "--tree", str(sample_h5ad_file)])
        assert result.exit_code == 0
        # Should show type annotations in angle brackets
        # Output may go to stdout or stderr depending on console config
        output = result.stdout + (result.stderr or "")
        assert "<" in output
        assert ">" in output

    def test_info_tree_short_flag(self, sample_h5ad_file):
        """Test info command with -t short flag."""
        result = runner.invoke(app, ["info", "-t", str(sample_h5ad_file)])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_info_depth_flag(self, sample_h5ad_file):
        """Test info command with --depth flag."""
        result = runner.invoke(
            app, ["info", "--tree", "--depth", "1", str(sample_h5ad_file)]
        )
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_info_depth_short_flag(self, sample_h5ad_file):
        """Test info command with -d short flag."""
        result = runner.invoke(app, ["info", "-t", "-d", "2", str(sample_h5ad_file)])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_info_entry_positional(self, sample_h5ad_file):
        """Test info command with entry as positional argument."""
        result = runner.invoke(app, ["info", str(sample_h5ad_file), "X"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output
        assert "Type:" in output

    def test_info_entry_obs(self, sample_h5ad_file):
        """Test info command with obs entry."""
        result = runner.invoke(app, ["info", str(sample_h5ad_file), "obs"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output
        assert "dataframe" in output

    def test_info_entry_nested_path(self, sample_h5ad_file):
        """Test info command with nested object path."""
        result = runner.invoke(app, ["info", str(sample_h5ad_file), "uns/description"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output

    def test_info_entry_not_found(self, sample_h5ad_file):
        """Test info command with non-existent object path."""
        result = runner.invoke(app, ["info", str(sample_h5ad_file), "nonexistent"])
        assert result.exit_code == 0  # Doesn't exit with error, just shows message
        output = result.stdout + (result.stderr or "")
        assert "not found" in output


class TestExportDataframeCommand:
    """Tests for export dataframe command (replaces table command)."""

    def test_export_dataframe_obs(self, sample_h5ad_file, temp_dir):
        """Test export dataframe for obs axis."""
        output = temp_dir / "obs_table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        # Check CSV content
        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6  # header + 5 rows
            assert "obs_names" in rows[0]

    def test_export_dataframe_var(self, sample_h5ad_file, temp_dir):
        """Test export dataframe for var axis."""
        output = temp_dir / "var_table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "var",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 5  # header + 4 rows

    def test_export_dataframe_columns_filter(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with column filter."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--columns",
                "obs_names,cell_type",
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

    def test_export_dataframe_head(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with head limit."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--head",
                "2",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 rows

    def test_export_dataframe_head_short_flag(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with -n short flag."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "-n",
                "3",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # header + 3 rows

    def test_export_dataframe_stdout(self, sample_h5ad_file):
        """Test export dataframe to stdout (no --output)."""
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--head",
                "2",
            ],
        )
        assert result.exit_code == 0
        # Output should go to stdout
        assert "obs_names" in result.stdout
        assert "cell_" in result.stdout

    def test_export_dataframe_columns_short_flag(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with -c short flag for columns."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "-o",
                str(output),
                "-c",
                "obs_names",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            assert len(header) == 1
            assert "obs_names" in header

    def test_export_dataframe_chunk_rows(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with custom chunk size."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--chunk-rows",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6  # header + 5 rows

    def test_export_dataframe_combined_options(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with multiple options combined."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "-o",
                str(output),
                "-c",
                "obs_names,cell_type",
                "-n",
                "3",
                "-r",
                "1",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # header + 3 rows
            header = rows[0]
            assert "obs_names" in header
            assert "cell_type" in header
            assert "n_counts" not in header

    def test_export_dataframe_invalid_axis(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with invalid axis."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "invalid",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 1
        # Check both stdout and stderr since Console uses stderr=True
        output_text = result.stdout + result.stderr
        assert "obs" in output_text or "var" in output_text

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
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--obs",
                str(obs_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_var(self, sample_h5ad_file, temp_dir):
        """Test subset command with var file."""
        var_file = temp_dir / "var.txt"
        var_file.write_text("gene_1\ngene_2\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--var",
                str(var_file),
            ],
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
                "--output",
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
        result = runner.invoke(
            app, ["subset", str(sample_h5ad_file), "--output", str(output)]
        )
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
                "--output",
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
                "--output",
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

    def test_export_help(self):
        """Test export command help."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "dataframe" in result.stdout
        assert "array" in result.stdout

    def test_export_dataframe_help(self):
        """Test export dataframe command help."""
        result = runner.invoke(app, ["export", "dataframe", "--help"])
        assert result.exit_code == 0
        assert "Export a dataframe" in result.stdout

    def test_import_help(self):
        """Test import command help."""
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "dataframe" in result.stdout
        assert "array" in result.stdout

    def test_subset_help(self):
        """Test subset command help."""
        result = runner.invoke(app, ["subset", "--help"])
        assert result.exit_code == 0
        assert "Subset an h5ad" in result.stdout
