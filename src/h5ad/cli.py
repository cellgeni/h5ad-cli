import sys
import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List

import rich
from rich.console import Console
import typer
import h5py
import numpy as np


app = typer.Typer(
    help="Streaming CLI for huge .h5ad files (info, table, subset, export)."
)
console = Console(stderr=True)

export_app = typer.Typer(help="Export objects from an .h5ad file to common formats.")
app.add_typer(export_app, name="export")


@app.command()
def info(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad file",
        exists=True,
        readable=True,
    ),
    obj: Optional[str] = typer.Option(
        None,
        "--object",
        "-o",
        help="Object path to inspect (e.g., 'obsm/X_pca', 'X', 'uns')",
    ),
    types: bool = typer.Option(
        False,
        "--types",
        "-t",
        help="Show detailed type information for all entries",
    ),
) -> None:
    """
    Show high-level information about the .h5ad file.

    Use --types to see type information for each entry.
    Use --object to inspect a specific object in detail.

    Examples:
        h5ad info data.h5ad
        h5ad info --types data.h5ad
        h5ad info --object obsm/X_pca data.h5ad
    """
    show_info(file, console, show_types=types, obj_path=obj)


@app.command()
def table(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad file",
        exists=True,
        readable=True,
    ),
    axis: str = typer.Option("obs", help="Axis to read from ('obs' or 'var')"),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma separated column names to include in the output table",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (defaults to stdout)",
        writable=True,
    ),
    chunk_rows: int = typer.Option(
        10000, "--chunk-rows", "-r", help="Number of rows to read per chunk"
    ),
    head: Optional[int] = typer.Option(
        None, "--head", "-n", help="Output only the first n rows"
    ),
) -> None:
    """
    Export a table of the specified axis ('obs' or 'var') to CSV format.
    Args:
        file (Path): Path to the .h5ad file
        axis (str): Axis to read from ('obs' or 'var')
        columns (Optional[str]): Comma separated column names to include in the output table
        out (Optional[Path]): Output file path (defaults to stdout)
        chunk_rows (int): Number of rows to read per chunk
        head (Optional[int]): Output only the first n rows
    """
    # Validate axis parameter
    if axis not in ("obs", "var"):
        console.print(
            f"[bold red]Error:[/] Invalid axis '{axis}'. Must be either 'obs' or 'var'.",
        )
        raise typer.Exit(code=1)

    col_list: Optional[List[str]] = None
    if columns:
        col_list = [col.strip() for col in columns.split(",") if col.strip()]

    export_table(
        file=file,
        axis=axis,
        columns=col_list,
        out=out,
        chunk_rows=chunk_rows,
        head=head,
        console=console,
    )


@app.command()
def subset(
    file: Path = typer.Argument(..., help="Input .h5ad", exists=True, readable=True),
    output: Path = typer.Argument(..., help="Output .h5ad", writable=True),
    obs: Optional[Path] = typer.Option(
        None,
        "--obs",
        help="File with obs names (one per line)",
        exists=True,
        readable=True,
    ),
    var: Optional[Path] = typer.Option(
        None,
        "--var",
        help="File with var names (one per line)",
        exists=True,
        readable=True,
    ),
    chunk_rows: int = typer.Option(
        1024, "--chunk-rows", "-r", help="Row chunk size for dense matrices"
    ),
) -> None:
    """Subset an h5ad by obs and/or var names."""
    if obs is None and var is None:
        console.print(
            "[bold red]Error:[/] At least one of --obs or --var must be provided.",
        )
        raise typer.Exit(code=1)

    try:
        subset_h5ad(
            file=file,
            output=output,
            obs_file=obs,
            var_file=var,
            chunk_rows=chunk_rows,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)
