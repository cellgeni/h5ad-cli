import sys
import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List

import rich
from rich.console import Console
import typer
import h5py
import numpy as np

from h5ad.commands import show_info, export_table

app = typer.Typer(
    help="Streaming CLI for huge .h5ad files (info, ls, table, matrix, subset-obs-range)."
)
console = Console(stderr=True)


@app.command()
def info(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad file",
        exists=True,
        readable=True,
    )
) -> None:
    """
    Show high-level information about the .h5ad file.
    Args:
        file (Path): Path to the .h5ad file
    """
    show_info(file, console)


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


def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)
