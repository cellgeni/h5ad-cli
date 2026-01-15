import sys
import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List

import rich
from rich.console import Console
import typer
import h5py
import numpy as np

from h5ad.commands import show_info

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
        "--cols",
        "-c",
        help="Comma separated column names to include in the output table",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output file path (defaults to stdout)", writable=True
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
    if axis not in ("obs", "var"):
        raise typer.BadParameter("axis must be 'obs' or 'var'.")

    col_list: List[str] = []
    if columns:
        col_list = [c for c in columns.split(",") if c]

    with h5py.File(file, "r") as f:
        group, n_rows, index_name = _get_axis_group(f, axis)

        # Determine columns to read
        if col_list:
            col_names = list(col_list)
        else:
            col_order = group.attrs.get("column-order", None)
            if col_order is not None:
                col_order = _decode_str_array(np.asarray(col_order))
                col_names = list(col_order.tolist())
            else:
                col_names = sorted(
                    name
                    for name in group.keys()
                    if isinstance(group[name], (h5py.Dataset, h5py.Group))
                    and name != index_name
                )

        if isinstance(index_name, bytes):
            index_name = index_name.decode("utf-8")

        if index_name not in col_names:
            col_names = [index_name] + col_names
        else:
            col_names = [index_name] + [c for c in col_names if c != index_name]

        # Limit rows if head option is specified
        if head is not None and head > 0:
            n_rows = min(n_rows, head)

        # Open writer
        if out is None or str(out) == "-":
            out_fh = sys.stdout
        else:
            out_fh = open(out, "w", newline="", encoding="utf8")
        writer = csv.writer(out_fh)

        # Write data in chunks
        try:
            writer.writerow(col_names)
            cat_cache: Dict[int, np.ndarray] = {}
            with console.status(
                f"[magenta]Exporting {axis} table...[/] to {'stdout' if out_fh is sys.stdout else out}"
            ) as status:
                for start in range(0, n_rows, chunk_rows):
                    end = min(start + chunk_rows, n_rows)
                    status.update(
                        f"[magenta]Exporting rows {start}-{end} of {n_rows}...[/]"
                    )
                    cols_data: List[List[str]] = []
                    # Read each column for the current chunk
                    for col in col_names:
                        cols_data.append(
                            _col_chunk_as_strings(group, col, start, end, cat_cache)
                        )
                    # Write rows
                    for row_idx in range(end - start):
                        row = [
                            cols_data[col_idx][row_idx]
                            for col_idx in range(len(col_names))
                        ]
                        writer.writerow(row)
        finally:
            if out_fh is not sys.stdout:
                out_fh.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)
