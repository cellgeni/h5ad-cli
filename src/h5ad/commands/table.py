import sys
import csv
from pathlib import Path
from typing import List, Optional, Dict

import h5py
import numpy as np
from rich.console import Console
from h5ad.info import get_axis_group
from h5ad.read import col_chunk_as_strings


def export_table(
    file: Path,
    axis: str,
    columns: Optional[List[str]],
    out: Optional[Path],
    chunk_rows: int,
    head: Optional[int],
    console: Console,
) -> None:
    """
    Export a table of the specified axis to CSV format.
    Args:
        file (Path): Path to the .h5ad file
        axis (str): Axis to read from ('obs' or 'var')
        columns (Optional[List[str]]): List of column names to include in the output table
        out (Optional[Path]): Output file path (defaults to stdout)
        chunk_rows (int): Number of rows to read per chunk
        head (Optional[int]): Output only the first n rows
    """
    with h5py.File(file, "r") as f:
        group, n_rows, index_name = get_axis_group(f, axis)

        # Determine columns to read
        if columns:
            col_names = list(columns)
        else:
            col_names = [k for k in group.keys() if k != "_index" and k != index_name]
            # Add index name if not already present
            if index_name and index_name not in col_names:
                col_names.insert(0, index_name)

        if isinstance(index_name, bytes):
            index_name = index_name.decode("utf-8")

        if index_name not in col_names:
            col_names.insert(0, index_name)
        else:
            col_names = [index_name] + [c for c in col_names if c != index_name]

        # Limit rows if head option is specified
        if head is not None and head > 0:
            n_rows = min(n_rows, head)

        # Open writer
        if out is None or str(out) == "-":
            out_fh = sys.stdout
        else:
            out_fh = open(out, "w", newline="", encoding="utf-8")
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
                            col_chunk_as_strings(group, col, start, end, cat_cache)
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
