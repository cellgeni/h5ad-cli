from __future__ import annotations

import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from rich.console import Console

from h5ad.core.info import get_axis_group
from h5ad.core.read import col_chunk_as_strings
from h5ad.formats.validate import validate_dimensions
from h5ad.storage import create_dataset, is_zarr_group


def export_dataframe(
    root: Any,
    axis: str,
    columns: Optional[List[str]],
    out: Optional[Path],
    chunk_rows: int,
    head: Optional[int],
    console: Console,
) -> None:
    group, n_rows, index_name = get_axis_group(root, axis)

    reserved_keys = {"_index", "__categories"}

    if columns:
        col_names = list(columns)
    else:
        col_names = [
            k for k in group.keys() if k not in reserved_keys and k != index_name
        ]
        if index_name and index_name not in col_names:
            col_names.insert(0, index_name)

    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")

    if index_name not in col_names:
        col_names.insert(0, index_name)
    else:
        col_names = [index_name] + [c for c in col_names if c != index_name]

    if head is not None and head > 0:
        n_rows = min(n_rows, head)

    if out is None or str(out) == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_fh)

    try:
        writer.writerow(col_names)
        cat_cache = {}

        use_status = out_fh is not sys.stdout
        status_ctx = (
            console.status(f"[magenta]Exporting {axis} table to {out}...[/]")
            if use_status
            else nullcontext()
        )

        with status_ctx as status:
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                if use_status and status:
                    status.update(
                        f"[magenta]Exporting rows {start}-{end} of {n_rows}...[/]"
                    )
                cols_data: List[List[str]] = []
                for col in col_names:
                    cols_data.append(
                        col_chunk_as_strings(group, col, start, end, cat_cache)
                    )
                for row_idx in range(end - start):
                    row = [
                        cols_data[col_idx][row_idx]
                        for col_idx in range(len(col_names))
                    ]
                    writer.writerow(row)
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


def _read_csv(
    input_file: Path,
    index_column: Optional[str],
) -> Tuple[List[dict], List[str], List[str], str]:
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")
        fieldnames = list(reader.fieldnames)

        if index_column:
            if index_column not in fieldnames:
                raise ValueError(
                    f"Index column '{index_column}' not found in CSV. "
                    f"Available columns: {', '.join(fieldnames)}"
                )
            idx_col = index_column
        else:
            idx_col = fieldnames[0]

        rows = list(reader)

    index_values = [row[idx_col] for row in rows]
    data_columns = [c for c in fieldnames if c != idx_col]

    return rows, data_columns, index_values, idx_col


def import_dataframe(
    root: Any,
    obj: str,
    input_file: Path,
    index_column: Optional[str],
    console: Console,
) -> None:
    if obj not in ("obs", "var"):
        raise ValueError(
            f"CSV import is only supported for 'obs' or 'var', not '{obj}'."
        )

    rows, data_columns, index_values, _ = _read_csv(input_file, index_column)
    n_rows = len(rows)

    validate_dimensions(root, obj, (n_rows,), console)

    if obj in root:
        del root[obj]

    group = root.create_group(obj)
    index_name = "obs_names" if obj == "obs" else "var_names"
    group.attrs["_index"] = index_name
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"

    if is_zarr_group(group):
        group.attrs["column-order"] = list(data_columns)
    else:
        group.attrs["column-order"] = np.array(data_columns, dtype="S")

    create_dataset(group, index_name, data=np.array(index_values, dtype="S"))

    for col in data_columns:
        values = [row[col] for row in rows]
        try:
            arr = np.array(values, dtype=np.float64)
            create_dataset(group, col, data=arr)
        except (ValueError, TypeError):
            try:
                arr = np.array(values, dtype=np.int64)
                create_dataset(group, col, data=arr)
            except (ValueError, TypeError):
                arr = np.array(values, dtype="S")
                ds = create_dataset(group, col, data=arr)
                ds.attrs["encoding-type"] = "string-array"
                ds.attrs["encoding-version"] = "0.2.0"

    console.print(
        f"[green]Imported[/] {n_rows} rows × {len(data_columns)} columns into '{obj}'"
    )
