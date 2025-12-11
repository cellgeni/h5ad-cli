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
    help="Streaming CLI for huge .h5ad files (info, ls, table, matrix, subset-obs-range)."
)
console = Console(stderr=True)


def open_file(path: Path) -> h5py.File:
    """Open an h5ad file in read-only mode."""
    return h5py.File(path, "r")


def axis_len(file: h5py.File, axis: str) -> Optional[int]:
    """Get the length of the specified axis ('obs' or 'var') in the h5ad file."""
    # Check if the specified axis exists in the file
    if axis not in file:
        return None

    # Get the group corresponding to the axis
    group = file[axis]
    if not isinstance(group, h5py.Group):
        return None

    # Determine the index name for the axis
    index_name = group.attrs.get("_index", None)
    if index_name is None:
        if axis == "obs":
            index_name = "obs_names"
        elif axis == "var":
            index_name = "var_names"
        else:
            return None

    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")

    if index_name not in group:
        return None

    # Return the length of the index dataset
    dataset = group[index_name]
    if not isinstance(dataset, h5py.Dataset):
        return None
    if dataset.shape:
        return int(dataset.shape[0])
    return None


def _decode_str_array(array: np.ndarray) -> np.ndarray:
    """Turn HDF5 bytes/objects into plain unicode strings."""
    if np.issubdtype(array.dtype, np.bytes_):
        return array.astype("U")
    if array.dtype.kind == "O":
        return array.astype(str)
    return array.astype(str)


@app.command()
def info(file: Path = typer.Argument(..., help="Path to the .h5ad file")) -> None:
    """
    Show high-level information about the .h5ad file.
    """
    with open_file(file) as f:
        # Get n_obs and n_var
        n_obs = axis_len(f, "obs")
        n_var = axis_len(f, "var")
        rich.print(
            f"[bold cyan]An object with n_obs × n_var: {n_obs if n_obs is not None else '?'} × {n_var if n_var is not None else '?'}[/]"
        )
        # List top-level keys and their sub-keys
        for key, obj in sorted(f.items(), key=lambda x: len(x[0])):
            sub_keys = [k for k in obj.keys() if k != "_index"]
            if sub_keys and key != "X":
                rich.print(
                    f"\t[bold yellow]{key}:[/]\t"
                    + ", ".join(f"[bright_white]{sub}[/]" for sub in sub_keys)
                )


def _get_axis_group(file: h5py.File, axis: str) -> Tuple[h5py.Group, int, str]:
    """"""
    if axis not in ("obs", "var"):
        raise ValueError("axis must be 'obs' or 'var'.")
    if axis not in file:
        raise KeyError(f"'{axis}' not found in the file.")

    group = file[axis]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"'{axis}' is not a group.")

    n = axis_len(file, axis)
    if n is None:
        raise RuntimeError(f"Could not determine length of axis '{axis}'.")

    index_name = group.attrs.get("_index", None)
    if index_name is None:
        index_name = "obs_names" if axis == "obs" else "var_names"
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    return group, n, index_name


def _read_categorical_column(
    col_group: h5py.Group, start: int, end: int, cache: Dict[int, np.ndarray]
) -> List[str]:
    """
    Decode an AnnData 'categorical' column for a slice [start:end].
    """
    key = id(col_group)
    if key not in cache:
        cats = col_group["categories"][...]
        cats = _decode_str_array(cats)
        cache[key] = np.asarray(cats, dtype=str)
    cats = cache[key]

    codes_ds = col_group["codes"]
    codes = codes_ds[start:end]
    codes = np.asarray(codes, dtype=np.int64)
    return [cats[c] if 0 <= c < len(cats) else "" for c in codes]


def _col_chunk_as_strings(
    group: h5py.Group,
    col_name: str,
    start: int,
    end: int,
    cat_cache: Dict[int, np.ndarray],
) -> List[str]:
    """Read a column from an obs/var group as strings."""
    if col_name in group and isinstance(group[col_name], h5py.Dataset):
        dataset = group[col_name]
        chunk = dataset[start:end]
        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)
        chunk = _decode_str_array(np.asarray(chunk))
        return chunk.tolist()

    if col_name in group and isinstance(group[col_name], h5py.Group):
        col_group = group[col_name]
        enc = col_group.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")
        if enc == "categorical":
            return _read_categorical_column(col_group, start, end, cat_cache)

    raise RuntimeError(f"Unsupported column {col_name!r} in group {group.name}")


@app.command()
def table(
    file: Path = typer.Argument(..., help="Path to the .h5ad file"),
    axis: str = typer.Option("obs", help="Axis to read from ('obs' or 'var')"),
    columns: Optional[str] = typer.Option(
        None, "--cols", "-c", help="Columns to include in the output table"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output file path (defaults to stdout)"
    ),
    chunk_rows: int = typer.Option(
        10000, "--chunk-rows", "-r", help="Number of rows to read per chunk"
    ),
) -> None:
    """
    Export a table of the specified axis ('obs' or 'var') to CSV format.
    """
    if axis not in ("obs", "var"):
        raise typer.BadParameter("axis must be 'obs' or 'var'.")

    col_list: List[str] = []
    if columns:
        col_list = [c for c in columns.split(",") if c]

    with open_file(file) as f:
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
