"""Import command for creating/replacing objects in h5ad files."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import h5py
import numpy as np
from rich.console import Console


# Map file extensions to expected input formats
EXTENSION_FORMAT = {
    ".csv": "csv",
    ".npy": "npy",
    ".mtx": "mtx",
    ".json": "json",
}

# Define which object paths expect which dimensions
# obs-axis: first dimension must match n_obs
# var-axis: first dimension must match n_var
# matrix: must match (n_obs, n_var)
OBS_AXIS_PREFIXES = ("obs", "obsm/", "obsp/")
VAR_AXIS_PREFIXES = ("var", "varm/", "varp/")
MATRIX_PREFIXES = ("X", "layers/")


def _norm_path(p: str) -> str:
    p = p.strip()
    if not p:
        raise ValueError("Object path must be non-empty.")
    return p.lstrip("/")


def _get_axis_length(file: h5py.File, axis: str) -> Optional[int]:
    """Get the length of obs or var axis."""
    if axis not in file:
        return None
    group = file[axis]
    if not isinstance(group, h5py.Group):
        return None
    index_name = group.attrs.get("_index", None)
    if index_name is None:
        index_name = "obs_names" if axis == "obs" else "var_names"
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    if index_name not in group:
        return None
    dataset = group[index_name]
    if isinstance(dataset, h5py.Dataset) and dataset.shape:
        return int(dataset.shape[0])
    return None


def _validate_dimensions(
    file: h5py.File,
    obj_path: str,
    data_shape: tuple,
    console: Console,
) -> None:
    """Validate that data dimensions match the target path requirements."""
    n_obs = _get_axis_length(file, "obs")
    n_var = _get_axis_length(file, "var")

    # Check obs/var replacement (dataframe)
    if obj_path == "obs":
        if n_obs is not None and data_shape[0] != n_obs:
            raise ValueError(
                f"Row count mismatch: input has {data_shape[0]} rows, "
                f"but obs has {n_obs} cells."
            )
        return
    if obj_path == "var":
        if n_var is not None and data_shape[0] != n_var:
            raise ValueError(
                f"Row count mismatch: input has {data_shape[0]} rows, "
                f"but var has {n_var} features."
            )
        return

    # Check matrix (X, layers/*)
    for prefix in MATRIX_PREFIXES:
        if (
            obj_path == prefix
            or obj_path.startswith(prefix + "/")
            or obj_path.startswith(prefix)
        ):
            if obj_path == "X" or obj_path.startswith("layers/"):
                if len(data_shape) < 2:
                    raise ValueError(
                        f"Matrix data requires 2D shape, got {len(data_shape)}D."
                    )
                if n_obs is not None and data_shape[0] != n_obs:
                    raise ValueError(
                        f"First dimension mismatch: input has {data_shape[0]} rows, "
                        f"but obs has {n_obs} cells."
                    )
                if n_var is not None and data_shape[1] != n_var:
                    raise ValueError(
                        f"Second dimension mismatch: input has {data_shape[1]} columns, "
                        f"but var has {n_var} features."
                    )
                return

    # Check obs-axis matrices (obsm/*, obsp/*)
    for prefix in OBS_AXIS_PREFIXES:
        if obj_path.startswith(prefix) and obj_path != "obs":
            if n_obs is not None and data_shape[0] != n_obs:
                raise ValueError(
                    f"First dimension mismatch: input has {data_shape[0]} rows, "
                    f"but obs has {n_obs} cells."
                )
            # obsp should be square n_obs x n_obs
            if obj_path.startswith("obsp/") and len(data_shape) >= 2:
                if data_shape[1] != n_obs:
                    raise ValueError(
                        f"obsp matrix must be square (n_obs × n_obs): "
                        f"got {data_shape[0]}×{data_shape[1]}, expected {n_obs}×{n_obs}."
                    )
            return

    # Check var-axis matrices (varm/*, varp/*)
    for prefix in VAR_AXIS_PREFIXES:
        if obj_path.startswith(prefix) and obj_path != "var":
            if n_var is not None and data_shape[0] != n_var:
                raise ValueError(
                    f"First dimension mismatch: input has {data_shape[0]} rows, "
                    f"but var has {n_var} features."
                )
            # varp should be square n_var x n_var
            if obj_path.startswith("varp/") and len(data_shape) >= 2:
                if data_shape[1] != n_var:
                    raise ValueError(
                        f"varp matrix must be square (n_var × n_var): "
                        f"got {data_shape[0]}×{data_shape[1]}, expected {n_var}×{n_var}."
                    )
            return

    # For other paths (like uns/*), no dimension validation
    console.print(f"[dim]Note: No dimension validation for path '{obj_path}'[/]")


def _read_csv(
    input_file: Path,
    index_column: Optional[str],
) -> Tuple[List[dict], List[str], List[str], str]:
    """
    Read CSV file and return rows, column names, index values, and index column name.

    Returns:
        (rows, column_names, index_values, index_column_name)
    """
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")
        fieldnames = list(reader.fieldnames)

        # Determine index column
        if index_column:
            if index_column not in fieldnames:
                raise ValueError(
                    f"Index column '{index_column}' not found in CSV. "
                    f"Available columns: {', '.join(fieldnames)}"
                )
            idx_col = index_column
        else:
            idx_col = fieldnames[0]

        # Read all rows
        rows = list(reader)

    index_values = [row[idx_col] for row in rows]
    data_columns = [c for c in fieldnames if c != idx_col]

    return rows, data_columns, index_values, idx_col


def _read_mtx(
    input_file: Path,
) -> Tuple[List[Tuple[int, int, float]], Tuple[int, int], int]:
    """
    Read Matrix Market file and return sparse matrix data.

    Returns:
        (data, indices, indptr, shape, nnz, is_csr)
    """
    with open(input_file, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError("Invalid MTX file: missing MatrixMarket header.")

        # Parse header for field type
        parts = header.lower().split()
        field = "real"
        for p in parts:
            if p in ("real", "integer", "complex", "pattern"):
                field = p
                break

        # Skip comments
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()

        # Read dimensions
        dims = line.split()
        n_rows, n_cols, nnz = int(dims[0]), int(dims[1]), int(dims[2])

        # Read entries
        entries = []
        for _ in range(nnz):
            parts = fh.readline().split()
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            if field == "pattern":
                v = 1.0
            else:
                v = float(parts[2])
            entries.append((r, c, v))

    return entries, (n_rows, n_cols), nnz


def _create_csr_from_entries(
    entries: List[Tuple[int, int, float]], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert coordinate entries to CSR format."""
    n_rows, _ = shape
    # Sort by row, then column
    entries.sort(key=lambda x: (x[0], x[1]))

    data = np.array([e[2] for e in entries], dtype=np.float32)
    indices = np.array([e[1] for e in entries], dtype=np.int32)

    # Build indptr
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    for r, _, _ in entries:
        indptr[r + 1] += 1
    indptr = np.cumsum(indptr)

    return data, indices, indptr


def import_object(
    file: Path,
    obj: str,
    input_file: Path,
    output_file: Optional[Path],
    inplace: bool,
    index_column: Optional[str],
    console: Console,
) -> None:
    """
    Import data from a file into an h5ad object.

    Args:
        file: Path to the source h5ad file
        obj: Object path to create/replace (e.g., 'obs', 'obsm/X_pca', 'X')
        input_file: Input data file (.csv, .npy, .mtx, .json)
        output_file: Path to output h5ad file (None if inplace)
        inplace: If True, modify the source file directly
        index_column: Column to use as index for obs/var CSV import
        console: Console for output
    """
    # Determine target file
    if inplace:
        target_file = file
    else:
        if output_file is None:
            raise ValueError("Output file is required unless --inplace is specified.")
        # Copy source to output first
        shutil.copy2(file, output_file)
        target_file = output_file
        console.print(f"[dim]Copied {file} → {output_file}[/]")

    obj = _norm_path(obj)
    ext = input_file.suffix.lower()

    if ext not in EXTENSION_FORMAT:
        raise ValueError(
            f"Unsupported input file extension '{ext}'. "
            f"Supported: {', '.join(sorted(EXTENSION_FORMAT.keys()))}"
        )

    fmt = EXTENSION_FORMAT[ext]

    # Validate index_column is only used for obs/var CSV
    if index_column and (fmt != "csv" or obj not in ("obs", "var")):
        raise ValueError(
            "--index-column is only valid for CSV import into 'obs' or 'var'."
        )

    if fmt == "csv":
        _import_csv(target_file, obj, input_file, index_column, console)
    elif fmt == "npy":
        _import_npy(target_file, obj, input_file, console)
    elif fmt == "mtx":
        _import_mtx(target_file, obj, input_file, console)
    elif fmt == "json":
        _import_json(target_file, obj, input_file, console)


def _import_csv(
    file: Path,
    obj: str,
    input_file: Path,
    index_column: Optional[str],
    console: Console,
) -> None:
    """Import CSV data into obs or var."""
    if obj not in ("obs", "var"):
        raise ValueError(
            f"CSV import is only supported for 'obs' or 'var', not '{obj}'."
        )

    rows, data_columns, index_values, _ = _read_csv(input_file, index_column)
    n_rows = len(rows)

    with h5py.File(file, "a") as f:
        # Validate dimensions if the file already has obs/var
        _validate_dimensions(f, obj, (n_rows,), console)

        # Delete existing group if present
        if obj in f:
            del f[obj]

        # Create new group
        group = f.create_group(obj)
        index_name = "obs_names" if obj == "obs" else "var_names"
        group.attrs["_index"] = index_name
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.2.0"
        group.attrs["column-order"] = np.array(data_columns, dtype="S")

        # Create index dataset
        group.create_dataset(
            index_name,
            data=np.array(index_values, dtype="S"),
        )

        # Create column datasets
        for col in data_columns:
            values = [row[col] for row in rows]
            # Try to infer type
            try:
                arr = np.array(values, dtype=np.float64)
                group.create_dataset(col, data=arr)
            except (ValueError, TypeError):
                try:
                    arr = np.array(values, dtype=np.int64)
                    group.create_dataset(col, data=arr)
                except (ValueError, TypeError):
                    # Fallback to string
                    arr = np.array(values, dtype="S")
                    ds = group.create_dataset(col, data=arr)
                    ds.attrs["encoding-type"] = "string-array"
                    ds.attrs["encoding-version"] = "0.2.0"

    console.print(
        f"[green]Imported[/] {n_rows} rows × {len(data_columns)} columns into '{obj}'"
    )


def _import_npy(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    """Import NPY data into a dataset."""
    arr = np.load(input_file)

    with h5py.File(file, "a") as f:
        _validate_dimensions(f, obj, arr.shape, console)

        # Handle nested paths
        parts = obj.split("/")
        parent_path = "/".join(parts[:-1])
        name = parts[-1]

        # Ensure parent groups exist
        if parent_path:
            if parent_path not in f:
                f.create_group(parent_path)
            parent = cast(h5py.Group, f[parent_path])
        else:
            parent = f

        # Delete existing if present
        if name in parent:
            del parent[name]

        # Create dataset
        parent.create_dataset(name, data=arr)

    shape_str = "×".join(str(d) for d in arr.shape)
    console.print(f"[green]Imported[/] {shape_str} array into '{obj}'")


def _import_mtx(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    """Import MTX (Matrix Market) data as CSR sparse matrix."""
    entries, shape, nnz = _read_mtx(input_file)
    data, indices, indptr = _create_csr_from_entries(entries, shape)

    with h5py.File(file, "a") as f:
        _validate_dimensions(f, obj, shape, console)

        # Handle nested paths
        parts = obj.split("/")
        parent_path = "/".join(parts[:-1])
        name = parts[-1]

        if parent_path:
            if parent_path not in f:
                f.create_group(parent_path)
            parent = cast(h5py.Group, f[parent_path])
        else:
            parent = f

        # Delete existing if present
        if name in parent:
            del parent[name]

        # Create sparse matrix group
        group = parent.create_group(name)
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.array(shape, dtype=np.int64)

        group.create_dataset("data", data=data)
        group.create_dataset("indices", data=indices)
        group.create_dataset("indptr", data=indptr)

    console.print(
        f"[green]Imported[/] {shape[0]}×{shape[1]} sparse matrix ({nnz} non-zero) into '{obj}'"
    )


def _import_json(
    file: Path,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    """Import JSON data into uns or other dict-like groups."""
    with open(input_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    with h5py.File(file, "a") as f:
        # Handle nested paths
        parts = obj.split("/")
        parent_path = "/".join(parts[:-1])
        name = parts[-1]

        if parent_path:
            if parent_path not in f:
                f.create_group(parent_path)
            parent = cast(h5py.Group, f[parent_path])
        else:
            parent = f

        # Delete existing if present
        if name in parent:
            del parent[name]

        # Create from JSON
        _write_json_to_h5(parent, name, payload)

    console.print(f"[green]Imported[/] JSON data into '{obj}'")


def _write_json_to_h5(parent: h5py.Group, name: str, value: Any) -> None:
    """Recursively write JSON-like data to HDF5."""
    if isinstance(value, dict):
        group = parent.create_group(name)
        for k, v in value.items():
            _write_json_to_h5(group, k, v)
    elif isinstance(value, list):
        # Try to convert to array
        try:
            arr = np.array(value)
            if arr.dtype.kind in ("U", "O"):
                arr = np.array(value, dtype="S")
            parent.create_dataset(name, data=arr)
        except (ValueError, TypeError):
            # Fallback: store as JSON string
            parent.create_dataset(name, data=json.dumps(value).encode("utf-8"))
    elif isinstance(value, str):
        parent.create_dataset(name, data=np.array([value], dtype="S"))
    elif isinstance(value, bool):
        parent.create_dataset(name, data=np.array(value, dtype=bool))
    elif isinstance(value, int):
        parent.create_dataset(name, data=np.array(value, dtype=np.int64))
    elif isinstance(value, float):
        parent.create_dataset(name, data=np.array(value, dtype=np.float64))
    elif value is None:
        # Store None as empty string attribute or special marker
        ds = parent.create_dataset(name, data=np.array([], dtype="S"))
        ds.attrs["_is_none"] = True
    else:
        raise ValueError(f"Cannot convert JSON value of type {type(value).__name__}")
