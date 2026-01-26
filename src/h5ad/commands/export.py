from __future__ import annotations

import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import h5py
import numpy as np
from rich.console import Console

from h5ad.read import col_chunk_as_strings, decode_str_array
from h5ad.info import get_axis_group, get_entry_type


H5Obj = Union[h5py.Group, h5py.Dataset]


# ============================================================================
# DATAFRAME EXPORT (CSV)
# ============================================================================
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
    Export a dataframe (obs or var) to CSV format.

    Args:
        file: Path to the .h5ad file
        axis: Axis to read from ('obs' or 'var')
        columns: List of column names to include in the output table
        out: Output file path (defaults to stdout if None)
        chunk_rows: Number of rows to read per chunk
        head: Output only the first n rows
        console: Rich console for status output

    Supports both v0.2.0 (modern) and v0.1.0 (legacy) dataframe formats.
    """
    with h5py.File(file, "r") as f:
        group, n_rows, index_name = get_axis_group(f, axis)

        # Reserved keys to exclude from column list
        # __categories is used in v0.1.0 for storing categorical labels
        reserved_keys = {"_index", "__categories"}

        # Determine columns to read
        if columns:
            col_names = list(columns)
        else:
            col_names = [
                k for k in group.keys() if k not in reserved_keys and k != index_name
            ]
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

            # Use status spinner only when writing to file (not stdout)
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


# ============================================================================
# TYPE DETECTION AND VALIDATION
# ============================================================================
# Map object types to valid output extensions
TYPE_EXTENSIONS = {
    "dataframe": {".csv"},
    "sparse-matrix": {".mtx"},
    "dense-matrix": {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "array": {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "dict": {".json"},
    "scalar": {".json"},
    "categorical": {".csv"},
    "awkward-array": {".json"},
}

# Image extensions for validation
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Known exportable types
EXPORTABLE_TYPES = set(TYPE_EXTENSIONS.keys())


def _norm_path(p: str) -> str:
    p = p.strip()
    if not p:
        raise ValueError("Object path must be non-empty.")
    return p.lstrip("/")


def _get_encoding_type(group: h5py.Group) -> str:
    enc = group.attrs.get("encoding-type", "")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    return str(enc)


def _resolve(file: h5py.File, obj: str) -> H5Obj:
    obj = _norm_path(obj)
    if obj not in file:
        raise KeyError(f"'{obj}' not found in the file.")
    return cast(H5Obj, file[obj])


def _check_json_exportable(h5obj: H5Obj, max_elements: int, path: str = "") -> None:
    """
    Recursively check if a group/dataset can be exported to JSON.
    Raises ValueError if it contains non-exportable structures.
    """
    if isinstance(h5obj, h5py.Dataset):
        if h5obj.shape == ():
            return  # scalar is fine
        n = int(np.prod(h5obj.shape)) if h5obj.shape else 0
        if n > max_elements:
            raise ValueError(
                f"Cannot export to JSON: '{path or h5obj.name}' has {n} elements "
                f"(max {max_elements}). Use --max-elements to increase limit."
            )
        return

    # It's a Group - check encoding
    enc = _get_encoding_type(h5obj)
    if enc in ("csr_matrix", "csc_matrix"):
        raise ValueError(
            f"Cannot export to JSON: '{path or h5obj.name}' is a sparse matrix. "
            f"Export it as .mtx instead."
        )

    # Check children recursively
    for key in h5obj.keys():
        child = h5obj[key]
        child_path = f"{path}/{key}" if path else key
        if isinstance(child, (h5py.Group, h5py.Dataset)):
            _check_json_exportable(
                cast(H5Obj, child), max_elements=max_elements, path=child_path
            )


def export_npy(
    file: Path,
    obj: str,
    out: Path,
    chunk_elements: int,
    console: Console,
) -> None:
    """
    Export a dense HDF5 dataset to NumPy .npy without loading it all at once.

    Supports both:
    - v0.2.0 (modern): Datasets with encoding-type="array"
    - v0.1.0 (legacy): Plain datasets without encoding attributes
    - Encoded groups: nullable-integer, nullable-boolean, string-array (extracts values)

    Args:
        file: Path to the .h5ad file
        obj: HDF5 path to the dataset or encoded group
        out: Output .npy file path
        chunk_elements: Number of elements to read per chunk
        console: Rich console for status output

    Raises:
        ValueError: If the target object is not exportable as .npy
    """
    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)

        # Handle encoded groups that contain array data
        if isinstance(h5obj, h5py.Group):
            enc = _get_encoding_type(h5obj)
            if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
                # Extract values from nullable array group
                if "values" not in h5obj:
                    raise ValueError(
                        f"Encoded group '{obj}' is missing 'values' dataset."
                    )
                ds = h5obj["values"]
                has_mask = "mask" in h5obj
                console.print(f"[dim]Exporting nullable array values from '{obj}'[/]")
            else:
                raise ValueError(
                    f"Target '{obj}' is a group with encoding '{enc}'; cannot export as .npy directly."
                )
        else:
            ds = h5obj
            has_mask = False

        out.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=ds.dtype, shape=ds.shape)
        try:
            if ds.shape == ():
                mm[...] = ds[()]
                console.print(f"[green]Wrote[/] {out}")
                return

            if ds.ndim == 1:
                n = int(ds.shape[0])
                step = max(1, int(chunk_elements))
                with console.status(
                    f"[magenta]Exporting {obj} to {out}...[/]"
                ) as status:
                    for start in range(0, n, step):
                        end = min(start + step, n)
                        status.update(
                            f"[magenta]Exporting {obj}: {start}-{end} of {n}...[/]"
                        )
                        mm[start:end] = ds[start:end]
                console.print(f"[green]Wrote[/] {out}")
                return

            n0 = int(ds.shape[0])
            row_elems = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
            # Convert element budget into a row count; fallback to 1 row if rows are larger.
            step0 = max(1, int(chunk_elements) // max(1, row_elems))
            with console.status(f"[magenta]Exporting {obj} to {out}...[/]") as status:
                for start in range(0, n0, step0):
                    end = min(start + step0, n0)
                    status.update(
                        f"[magenta]Exporting {obj}: {start}-{end} of {n0}...[/]"
                    )
                    mm[start:end, ...] = ds[start:end, ...]
            console.print(f"[green]Wrote[/] {out}")
        finally:
            del mm


def export_mtx(
    file: Path,
    obj: str,
    out: Optional[Path],
    head: Optional[int],
    chunk_elements: int,
    in_memory: bool,
    console: Console,
) -> None:
    """Export a CSR/CSC matrix group (AnnData encoding) to Matrix Market (.mtx).

    If out is None or "-", writes to stdout. The head parameter limits output lines.
    chunk_elements controls how many rows/columns are processed per slice when
    streaming. Use in_memory for small matrices to load everything at once.

    Args:
        file: Path to the .h5ad file
        obj: HDF5 path to the matrix group
        out: Output .mtx file path (or None for stdout)
        head: Output only the first n nonzero entries
        chunk_elements: Number of rows/columns to process per chunk
        in_memory: Load the entire sparse matrix into memory before exporting
        console: Rich console for status output

    Raises:
        ValueError: If the target object is not a valid CSR/CSC matrix group.
    """
    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)
        if not isinstance(h5obj, h5py.Group):
            raise ValueError(
                "MTX export requires a CSR/CSC matrix group (not a dataset)."
            )

        enc = _get_encoding_type(h5obj)
        if enc not in ("csr_matrix", "csc_matrix"):
            raise ValueError(
                f"Target group encoding-type is {enc!r}; expected 'csr_matrix' or 'csc_matrix'."
            )

        data = h5obj.get("data")
        indices = h5obj.get("indices")
        indptr = h5obj.get("indptr")
        if (
            not isinstance(data, h5py.Dataset)
            or not isinstance(indices, h5py.Dataset)
            or not isinstance(indptr, h5py.Dataset)
        ):
            raise ValueError(
                "Sparse matrix group must contain datasets: data, indices, indptr"
            )

        shape = h5obj.attrs.get("shape", None)
        if shape is None:
            raise ValueError(
                "Sparse matrix group is missing required 'shape' attribute."
            )
        n_rows, n_cols = (int(shape[0]), int(shape[1]))

        field = "real" if np.issubdtype(data.dtype, np.floating) else "integer"

        # Load sparse index pointers (1 per major axis row/col); used to slice data/indices.
        indptr_arr = np.asarray(indptr[...], dtype=np.int64)
        nnz_ptr = int(indptr_arr[-1]) if indptr_arr.size else 0
        nnz_data = int(data.shape[0])
        nnz_idx = int(indices.shape[0])

        # Check consistency of sparse data
        if not (nnz_ptr == nnz_data == nnz_idx):
            raise ValueError(
                f"Sparse matrix data inconsistency: indptr implies {nnz_ptr} nonzeros, "
                f"but data has {nnz_data} and indices has {nnz_idx}."
            )

        # Determine number of nonzero entries to write
        nnz = nnz_data
        major_step = max(1, int(chunk_elements))
        if head is not None and head > 0:
            nnz = min(nnz_data, head)

        # Write to stdout when out is None or "-", otherwise open a file on disk.
        if out is None or str(out) == "-":
            out_fh = sys.stdout
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(out, "w", encoding="utf-8", newline="\n")

        use_status = out_fh is not sys.stdout
        status_ctx = (
            console.status(f"[magenta]Exporting {obj} to {out}...[/]")
            if use_status
            else nullcontext()
        )
        try:
            # Matrix Market header: type, generator line, then shape and nnz.
            out_fh.write(f"%%MatrixMarket matrix coordinate {field} general\n")
            out_fh.write("% generated by h5ad-cli\n")
            if head is not None and head > 0:
                out_fh.write(
                    f"% output limited to first {nnz}/{nnz_data} nonzero entries\n"
                )
            out_fh.write(f"{n_rows} {n_cols} {nnz}\n")

            if in_memory:
                with status_ctx as status:
                    if use_status and status:
                        status.update(
                            f"[magenta]Loading entire matrix {obj} into memory...[/]"
                        )
                    data_arr = np.asarray(data[...])
                    indices_arr = np.asarray(indices[...], dtype=np.int64)
                    counts = np.diff(indptr_arr)
                    if int(counts.sum()) != nnz_data:
                        raise ValueError(
                            "Sparse matrix indptr does not match data/indices length."
                        )

                    if enc == "csr_matrix":
                        major_idx = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
                        row_idx = major_idx
                        col_idx = indices_arr
                    else:
                        major_idx = np.repeat(np.arange(n_cols, dtype=np.int64), counts)
                        row_idx = indices_arr
                        col_idx = major_idx

                    if head is not None and head > 0:
                        row_idx = row_idx[:nnz]
                        col_idx = col_idx[:nnz]
                        data_arr = data_arr[:nnz]

                    data_fmt = "%.18g" if field == "real" else "%d"
                    coords = np.column_stack((row_idx + 1, col_idx + 1, data_arr))
                    if use_status and status:
                        status.update(f"[magenta]Saving {nnz} entries to {out}...[/]")
                    np.savetxt(out_fh, coords, fmt=["%d", "%d", data_fmt], newline="\n")
            else:
                # Iterate over major axis (rows for CSR, cols for CSC)
                major = n_rows if enc == "csr_matrix" else n_cols
                max_lines = head if head is not None and head > 0 else None
                written = 0
                with status_ctx as status:
                    for major_start in range(0, major, major_step):
                        major_end = min(major_start + major_step, major)
                        if use_status and status:
                            status.update(
                                f"[magenta]Exporting {obj}: {major_start+1}-{major_end} of {major}...[/]"
                            )
                        for major_i in range(major_start, major_end):
                            start = min(int(indptr_arr[major_i]), nnz_data)
                            end = min(int(indptr_arr[major_i + 1]), nnz_data)
                            if end <= start:
                                continue
                            idx = np.asarray(indices[start:end], dtype=np.int64)
                            vals = np.asarray(data[start:end])
                            m = min(len(idx), len(vals))
                            if m == 0:
                                raise ValueError("Sparse matrix chunk has zero length.")
                            if max_lines is not None:
                                remaining = max_lines - written
                                if remaining <= 0:
                                    break
                                if m > remaining:
                                    m = remaining
                            idx = idx[:m]
                            vals = vals[:m]
                            idx_list = idx.tolist()
                            vals_list = vals.tolist()
                            if enc == "csr_matrix":
                                r = major_i + 1
                                lines = [
                                    f"{r} {c + 1} {v}\n"
                                    for c, v in zip(idx_list, vals_list)
                                ]
                            else:
                                c = major_i + 1
                                lines = [
                                    f"{r + 1} {c} {v}\n"
                                    for r, v in zip(idx_list, vals_list)
                                ]
                            out_fh.write("".join(lines))
                            written += m
                            if max_lines is not None and written >= max_lines:
                                break
                        if max_lines is not None and written >= max_lines:
                            break
        finally:
            if out_fh is not sys.stdout:
                out_fh.close()
        if out_fh is not sys.stdout:
            console.print(f"[green]Wrote[/] {out}")


def export_json(
    file: Path,
    obj: str,
    out: Path,
    max_elements: int,
    include_attrs: bool,
    console: Console,
) -> None:
    """Export an HDF5 group/dataset to JSON (best-effort, with size limits)."""
    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)

        # Check if exportable before attempting
        _check_json_exportable(h5obj, max_elements=max_elements)

        payload = _to_jsonable(
            h5obj, max_elements=max_elements, include_attrs=include_attrs
        )
        # Write to stdout when out is None or "-", otherwise open a file on disk.
        if out is None or str(out) == "-":
            out_fh = sys.stdout
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(out, "w", encoding="utf-8")
        try:
            json.dump(payload, out_fh, indent=2, ensure_ascii=False, sort_keys=True)
            out_fh.write("\n")
        finally:
            if out_fh is not sys.stdout:
                out_fh.close()
        if out_fh is not sys.stdout:
            console.print(f"[green]Wrote[/] {out}")


def _attrs_to_jsonable(
    attrs: h5py.AttributeManager, max_elements: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in attrs.keys():
        v = attrs.get(k)
        out[str(k)] = _pyify(v, max_elements=max_elements)
    return out


def _pyify(value: Any, max_elements: int) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > max_elements:
            raise ValueError(
                f"Refusing to convert array of size {value.size} (> {max_elements}) to JSON."
            )
        if np.issubdtype(value.dtype, np.bytes_) or value.dtype.kind == "O":
            value = decode_str_array(value)
        return value.tolist()
    return value


def _dataset_to_jsonable(ds: h5py.Dataset, max_elements: int) -> Any:
    if ds.shape == ():
        v = ds[()]
        return _pyify(v, max_elements=max_elements)
    n = int(np.prod(ds.shape)) if ds.shape else 0
    if n > max_elements:
        raise ValueError(
            f"Refusing to convert dataset {ds.name!r} with {n} elements (> {max_elements}) to JSON."
        )
    arr = np.asarray(ds[...])
    return _pyify(arr, max_elements=max_elements)


def _to_jsonable(h5obj: H5Obj, max_elements: int, include_attrs: bool) -> Any:
    if isinstance(h5obj, h5py.Dataset):
        return _dataset_to_jsonable(h5obj, max_elements=max_elements)

    # Group
    d: Dict[str, Any] = {}
    if include_attrs and len(h5obj.attrs):
        d["__attrs__"] = _attrs_to_jsonable(h5obj.attrs, max_elements=max_elements)

    for key in h5obj.keys():
        child = h5obj[key]
        if isinstance(child, (h5py.Group, h5py.Dataset)):
            d[str(key)] = _to_jsonable(
                cast(H5Obj, child),
                max_elements=max_elements,
                include_attrs=include_attrs,
            )
    return d


def export_image(file: Path, obj: str, out: Path, console: Console) -> None:
    """Export an image-like dataset (H,W) or (H,W,C) to PNG/JPG/TIFF."""
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Pillow is required for image export. Install with: pip install h5ad[images]"
        ) from e

    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)
        if not isinstance(h5obj, h5py.Dataset):
            raise ValueError("Image export requires a dataset.")
        arr = np.asarray(h5obj[...])

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array; got shape {arr.shape}.")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Expected last dimension (channels) to be 1, 3, or 4; got {arr.shape}."
        )

    # Convert to uint8 for common image formats
    if np.issubdtype(arr.dtype, np.floating):
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image dtype: {arr.dtype}")

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    img = Image.fromarray(arr)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]Wrote[/] {out}")
