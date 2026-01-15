from __future__ import annotations

import csv
import json
import sys
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


def export_object(
    file: Path,
    obj: str,
    out: Path,
    columns: Optional[List[str]],
    chunk_rows: int,
    head: Optional[int],
    max_elements: int,
    include_attrs: bool,
    console: Console,
) -> None:
    """
    Export an HDF5 object to an appropriate format based on its type.

    Auto-detects the object type and validates the output file extension.
    """
    obj = _norm_path(obj)
    out_ext = out.suffix.lower()

    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)
        info = get_entry_type(h5obj)
        obj_type = info["type"]

        # Check if type is exportable
        if obj_type not in EXPORTABLE_TYPES:
            raise ValueError(
                f"Cannot export object of type '{obj_type}'. "
                f"Exportable types: {', '.join(sorted(EXPORTABLE_TYPES))}."
            )

        # Check if extension matches the type
        valid_exts = TYPE_EXTENSIONS.get(obj_type, set())
        if out_ext not in valid_exts:
            ext_list = ", ".join(sorted(valid_exts))
            raise ValueError(
                f"Output extension '{out_ext}' does not match object type '{obj_type}'. "
                f"Expected: {ext_list}."
            )

    # Dispatch to appropriate export function
    if obj_type == "dataframe":
        # For dataframe, obj must be obs or var
        if obj not in ("obs", "var"):
            raise ValueError(
                f"CSV export for dataframes currently supports only 'obs' or 'var', "
                f"not '{obj}'."
            )
        export_table(
            file=file,
            axis=obj,
            columns=columns,
            out=out,
            chunk_rows=chunk_rows,
            head=head,
            console=console,
        )

    elif obj_type == "categorical":
        # Categorical is also exported via table if it's a column in obs/var
        raise ValueError(
            f"Categorical objects should be exported as part of 'obs' or 'var' table. "
            f"Use: h5ad export <file> obs <output.csv>"
        )

    elif obj_type in ("dense-matrix", "array"):
        if out_ext in IMAGE_EXTENSIONS:
            # User wants image output - validate dimensions
            _export_image(file=file, obj=obj, out=out, console=console)
        else:
            _export_npy(
                file=file, obj=obj, out=out, chunk_rows=chunk_rows, console=console
            )

    elif obj_type == "sparse-matrix":
        _export_mtx(file=file, obj=obj, out=out, console=console)

    elif obj_type in ("dict", "scalar"):
        _export_json(
            file=file,
            obj=obj,
            out=out,
            max_elements=max_elements,
            include_attrs=include_attrs,
            console=console,
        )


def _export_npy(
    file: Path,
    obj: str,
    out: Path,
    chunk_rows: int,
    console: Console,
) -> None:
    """Export a dense HDF5 dataset to NumPy .npy without loading it all at once."""
    with h5py.File(file, "r") as f:
        h5obj = _resolve(f, obj)
        if isinstance(h5obj, h5py.Group):
            raise ValueError("Target is a group; cannot export as .npy.")

        ds = h5obj
        out.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=ds.dtype, shape=ds.shape)
        try:
            if ds.shape == ():
                mm[...] = ds[()]
                console.print(f"[green]Wrote[/] {out}")
                return

            if ds.ndim == 1:
                n = int(ds.shape[0])
                step = max(1, int(chunk_rows))
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
            step0 = max(1, int(chunk_rows))
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


def _export_mtx(file: Path, obj: str, out: Path, console: Console) -> None:
    """Export a CSR/CSC matrix group (AnnData encoding) to Matrix Market (.mtx)."""
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
            raise RuntimeError(
                "Sparse matrix group must contain datasets: data, indices, indptr"
            )

        shape = h5obj.attrs.get("shape", None)
        if shape is None:
            raise RuntimeError(
                "Sparse matrix group is missing required 'shape' attribute."
            )
        n_rows, n_cols = (int(shape[0]), int(shape[1]))

        field = "real" if np.issubdtype(data.dtype, np.floating) else "integer"

        out.parent.mkdir(parents=True, exist_ok=True)

        indptr_arr = np.asarray(indptr[...], dtype=np.int64)
        nnz_ptr = int(indptr_arr[-1]) if indptr_arr.size else 0
        nnz_data = int(data.shape[0])
        nnz_idx = int(indices.shape[0])
        nnz = min(nnz_ptr, nnz_data, nnz_idx)

        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"%%MatrixMarket matrix coordinate {field} general\n")
            fh.write("% generated by h5ad-cli\n")
            fh.write(f"{n_rows} {n_cols} {nnz}\n")

            major = n_rows if enc == "csr_matrix" else n_cols
            with console.status(f"[magenta]Exporting {obj} to {out}...[/]") as status:
                for major_i in range(major):
                    start = min(int(indptr_arr[major_i]), nnz)
                    end = min(int(indptr_arr[major_i + 1]), nnz)
                    if end <= start:
                        continue
                    status.update(
                        f"[magenta]Exporting {obj}: block {major_i+1}/{major}...[/]"
                    )
                    idx = np.asarray(indices[start:end], dtype=np.int64)
                    vals = np.asarray(data[start:end])
                    m = min(len(idx), len(vals))
                    if m == 0:
                        continue
                    idx = idx[:m]
                    vals = vals[:m]
                    for k in range(m):
                        if enc == "csr_matrix":
                            r = major_i + 1
                            c = int(idx[k]) + 1
                        else:
                            r = int(idx[k]) + 1
                            c = major_i + 1
                        v = vals[k]
                        if isinstance(v, np.generic):
                            v = v.item()
                        fh.write(f"{r} {c} {v}\n")
        console.print(f"[green]Wrote[/] {out}")


def _export_json(
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
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
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


def _export_image(file: Path, obj: str, out: Path, console: Console) -> None:
    """Export an image-like dataset (H,W) or (H,W,C) to PNG/JPG/TIFF."""
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
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
