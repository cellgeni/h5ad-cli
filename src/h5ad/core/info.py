from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union

import numpy as np

from h5ad.storage import is_dataset, is_group, is_hdf5_dataset


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def get_entry_type(entry: Any) -> Dict[str, Any]:
    """
    Determine the type/format of an object for export guidance.

    Supports both:
    - v0.2.0 (modern): Objects with encoding-type/encoding-version attributes
    - v0.1.0 (legacy): Objects without encoding attributes, inferred from structure
    """
    result: Dict[str, Any] = {
        "type": "unknown",
        "export_as": None,
        "encoding": None,
        "shape": None,
        "dtype": None,
        "details": "",
        "version": None,
    }

    enc = _decode_attr(entry.attrs.get("encoding-type", b""))
    result["encoding"] = enc if enc else None

    enc_ver = _decode_attr(entry.attrs.get("encoding-version", b""))
    result["version"] = enc_ver if enc_ver else None

    if is_dataset(entry):
        result["shape"] = entry.shape
        result["dtype"] = str(entry.dtype)

        if "categories" in entry.attrs:
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["version"] = result["version"] or "0.1.0"
            n_cats = "?"
            if is_hdf5_dataset(entry):
                try:
                    cats_ref = entry.attrs["categories"]
                    cats_ds = entry.file[cats_ref]
                    n_cats = cats_ds.shape[0]
                except Exception:
                    n_cats = "?"
            result["details"] = (
                f"Legacy categorical [{entry.shape[0]} values, {n_cats} categories]"
            )
            return result

        if entry.shape == ():
            result["type"] = "scalar"
            result["export_as"] = "json"
            result["details"] = f"Scalar value ({entry.dtype})"
            return result

        if entry.ndim == 1:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"1D array [{entry.shape[0]}] ({entry.dtype})"
        elif entry.ndim == 2:
            result["type"] = "dense-matrix"
            result["export_as"] = "npy"
            result["details"] = (
                f"Dense matrix {entry.shape[0]}×{entry.shape[1]} ({entry.dtype})"
            )
        elif entry.ndim == 3:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"3D array {entry.shape} ({entry.dtype})"
        else:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"ND array {entry.shape} ({entry.dtype})"
        return result

    if is_group(entry):
        if enc in ("csr_matrix", "csc_matrix"):
            shape = entry.attrs.get("shape", None)
            shape_str = f"{shape[0]}×{shape[1]}" if shape is not None else "?"
            result["type"] = "sparse-matrix"
            result["export_as"] = "mtx"
            result["details"] = (
                f"Sparse {enc.replace('_matrix', '').upper()} matrix {shape_str}"
            )
            return result

        if enc == "categorical":
            codes = entry.get("codes")
            cats = entry.get("categories")
            n_codes = codes.shape[0] if codes is not None else "?"
            n_cats = cats.shape[0] if cats is not None else "?"
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["details"] = f"Categorical [{n_codes} values, {n_cats} categories]"
            return result

        if (
            enc == "dataframe"
            or "_index" in entry.attrs
            or "obs_names" in entry
            or "var_names" in entry
        ):
            if enc == "dataframe":
                df_version = result["version"] or "0.2.0"
            else:
                df_version = "0.1.0"
            result["version"] = df_version

            has_legacy_cats = "__categories" in entry
            n_cols = len(
                [k for k in entry.keys() if k not in ("_index", "__categories")]
            )

            result["type"] = "dataframe"
            result["export_as"] = "csv"
            if has_legacy_cats:
                result["details"] = f"DataFrame with {n_cols} columns (legacy v0.1.0)"
            else:
                result["details"] = f"DataFrame with {n_cols} columns"
            return result

        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"Encoded array ({enc})"
            return result

        if enc == "string-array":
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = "Encoded string array"
            return result

        if enc == "awkward-array":
            length = entry.attrs.get("length", "?")
            result["type"] = "awkward-array"
            result["export_as"] = "json"
            result["details"] = f"Awkward array (length={length})"
            return result

        n_keys = len(list(entry.keys()))
        result["type"] = "dict"
        result["export_as"] = "json"
        result["details"] = f"Group with {n_keys} keys"
        return result

    return result


def format_type_info(info: Dict[str, Any]) -> str:
    type_colors = {
        "dataframe": "green",
        "sparse-matrix": "magenta",
        "dense-matrix": "blue",
        "array": "blue",
        "dict": "yellow",
        "categorical": "green",
        "scalar": "white",
        "unknown": "red",
    }

    color = type_colors.get(info["type"], "white")
    return f"[{color}]<{info['type']}>[/]"


def axis_len(file: Any, axis: str) -> int:
    if axis not in file:
        raise KeyError(f"'{axis}' not found in the file.")

    group = file[axis]
    if not is_group(group):
        raise TypeError(f"'{axis}' is not a group.")

    index_name = group.attrs.get("_index", None)
    if index_name is None:
        if axis == "obs":
            index_name = "obs_names"
        elif axis == "var":
            index_name = "var_names"
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'obs' or 'var'.")

    index_name = _decode_attr(index_name)

    if index_name not in group:
        raise KeyError(f"Index dataset '{index_name}' not found in '{axis}' group.")

    dataset = group[index_name]
    if not is_dataset(dataset):
        raise TypeError(f"Index '{index_name}' in '{axis}' is not a dataset.")
    if dataset.shape:
        return int(dataset.shape[0])
    raise ValueError(
        f"Cannot determine length of '{axis}': index dataset has no shape."
    )


def get_axis_group(file: Any, axis: str) -> Tuple[Any, int, str]:
    if axis not in ("obs", "var"):
        raise ValueError("axis must be 'obs' or 'var'.")

    n = axis_len(file, axis)
    group = file[axis]

    index_name = group.attrs.get("_index", None)
    if index_name is None:
        index_name = "obs_names" if axis == "obs" else "var_names"
    index_name = _decode_attr(index_name)

    return group, n, index_name
