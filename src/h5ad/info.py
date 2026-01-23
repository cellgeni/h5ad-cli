from typing import Optional, Tuple, Dict, Any, Union
import h5py
import numpy as np


def get_entry_type(entry: Union[h5py.Group, h5py.Dataset]) -> Dict[str, Any]:
    """
    Determine the type/format of an HDF5 object for export guidance.

    Returns a dict with:
        - type: str (e.g., 'dataframe', 'sparse-matrix', 'dense-matrix', 'dict', 'image', 'array', 'scalar')
        - export_as: str (suggested export format: csv, mtx, npy, json, image)
        - encoding: str (h5ad encoding-type if present)
        - shape: tuple or None
        - dtype: str or None
        - details: str (human-readable description)
    """
    result: Dict[str, Any] = {
        "type": "unknown",
        "export_as": None,
        "encoding": None,
        "shape": None,
        "dtype": None,
        "details": "",
    }

    # Get encoding-type attribute if present
    enc = entry.attrs.get("encoding-type", b"")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    result["encoding"] = enc if enc else None

    # Infer the type for Dataset entry
    if isinstance(entry, h5py.Dataset):
        result["shape"] = entry.shape
        result["dtype"] = str(entry.dtype)

        # Scalar
        if entry.shape == ():
            result["type"] = "scalar"
            result["export_as"] = "json"
            result["details"] = f"Scalar value ({entry.dtype})"
            return result

        # 1D or 2D numeric array -> dense matrix / array
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

    # It's a Group
    if isinstance(entry, h5py.Group):
        # Check for sparse matrix (CSR/CSC)
        if enc in ("csr_matrix", "csc_matrix"):
            shape = entry.attrs.get("shape", None)
            shape_str = f"{shape[0]}×{shape[1]}" if shape is not None else "?"
            result["type"] = "sparse-matrix"
            result["export_as"] = "mtx"
            result["details"] = (
                f"Sparse {enc.replace('_matrix', '').upper()} matrix {shape_str}"
            )
            return result

        # Check for categorical
        if enc == "categorical":
            codes = entry.get("codes")
            cats = entry.get("categories")
            n_codes = codes.shape[0] if codes is not None else "?"
            n_cats = cats.shape[0] if cats is not None else "?"
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["details"] = f"Categorical [{n_codes} values, {n_cats} categories]"
            return result

        # Check for dataframe (obs/var style with _index)
        if "_index" in entry.attrs or "obs_names" in entry or "var_names" in entry:
            n_cols = len([k for k in entry.keys() if k != "_index"])
            result["type"] = "dataframe"
            result["export_as"] = "csv"
            result["details"] = f"DataFrame with {n_cols} columns"
            return result

        # Check for array-like groups (nullable integer, string array, etc.)
        if enc in ("nullable-integer", "string-array"):
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"Encoded array ({enc})"
            return result

        # Generic dict/group
        n_keys = len(list(entry.keys()))
        result["type"] = "dict"
        result["export_as"] = "json"
        result["details"] = f"Group with {n_keys} keys"
        return result

    return result


def format_type_info(info: Dict[str, Any]) -> str:
    """Format type info as a colored string for display."""
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


def axis_len(file: h5py.File, axis: str) -> int:
    """
    Get the length of the specified axis ('obs' or 'var') in the h5ad file.

    Args:
        file (h5py.File): Opened h5ad file object
        axis (str): Axis name ('obs' or 'var')

    Returns:
        int: Length of the axis

    Raises:
        ValueError: If axis is not 'obs' or 'var'
        KeyError: If axis or index dataset not found in file
        TypeError: If axis is not a group or index is not a dataset
        RuntimeError: If axis length cannot be determined
    """
    # Check if the specified axis exists in the file
    if axis not in file:
        raise KeyError(f"'{axis}' not found in the file.")

    # Get the group corresponding to the axis
    group = file[axis]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"'{axis}' is not a group.")

    # Determine the index name for the axis
    index_name = group.attrs.get("_index", None)
    if index_name is None:
        if axis == "obs":
            index_name = "obs_names"
        elif axis == "var":
            index_name = "var_names"
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'obs' or 'var'.")

    # Decode bytes to string if necessary
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")

    # Check if the index dataset exists
    if index_name not in group:
        raise KeyError(f"Index dataset '{index_name}' not found in '{axis}' group.")

    # Return the length of the index dataset
    dataset = group[index_name]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Index '{index_name}' in '{axis}' is not a dataset.")
    if dataset.shape:
        return int(dataset.shape[0])
    raise RuntimeError(
        f"Cannot determine length of '{axis}': index dataset has no shape."
    )


def get_axis_group(file: h5py.File, axis: str) -> Tuple[h5py.Group, int, str]:
    """
    Get the axis group, its length, and index name.

    Args:
        file (h5py.File): Opened h5ad file object
        axis (str): Axis name ('obs' or 'var')

    Returns:
        Tuple[h5py.Group, int, str]: Axis group, its length, and index name

    Raises:
        ValueError: If axis is not 'obs' or 'var'
        KeyError: If axis or index dataset not found in file
        TypeError: If axis is not a group or index is not a dataset
        RuntimeError: If axis length cannot be determined
    """
    if axis not in ("obs", "var"):
        raise ValueError("axis must be 'obs' or 'var'.")

    # axis_len will validate existence and get length (raises exceptions if issues)
    n = axis_len(file, axis)

    # Get the group (already validated by axis_len)
    group = file[axis]

    # Get the index name
    index_name = group.attrs.get("_index", None)
    if index_name is None:
        index_name = "obs_names" if axis == "obs" else "var_names"
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")

    return group, n, index_name
