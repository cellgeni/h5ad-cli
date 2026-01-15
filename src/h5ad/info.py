from typing import Optional, Tuple, Dict, Any, Union
import h5py
import numpy as np


def get_entry_type(obj: Union[h5py.Group, h5py.Dataset]) -> Dict[str, Any]:
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
    enc = obj.attrs.get("encoding-type", b"")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    result["encoding"] = enc if enc else None

    if isinstance(obj, h5py.Dataset):
        result["shape"] = obj.shape
        result["dtype"] = str(obj.dtype)

        # Scalar
        if obj.shape == ():
            result["type"] = "scalar"
            result["export_as"] = "json"
            result["details"] = f"Scalar value ({obj.dtype})"
            return result

        # 1D or 2D numeric array -> dense matrix / array
        if obj.ndim == 1:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"1D array [{obj.shape[0]}] ({obj.dtype})"
        elif obj.ndim == 2:
            # Check if it looks like an image (2D with reasonable image dimensions)
            # Minimum 16x16, maximum 10000x10000, numeric dtype
            if (
                obj.shape[0] >= 16
                and obj.shape[1] >= 16
                and obj.shape[0] <= 10000
                and obj.shape[1] <= 10000
                and (np.issubdtype(obj.dtype, np.number) or obj.dtype == np.bool_)
            ):
                # Could be an image, but default to dense-matrix
                # Image export can still be used if user provides image extension
                pass
            result["type"] = "dense-matrix"
            result["export_as"] = "npy"
            result["details"] = (
                f"Dense matrix {obj.shape[0]}×{obj.shape[1]} ({obj.dtype})"
            )
        elif obj.ndim == 3:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"3D array {obj.shape} ({obj.dtype})"
        else:
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"ND array {obj.shape} ({obj.dtype})"

        return result

    # It's a Group
    if isinstance(obj, h5py.Group):
        # Check for sparse matrix (CSR/CSC)
        if enc in ("csr_matrix", "csc_matrix"):
            shape = obj.attrs.get("shape", None)
            shape_str = f"{shape[0]}×{shape[1]}" if shape is not None else "?"
            result["type"] = "sparse-matrix"
            result["export_as"] = "mtx"
            result["details"] = (
                f"Sparse {enc.replace('_matrix', '').upper()} matrix {shape_str}"
            )
            return result

        # Check for categorical
        if enc == "categorical":
            codes = obj.get("codes")
            cats = obj.get("categories")
            n_codes = codes.shape[0] if codes is not None else "?"
            n_cats = cats.shape[0] if cats is not None else "?"
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["details"] = f"Categorical [{n_codes} values, {n_cats} categories]"
            return result

        # Check for dataframe (obs/var style with _index)
        if "_index" in obj.attrs or "obs_names" in obj or "var_names" in obj:
            n_cols = len([k for k in obj.keys() if k != "_index"])
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
        n_keys = len(list(obj.keys()))
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


def axis_len(file: h5py.File, axis: str) -> Optional[int]:
    """
    Get the length of the specified axis ('obs' or 'var') in the h5ad file.
    Args:
        file (h5py.File): Opened h5ad file object
        axis (str): Axis name ('obs' or 'var')

    Returns:
        Optional[int]: Length of the axis, or None if not found
    """
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


def get_axis_group(file: h5py.File, axis: str) -> Tuple[h5py.Group, int, str]:
    """
    Get the axis group, its length, and index name.
    Args:
        file (h5py.File): Opened h5ad file object
        axis (str): Axis name ('obs' or 'var')

    Returns:
        Tuple[h5py.Group, int, str]: Axis group, its length, and index
    """
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
