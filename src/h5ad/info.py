from typing import Optional, Tuple, Dict, Any, Union
import h5py
import numpy as np


def get_entry_type(entry: Union[h5py.Group, h5py.Dataset]) -> Dict[str, Any]:
    """
    Determine the type/format of an HDF5 object for export guidance.

    Supports both:
    - v0.2.0 (modern): Objects with encoding-type/encoding-version attributes
    - v0.1.0 (legacy): Objects without encoding attributes, inferred from structure

    Returns a dict with:
        - type: str (e.g., 'dataframe', 'sparse-matrix', 'dense-matrix', 'dict', 'image', 'array', 'scalar')
        - export_as: str (suggested export format: csv, mtx, npy, json, image)
        - encoding: str (h5ad encoding-type if present)
        - shape: tuple or None
        - dtype: str or None
        - details: str (human-readable description)
        - version: str ('0.2.0', '0.1.0', or None for unknown)
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

    # Get encoding-type attribute if present
    enc = entry.attrs.get("encoding-type", b"")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    result["encoding"] = enc if enc else None

    # Get encoding-version if present
    enc_ver = entry.attrs.get("encoding-version", b"")
    if isinstance(enc_ver, bytes):
        enc_ver = enc_ver.decode("utf-8")
    result["version"] = enc_ver if enc_ver else None

    # Infer the type for Dataset entry
    if isinstance(entry, h5py.Dataset):
        result["shape"] = entry.shape
        result["dtype"] = str(entry.dtype)

        # Check for legacy categorical (v0.1.0): dataset with 'categories' attribute
        if "categories" in entry.attrs:
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["version"] = result["version"] or "0.1.0"
            # Try to get category count from referenced dataset
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
        # Check for sparse matrix (CSR/CSC) - same in both versions
        if enc in ("csr_matrix", "csc_matrix"):
            shape = entry.attrs.get("shape", None)
            shape_str = f"{shape[0]}×{shape[1]}" if shape is not None else "?"
            result["type"] = "sparse-matrix"
            result["export_as"] = "mtx"
            result["details"] = (
                f"Sparse {enc.replace('_matrix', '').upper()} matrix {shape_str}"
            )
            return result

        # Check for v0.2.0 categorical (Group with codes/categories)
        if enc == "categorical":
            codes = entry.get("codes")
            cats = entry.get("categories")
            n_codes = codes.shape[0] if codes is not None else "?"
            n_cats = cats.shape[0] if cats is not None else "?"
            result["type"] = "categorical"
            result["export_as"] = "csv"
            result["details"] = f"Categorical [{n_codes} values, {n_cats} categories]"
            return result

        # Check for dataframe (obs/var style)
        # v0.2.0: has encoding-type="dataframe"
        # v0.1.0: has _index attribute or obs_names/var_names dataset
        if (
            enc == "dataframe"
            or "_index" in entry.attrs
            or "obs_names" in entry
            or "var_names" in entry
        ):
            # Detect version
            if enc == "dataframe":
                df_version = result["version"] or "0.2.0"
            else:
                df_version = "0.1.0"  # No encoding-type, legacy format
            result["version"] = df_version

            # Check for __categories subgroup (v0.1.0 legacy)
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

        # Check for nullable arrays (v0.2.0)
        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = f"Encoded array ({enc})"
            return result

        # Check for string-array encoding
        if enc == "string-array":
            result["type"] = "array"
            result["export_as"] = "npy"
            result["details"] = "Encoded string array"
            return result

        # Check for awkward-array (experimental)
        if enc == "awkward-array":
            length = entry.attrs.get("length", "?")
            result["type"] = "awkward-array"
            result["export_as"] = "json"
            result["details"] = f"Awkward array (length={length})"
            return result

        # Generic dict/group (v0.2.0 has encoding-type="dict", v0.1.0 has no attributes)
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
