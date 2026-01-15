from typing import Optional, Tuple
import h5py
import numpy as np


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
