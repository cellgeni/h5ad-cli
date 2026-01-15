import numpy as np
import h5py
from typing import List, Dict

def _decode_str_array(array: np.ndarray) -> np.ndarray:
    """
    Decode a numpy array of bytes or objects to strings.
    Args:
        array (np.ndarray): Input numpy array of bytes or objects

    Returns:
        np.ndarray: Decoded numpy array of strings
    """
    if np.issubdtype(array.dtype, np.bytes_):
        return array.astype("U")
    if array.dtype.kind == "O":
        return array.astype(str)
    return array.astype(str)


def read_categorical_column(
    col_group: h5py.Group, start: int, end: int, cache: Dict[int, np.ndarray]
) -> List[str]:
    """
    Decode an AnnData 'categorical' column for a slice [start:end].
    Args:
        col_group (h5py.Group): Column group containing 'categories' and 'codes'
        start (int): Start index of the slice
        end (int): End index of the slice
        cache (Dict[int, np.ndarray]): Cache for decoded categories
    Returns:
        List[str]: Decoded categorical values for the specified slice
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


def col_chunk_as_strings(
    group: h5py.Group,
    col_name: str,
    start: int,
    end: int,
    cat_cache: Dict[int, np.ndarray],
) -> List[str]:
    """
    Read a column from an obs/var group as strings.
    Args:
        group (h5py.Group): The obs/var group
        col_name (str): Name of the column to read
        start (int): Start index of the slice
        end (int): End index of the slice
        cat_cache (Dict[int, np.ndarray]): Cache for decoded categorical columns
    Returns:
        List[str]: Column values as strings for the specified slice
    """
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
            return read_categorical_column(col_group, start, end, cat_cache)

    raise RuntimeError(f"Unsupported column {col_name!r} in group {group.name}")