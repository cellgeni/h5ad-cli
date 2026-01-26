import numpy as np
import h5py
from typing import List, Dict


def decode_str_array(array: np.ndarray) -> np.ndarray:
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
    col: h5py.Group | h5py.Dataset,
    start: int,
    end: int,
    cache: Dict[int, np.ndarray],
    parent_group: h5py.Group | None = None,
) -> List[str]:
    """
    Decode an AnnData 'categorical' column for a slice [start:end].

    Supports both:
    - v0.2.0 (modern): Group with 'codes' and 'categories' datasets
    - v0.1.0 (legacy): Dataset with 'categories' attribute referencing __categories/<colname>

    Args:
        col: Column group (v0.2.0) or dataset (v0.1.0)
        start: Start index of the slice
        end: End index of the slice
        cache: Cache for decoded categories
        parent_group: Parent obs/var group (needed for v0.1.0 to resolve __categories)

    Returns:
        List[str]: Decoded categorical values for the specified slice
    """
    key = id(col)

    # v0.2.0 format: Group with 'codes' and 'categories' datasets
    if isinstance(col, h5py.Group):
        if key not in cache:
            cats = col["categories"][...]
            cats = decode_str_array(cats)
            cache[key] = np.asarray(cats, dtype=str)
        cats = cache[key]

        codes_ds = col["codes"]
        codes = codes_ds[start:end]
        codes = np.asarray(codes, dtype=np.int64)
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]

    # v0.1.0 format: Dataset with 'categories' attribute (object reference)
    if isinstance(col, h5py.Dataset):
        if key not in cache:
            cats_ref = col.attrs.get("categories", None)
            if cats_ref is not None:
                # Dereference the HDF5 object reference
                cats_ds = col.file[cats_ref]
                cats = cats_ds[...]
            elif parent_group is not None and "__categories" in parent_group:
                # Fallback: look for __categories subgroup
                col_name = col.name.split("/")[-1]
                cats_grp = parent_group["__categories"]
                if col_name in cats_grp:
                    cats = cats_grp[col_name][...]
                else:
                    raise KeyError(
                        f"Cannot find categories for legacy column {col.name}"
                    )
            else:
                raise KeyError(
                    f"Cannot find categories for legacy column {col.name}"
                )
            cats = decode_str_array(cats)
            cache[key] = np.asarray(cats, dtype=str)
        cats = cache[key]

        codes = col[start:end]
        codes = np.asarray(codes, dtype=np.int64)
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]

    raise TypeError(f"Unsupported categorical column type: {type(col)}")


def col_chunk_as_strings(
    group: h5py.Group,
    col_name: str,
    start: int,
    end: int,
    cat_cache: Dict[int, np.ndarray],
) -> List[str]:
    """
    Read a column from an obs/var group as strings.

    Supports both:
    - v0.2.0 (modern): Columns with encoding-type attribute
    - v0.1.0 (legacy): Categorical columns with 'categories' attribute referencing __categories

    Args:
        group (h5py.Group): The obs/var group
        col_name (str): Name of the column to read
        start (int): Start index of the slice
        end (int): End index of the slice
        cat_cache (Dict[int, np.ndarray]): Cache for decoded categorical columns

    Returns:
        List[str]: Column values as strings for the specified slice
    """
    if col_name not in group:
        raise KeyError(f"Column {col_name!r} not found in group {group.name}")

    col = group[col_name]

    # Case 1: Dataset (could be plain array or legacy categorical)
    if isinstance(col, h5py.Dataset):
        # Check for v0.1.0 legacy categorical (has 'categories' attribute)
        if "categories" in col.attrs:
            return read_categorical_column(col, start, end, cat_cache, group)

        # Plain dataset (numeric, string, etc.)
        chunk = col[start:end]
        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)
        chunk = decode_str_array(np.asarray(chunk))
        return chunk.tolist()

    # Case 2: Group (v0.2.0 encoded types like categorical, nullable, etc.)
    if isinstance(col, h5py.Group):
        enc = col.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")

        if enc == "categorical":
            return read_categorical_column(col, start, end, cat_cache)

        # Handle nullable arrays (nullable-integer, nullable-boolean, nullable-string-array)
        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            values = col["values"][start:end]
            mask = col["mask"][start:end]
            values = decode_str_array(np.asarray(values))
            # Apply mask: masked values become empty string
            return ["" if m else str(v) for v, m in zip(values, mask)]

        raise ValueError(
            f"Unsupported group encoding {enc!r} for column {col_name!r}"
        )

    raise TypeError(
        f"Unsupported column type for {col_name!r} in group {group.name}"
    )
