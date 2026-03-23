from __future__ import annotations

from typing import List, Dict, Any

import h5py
import numpy as np

from h5ad.storage import is_group, is_dataset, is_hdf5_dataset


def decode_str_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)

    if np.issubdtype(arr.dtype, np.bytes_):
        flat = arr.reshape(-1)
        decoded = [
            b.decode("utf-8", errors="replace") if isinstance(b, (bytes, np.bytes_)) else str(b)
            for b in flat
        ]
        return np.asarray(decoded, dtype=str).reshape(arr.shape)

    if arr.dtype.kind == "O":
        flat = arr.reshape(-1)
        decoded = [
            v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in flat
        ]
        return np.asarray(decoded, dtype=str).reshape(arr.shape)

    return arr.astype(str)


def _categorical_cache_key(col: Any, parent_group: Any | None = None) -> str:
    col_name = getattr(col, "name", None)
    if isinstance(col_name, str) and col_name:
        return col_name

    if parent_group is not None:
        parent_name = getattr(parent_group, "name", "")
        rel_name = getattr(col, "path", "")
        if parent_name or rel_name:
            return f"{parent_name}/{rel_name}"

    return repr(col)


def read_categorical_column(
    col: Any,
    start: int,
    end: int,
    cache: Dict[str, np.ndarray],
    parent_group: Any | None = None,
) -> List[str]:
    key = _categorical_cache_key(col, parent_group)

    if is_group(col):
        if key not in cache:
            cats = col["categories"][...]
            cats = decode_str_array(cats)
            cache[key] = np.asarray(cats, dtype=str)
        cats = cache[key]

        codes_ds = col["codes"]
        codes = codes_ds[start:end]
        codes = np.asarray(codes, dtype=np.int64)
        return [cats[c] if 0 <= c < len(cats) else "" for c in codes]

    if is_dataset(col):
        if key not in cache:
            cats_ref = col.attrs.get("categories", None)
            if cats_ref is not None and is_hdf5_dataset(col):
                cats_ds = col.file[cats_ref]
                cats = cats_ds[...]
            elif parent_group is not None and "__categories" in parent_group:
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
    group: Any,
    col_name: str,
    start: int,
    end: int,
    cat_cache: Dict[str, np.ndarray],
) -> List[str]:
    if col_name not in group:
        raise RuntimeError(f"Column {col_name!r} not found in group {group.name}")

    col = group[col_name]

    if is_dataset(col):
        if "categories" in col.attrs:
            return read_categorical_column(col, start, end, cat_cache, group)

        chunk = col[start:end]
        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)
        chunk = decode_str_array(np.asarray(chunk))
        return chunk.tolist()

    if is_group(col):
        enc = col.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")

        if enc == "categorical":
            return read_categorical_column(col, start, end, cat_cache)

        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            values = col["values"][start:end]
            mask = col["mask"][start:end]
            values = decode_str_array(np.asarray(values))
            return ["" if m else str(v) for v, m in zip(values, mask)]

        raise ValueError(
            f"Unsupported group encoding {enc!r} for column {col_name!r}"
        )

    raise TypeError(
        f"Unsupported column type for {col_name!r} in group {group.name}"
    )
