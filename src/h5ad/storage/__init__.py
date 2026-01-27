from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import shutil

import h5py

try:
    import zarr
except Exception:  # pragma: no cover - optional dependency
    zarr = None

import numpy as np


@dataclass
class Store:
    backend: str
    root: Any
    path: Path

    def close(self) -> None:
        if self.backend == "hdf5":
            try:
                self.root.close()
            except Exception:
                return

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _require_zarr() -> None:
    if zarr is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "zarr is required for .zarr support. Install with: uv sync --extra zarr"
        )


def is_hdf5_group(obj: Any) -> bool:
    return isinstance(obj, (h5py.File, h5py.Group))


def is_hdf5_dataset(obj: Any) -> bool:
    return isinstance(obj, h5py.Dataset)


def is_zarr_group(obj: Any) -> bool:
    return zarr is not None and isinstance(obj, zarr.Group)


def is_zarr_array(obj: Any) -> bool:
    return zarr is not None and isinstance(obj, zarr.Array)


def is_group(obj: Any) -> bool:
    return is_hdf5_group(obj) or is_zarr_group(obj)


def is_dataset(obj: Any) -> bool:
    return is_hdf5_dataset(obj) or is_zarr_array(obj)


def is_zarr_path(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "zarr.json").exists():
        return True
    if (path / ".zgroup").exists() or (path / ".zattrs").exists():
        return True
    return False


def detect_backend(path: Path) -> str:
    if path.exists():
        if path.is_dir():
            if is_zarr_path(path):
                return "zarr"
            raise ValueError(
                f"Path '{path}' is a directory but does not look like a Zarr store."
            )
        return "hdf5"
    if path.suffix == ".zarr":
        return "zarr"
    return "hdf5"


def open_store(path: Path, mode: str) -> Store:
    path = Path(path)
    backend = detect_backend(path)
    if backend == "zarr":
        _require_zarr()
        root = zarr.open_group(str(path), mode=mode)
        return Store(backend="zarr", root=root, path=path)
    root = h5py.File(path, mode)
    return Store(backend="hdf5", root=root, path=path)


def _normalize_attr_value(value: Any, target_backend: str) -> Any:
    if target_backend == "zarr":
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (list, tuple)):
            return [
                v.decode("utf-8") if isinstance(v, bytes) else v for v in value
            ]
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ("S", "O"):
                return [
                    v.decode("utf-8") if isinstance(v, bytes) else v
                    for v in value.tolist()
                ]
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


def copy_attrs(src_attrs: Any, dst_attrs: Any, *, target_backend: str) -> None:
    for k, v in src_attrs.items():
        dst_attrs[k] = _normalize_attr_value(v, target_backend)


def dataset_create_kwargs(src: Any, *, target_backend: str) -> dict:
    kw: dict = {}
    chunks = getattr(src, "chunks", None)
    if chunks is not None:
        kw["chunks"] = chunks
    if target_backend == "hdf5" and is_hdf5_dataset(src):
        if src.compression is not None:
            kw["compression"] = src.compression
            kw["compression_opts"] = src.compression_opts
        kw["shuffle"] = bool(src.shuffle)
        kw["fletcher32"] = bool(src.fletcher32)
        if src.scaleoffset is not None:
            kw["scaleoffset"] = src.scaleoffset
        if src.fillvalue is not None:
            kw["fillvalue"] = src.fillvalue
    if target_backend == "zarr" and is_zarr_array(src):
        src_zarr_format = getattr(getattr(src, "metadata", None), "zarr_format", None)
        if src_zarr_format == 3:
            compressors = None
            try:
                compressors = getattr(src, "compressors", None)
            except Exception:
                compressors = None
            if compressors is not None:
                kw["compressors"] = compressors
        else:
            try:
                compressor = getattr(src, "compressor", None)
            except Exception:
                compressor = None
            if compressor is not None:
                kw["compressor"] = compressor
        try:
            filters = getattr(src, "filters", None)
        except Exception:
            filters = None
        if filters is not None:
            kw["filters"] = filters
        try:
            fill_value = getattr(src, "fill_value", None)
        except Exception:
            fill_value = None
        if fill_value is not None:
            kw["fill_value"] = fill_value
    return kw


def create_dataset(
    parent: Any,
    name: str,
    *,
    data: Any = None,
    shape: Optional[Sequence[int]] = None,
    dtype: Any = None,
    **kwargs: Any,
) -> Any:
    if is_zarr_group(parent):
        zarr_format = getattr(getattr(parent, "metadata", None), "zarr_format", None)
        if zarr_format == 3:
            kwargs = dict(kwargs)
            kwargs.pop("compressor", None)
        elif zarr_format == 2 and "compressors" in kwargs and "compressor" not in kwargs:
            kwargs = dict(kwargs)
            compressors = kwargs.pop("compressors")
            if isinstance(compressors, (list, tuple)) and len(compressors) == 1:
                kwargs["compressor"] = compressors[0]
        if data is not None:
            return parent.create_array(name, data=data, **kwargs)
        return parent.create_array(name, shape=shape, dtype=dtype, **kwargs)
    if data is not None:
        return parent.create_dataset(name, data=data, **kwargs)
    return parent.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


def _chunk_step(shape: Sequence[int], chunks: Optional[Sequence[int]]) -> int:
    if chunks is not None and len(chunks) > 0 and chunks[0]:
        return int(chunks[0])
    if not shape:
        return 1
    return max(1, min(1024, int(shape[0])))


def copy_dataset(src: Any, dst_group: Any, name: str) -> Any:
    shape = tuple(src.shape) if getattr(src, "shape", None) is not None else ()
    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    ds = create_dataset(
        dst_group,
        name,
        shape=shape,
        dtype=src.dtype,
        **dataset_create_kwargs(src, target_backend=target_backend),
    )
    copy_attrs(src.attrs, ds.attrs, target_backend=target_backend)

    if shape == ():
        ds[()] = src[()]
        return ds

    step = _chunk_step(shape, getattr(src, "chunks", None))
    for start in range(0, shape[0], step):
        end = min(start + step, shape[0])
        if len(shape) == 1:
            ds[start:end] = src[start:end]
        else:
            ds[start:end, ...] = src[start:end, ...]
    return ds


def copy_tree(src_obj: Any, dst_group: Any, name: str, *, exclude: Iterable[str] = ()) -> Any:
    if is_hdf5_group(dst_group) and (is_hdf5_group(src_obj) or is_hdf5_dataset(src_obj)):
        if not exclude:
            dst_group.copy(src_obj, dst_group, name)
            return dst_group[name]
    if is_dataset(src_obj):
        return copy_dataset(src_obj, dst_group, name)
    if not is_group(src_obj):
        raise TypeError(f"Unsupported object type for copy: {type(src_obj)}")

    target_backend = "zarr" if is_zarr_group(dst_group) else "hdf5"
    grp = dst_group.create_group(name)
    copy_attrs(src_obj.attrs, grp.attrs, target_backend=target_backend)
    for key in src_obj.keys():
        if key in exclude:
            continue
        child = src_obj[key]
        copy_tree(child, grp, key, exclude=exclude)
    return grp


def copy_store_contents(src_root: Any, dst_root: Any) -> None:
    for key in src_root.keys():
        copy_tree(src_root[key], dst_root, key)


def copy_path(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if is_zarr_path(src):
        if dst.exists():
            raise FileExistsError(f"Destination '{dst}' already exists.")
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)
