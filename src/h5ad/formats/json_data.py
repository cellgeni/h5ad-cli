from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from rich.console import Console

from h5ad.core.read import decode_str_array
from h5ad.formats.common import _check_json_exportable, _resolve
from h5ad.storage import create_dataset, is_dataset, is_group
from h5ad.util.path import norm_path


def export_json(
    root: Any,
    obj: str,
    out: Path | None,
    max_elements: int,
    include_attrs: bool,
    console: Console,
) -> None:
    h5obj = _resolve(root, obj)
    _check_json_exportable(h5obj, max_elements=max_elements)

    payload = _to_jsonable(
        h5obj, max_elements=max_elements, include_attrs=include_attrs
    )
    if out is None or str(out) == "-":
        out_fh = sys.stdout
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out, "w", encoding="utf-8")
    try:
        json.dump(payload, out_fh, indent=2, ensure_ascii=False, sort_keys=True)
        out_fh.write("\n")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()
    if out_fh is not sys.stdout:
        console.print(f"[green]Wrote[/] {out}")


def _attrs_to_jsonable(attrs: Any, max_elements: int) -> Dict[str, Any]:
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


def _dataset_to_jsonable(ds: Any, max_elements: int) -> Any:
    if ds.shape == ():
        v = ds[()]
        return _pyify(v, max_elements=max_elements)
    n = int(np.prod(ds.shape)) if ds.shape else 0
    if n > max_elements:
        ds_name = getattr(ds, "name", "<dataset>")
        raise ValueError(
            f"Refusing to convert dataset {ds_name!r} with {n} elements (> {max_elements}) to JSON."
        )
    arr = np.asarray(ds[...])
    return _pyify(arr, max_elements=max_elements)


def _to_jsonable(h5obj: Any, max_elements: int, include_attrs: bool) -> Any:
    if is_dataset(h5obj):
        return _dataset_to_jsonable(h5obj, max_elements=max_elements)

    d: Dict[str, Any] = {}
    if include_attrs and len(h5obj.attrs):
        d["__attrs__"] = _attrs_to_jsonable(h5obj.attrs, max_elements=max_elements)

    for key in h5obj.keys():
        child = h5obj[key]
        if is_group(child) or is_dataset(child):
            d[str(key)] = _to_jsonable(
                child,
                max_elements=max_elements,
                include_attrs=include_attrs,
            )
    return d


def import_json(
    root: Any,
    obj: str,
    input_file: Path,
    console: Console,
) -> None:
    obj = norm_path(obj)
    with open(input_file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    parts = obj.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent[part] if part in parent else parent.create_group(part)
    name = parts[-1]

    if name in parent:
        del parent[name]

    _write_json_to_group(parent, name, payload)

    console.print(f"[green]Imported[/] JSON data into '{obj}'")


def _write_json_to_group(parent: Any, name: str, value: Any) -> None:
    if isinstance(value, dict):
        group = parent.create_group(name)
        for k, v in value.items():
            _write_json_to_group(group, k, v)
    elif isinstance(value, list):
        try:
            arr = np.array(value)
            if arr.dtype.kind in ("U", "O"):
                arr = np.array(value, dtype="S")
            create_dataset(parent, name, data=arr)
        except (ValueError, TypeError):
            create_dataset(parent, name, data=json.dumps(value).encode("utf-8"))
    elif isinstance(value, str):
        create_dataset(parent, name, data=np.array([value], dtype="S"))
    elif isinstance(value, bool):
        create_dataset(parent, name, data=np.array(value, dtype=bool))
    elif isinstance(value, int):
        create_dataset(parent, name, data=np.array(value, dtype=np.int64))
    elif isinstance(value, float):
        create_dataset(parent, name, data=np.array(value, dtype=np.float64))
    elif value is None:
        ds = create_dataset(parent, name, data=np.array([], dtype="S"))
        ds.attrs["_is_none"] = True
    else:
        raise ValueError(f"Cannot convert JSON value of type {type(value).__name__}")
