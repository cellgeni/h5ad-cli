from __future__ import annotations

from typing import Any, Dict

import numpy as np

from h5ad.storage import is_dataset, is_group
from h5ad.util.path import norm_path


TYPE_EXTENSIONS = {
    "dataframe": {".csv"},
    "sparse-matrix": {".mtx"},
    "dense-matrix": {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "array": {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "dict": {".json"},
    "scalar": {".json"},
    "categorical": {".csv"},
    "awkward-array": {".json"},
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

EXPORTABLE_TYPES = set(TYPE_EXTENSIONS.keys())


def _get_encoding_type(group: Any) -> str:
    enc = group.attrs.get("encoding-type", "")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    return str(enc)


def _resolve(root: Any, obj: str) -> Any:
    obj = norm_path(obj)
    if obj not in root:
        raise KeyError(f"'{obj}' not found in the file.")
    return root[obj]


def _check_json_exportable(h5obj: Any, max_elements: int, path: str = "") -> None:
    if is_dataset(h5obj):
        if h5obj.shape == ():
            return
        n = int(np.prod(h5obj.shape)) if h5obj.shape else 0
        if n > max_elements:
            obj_name = getattr(h5obj, "name", "<object>")
            raise ValueError(
                f"Cannot export to JSON: '{path or obj_name}' has {n} elements "
                f"(max {max_elements}). Use --max-elements to increase limit."
            )
        return

    if is_group(h5obj):
        enc = _get_encoding_type(h5obj)
        if enc in ("csr_matrix", "csc_matrix"):
            obj_name = getattr(h5obj, "name", "<object>")
            raise ValueError(
                f"Cannot export to JSON: '{path or obj_name}' is a sparse matrix. "
                "Export it as .mtx instead."
            )

        for key in h5obj.keys():
            child = h5obj[key]
            child_path = f"{path}/{key}" if path else key
            if is_group(child) or is_dataset(child):
                _check_json_exportable(child, max_elements=max_elements, path=child_path)
