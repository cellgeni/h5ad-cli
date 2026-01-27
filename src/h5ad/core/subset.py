"""Subset operations for .h5ad and .zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict, Any

import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from h5ad.core.read import decode_str_array
from h5ad.storage import (
    create_dataset,
    copy_attrs,
    copy_tree,
    dataset_create_kwargs,
    is_dataset,
    is_group,
    is_zarr_group,
    is_zarr_array,
    open_store,
)


def _target_backend(dst_group: Any) -> str:
    return "zarr" if is_zarr_group(dst_group) else "hdf5"


def _ensure_group(parent: Any, name: str) -> Any:
    return parent[name] if name in parent else parent.create_group(name)


def _group_get(parent: Any, key: str) -> Any | None:
    return parent[key] if key in parent else None


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_name_file(path: Path) -> Set[str]:
    names: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    return names


def indices_from_name_set(
    names_ds: Any,
    keep: Set[str],
    *,
    chunk_size: int = 200_000,
) -> Tuple[np.ndarray, Set[str]]:
    if names_ds.ndim != 1:
        flat_len = int(np.prod(names_ds.shape))
    else:
        flat_len = names_ds.shape[0]

    remaining = set(keep)
    found_indices: List[int] = []

    for start in range(0, flat_len, chunk_size):
        end = min(start + chunk_size, flat_len)
        chunk = names_ds[start:end]
        chunk = decode_str_array(np.asarray(chunk)).astype(str)

        for i, name in enumerate(chunk):
            if name in remaining:
                found_indices.append(start + i)
                remaining.remove(name)

        if not remaining:
            break

    return np.asarray(found_indices, dtype=np.int64), remaining


def subset_axis_group(
    src: Any,
    dst: Any,
    indices: Optional[np.ndarray],
) -> None:
    copy_attrs(src.attrs, dst.attrs, target_backend=_target_backend(dst))
    target_backend = _target_backend(dst)

    for key in src.keys():
        obj = src[key]

        if is_dataset(obj):
            if indices is None:
                copy_tree(obj, dst, key)
            else:
                if is_zarr_array(obj):
                    if obj.ndim == 1:
                        data = obj.oindex[indices]
                    else:
                        selection = (indices,) + (slice(None),) * (obj.ndim - 1)
                        data = obj.oindex[selection]
                else:
                    data = obj[indices, ...]
                ds = create_dataset(
                    dst,
                    key,
                    data=data,
                    **dataset_create_kwargs(obj, target_backend=target_backend),
                )
                copy_attrs(obj.attrs, ds.attrs, target_backend=target_backend)
        elif is_group(obj):
            enc = obj.attrs.get("encoding-type", b"")
            if isinstance(enc, bytes):
                enc = enc.decode("utf-8")

            if enc == "categorical":
                gdst = dst.create_group(key)
                copy_attrs(obj.attrs, gdst.attrs, target_backend=target_backend)
                copy_tree(obj["categories"], gdst, "categories")

                codes = obj["codes"]
                if indices is None:
                    copy_tree(codes, gdst, "codes")
                else:
                    codes_sub = codes[indices, ...]
                    ds = create_dataset(
                        gdst,
                        "codes",
                        data=codes_sub,
                        **dataset_create_kwargs(codes, target_backend=target_backend),
                    )
                    copy_attrs(codes.attrs, ds.attrs, target_backend=target_backend)
            else:
                copy_tree(obj, dst, key)


def subset_dense_matrix(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int = 1024,
) -> None:
    if src.ndim != 2:
        copy_tree(src, dst_parent, name)
        return

    n_obs, n_var = src.shape
    out_obs = len(obs_idx) if obs_idx is not None else n_obs
    out_var = len(var_idx) if var_idx is not None else n_var

    target_backend = _target_backend(dst_parent)
    kw = dataset_create_kwargs(src, target_backend=target_backend)
    chunks = kw.get("chunks")
    if isinstance(chunks, (tuple, list)) and len(chunks) >= 2:
        kw["chunks"] = (min(int(chunks[0]), out_obs), min(int(chunks[1]), out_var))

    dst = create_dataset(
        dst_parent,
        name,
        shape=(out_obs, out_var),
        dtype=src.dtype,
        **kw,
    )
    copy_attrs(src.attrs, dst.attrs, target_backend=_target_backend(dst_parent))

    for out_start in range(0, out_obs, chunk_rows):
        out_end = min(out_start + chunk_rows, out_obs)

        if obs_idx is None:
            block = src[out_start:out_end, :]
        else:
            rows = obs_idx[out_start:out_end]
            block = src[rows, :]

        if var_idx is not None:
            block = block[:, var_idx]

        dst[out_start:out_end, :] = block


def subset_sparse_matrix_group(
    src: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
) -> None:
    enc = src.attrs.get("encoding-type", b"")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")

    if enc not in ("csr_matrix", "csc_matrix"):
        raise ValueError(f"Unsupported sparse encoding type: {enc}")

    data = np.asarray(src["data"][...])
    indices = np.asarray(src["indices"][...], dtype=np.int64)
    indptr = np.asarray(src["indptr"][...], dtype=np.int64)
    shape = src.attrs.get("shape", None)
    if shape is None:
        raise ValueError("Sparse matrix group missing 'shape' attribute.")
    n_rows, n_cols = int(shape[0]), int(shape[1])

    if enc == "csr_matrix":
        row_idx = obs_idx if obs_idx is not None else np.arange(n_rows, dtype=np.int64)
        col_idx = var_idx if var_idx is not None else np.arange(n_cols, dtype=np.int64)

        new_data = []
        new_indices = []
        new_indptr = [0]

        for r in row_idx:
            start = indptr[r]
            end = indptr[r + 1]
            row_cols = indices[start:end]
            row_data = data[start:end]

            if var_idx is not None:
                col_mask = np.isin(row_cols, col_idx)
                row_cols = row_cols[col_mask]
                row_data = row_data[col_mask]

            if var_idx is not None:
                col_map = {c: i for i, c in enumerate(col_idx)}
                row_cols = np.array([col_map[c] for c in row_cols], dtype=np.int64)

            new_indices.extend(row_cols.tolist())
            new_data.extend(row_data.tolist())
            new_indptr.append(len(new_indices))

        new_shape = (len(row_idx), len(col_idx))
    else:
        row_idx = obs_idx if obs_idx is not None else np.arange(n_rows, dtype=np.int64)
        col_idx = var_idx if var_idx is not None else np.arange(n_cols, dtype=np.int64)

        new_data = []
        new_indices = []
        new_indptr = [0]

        for c in col_idx:
            start = indptr[c]
            end = indptr[c + 1]
            col_rows = indices[start:end]
            col_data = data[start:end]

            if obs_idx is not None:
                row_mask = np.isin(col_rows, row_idx)
                col_rows = col_rows[row_mask]
                col_data = col_data[row_mask]

            if obs_idx is not None:
                row_map = {r: i for i, r in enumerate(row_idx)}
                col_rows = np.array([row_map[r] for r in col_rows], dtype=np.int64)

            new_indices.extend(col_rows.tolist())
            new_data.extend(col_data.tolist())
            new_indptr.append(len(new_indices))

        new_shape = (len(row_idx), len(col_idx))

    group = dst_parent.create_group(name)
    group.attrs["encoding-type"] = enc
    group.attrs["encoding-version"] = "0.1.0"
    if is_zarr_group(group):
        group.attrs["shape"] = list(new_shape)
    else:
        group.attrs["shape"] = np.array(new_shape, dtype=np.int64)

    create_dataset(group, "data", data=np.array(new_data, dtype=data.dtype))
    create_dataset(group, "indices", data=np.array(new_indices, dtype=indices.dtype))
    create_dataset(group, "indptr", data=np.array(new_indptr, dtype=indptr.dtype))


def subset_matrix_entry(
    obj: Any,
    dst_parent: Any,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int,
    entry_label: str,
) -> None:
    if is_dataset(obj):
        subset_dense_matrix(
            obj, dst_parent, name, obs_idx, var_idx, chunk_rows=chunk_rows
        )
        return

    if is_group(obj):
        enc = obj.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")
        if enc in ("csr_matrix", "csc_matrix"):
            subset_sparse_matrix_group(obj, dst_parent, name, obs_idx, var_idx)
            return
        raise ValueError(f"Unsupported {entry_label} encoding type: {enc}")

    raise ValueError(f"Unsupported {entry_label} object type")


def subset_h5ad(
    file: Path,
    output: Path,
    obs_file: Optional[Path],
    var_file: Optional[Path],
    *,
    chunk_rows: int = 1024,
    console: Console,
) -> None:
    obs_keep: Optional[Set[str]] = None
    if obs_file is not None:
        obs_keep = _read_name_file(obs_file)
        console.print(f"[cyan]Found {len(obs_keep)} obs names to keep[/]")

    var_keep: Optional[Set[str]] = None
    if var_file is not None:
        var_keep = _read_name_file(var_file)
        console.print(f"[cyan]Found {len(var_keep)} var names to keep[/]")

    if obs_keep is None and var_keep is None:
        raise ValueError("At least one of --obs or --var must be provided.")

    with console.status("[magenta]Opening files...[/]"):
        with open_store(file, "r") as src_store, open_store(output, "w") as dst_store:
            src = src_store.root
            dst = dst_store.root

            obs_idx = None
            if obs_keep is not None:
                console.print("[cyan]Matching obs names...[/]")
                obs_group = src["obs"]
                obs_index = _decode_attr(obs_group.attrs.get("_index", "obs_names"))
                obs_names_ds = _group_get(obs_group, "obs_names") or _group_get(
                    obs_group, obs_index
                )
                if obs_names_ds is None:
                    raise KeyError("Could not find obs names")

                obs_idx, missing_obs = indices_from_name_set(obs_names_ds, obs_keep)
                if missing_obs:
                    console.print(
                        f"[yellow]Warning: {len(missing_obs)} obs names not found in file[/]"
                    )
                console.print(
                    f"[green]Selected {len(obs_idx)} obs (of {obs_names_ds.shape[0]})[/]"
                )

            var_idx = None
            if var_keep is not None:
                console.print("[cyan]Matching var names...[/]")
                var_group = src["var"]
                var_index = _decode_attr(var_group.attrs.get("_index", "var_names"))
                var_names_ds = _group_get(var_group, "var_names") or _group_get(
                    var_group, var_index
                )
                if var_names_ds is None:
                    raise KeyError("Could not find var names")

                var_idx, missing_var = indices_from_name_set(var_names_ds, var_keep)
                if missing_var:
                    console.print(
                        f"[yellow]Warning: {len(missing_var)} var names not found in file[/]"
                    )
                console.print(
                    f"[green]Selected {len(var_idx)} var (of {var_names_ds.shape[0]})[/]"
                )

            tasks: List[str] = []
            if "obs" in src:
                tasks.append("obs")
            if "var" in src:
                tasks.append("var")
            if "X" in src:
                tasks.append("X")
            if "layers" in src:
                tasks.extend([f"layer:{k}" for k in src["layers"].keys()])
            if "obsm" in src:
                tasks.extend([f"obsm:{k}" for k in src["obsm"].keys()])
            if "varm" in src:
                tasks.extend([f"varm:{k}" for k in src["varm"].keys()])
            if "obsp" in src:
                tasks.extend([f"obsp:{k}" for k in src["obsp"].keys()])
            if "varp" in src:
                tasks.extend([f"varp:{k}" for k in src["varp"].keys()])
            if "uns" in src:
                tasks.append("uns")

            with Progress(
                SpinnerColumn(finished_text="[green]✓[/]"),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False,
            ) as progress:
                for task in tasks:
                    task_id = progress.add_task(
                        f"[cyan]Subsetting {task}...[/]", total=None
                    )
                    if task == "obs":
                        obs_dst = dst.create_group("obs")
                        subset_axis_group(src["obs"], obs_dst, obs_idx)
                    elif task == "var":
                        var_dst = dst.create_group("var")
                        subset_axis_group(src["var"], var_dst, var_idx)
                    elif task == "X":
                        X = src["X"]
                        if is_dataset(X):
                            subset_dense_matrix(
                                X, dst, "X", obs_idx, var_idx, chunk_rows=chunk_rows
                            )
                        elif is_group(X):
                            subset_sparse_matrix_group(X, dst, "X", obs_idx, var_idx)
                        else:
                            copy_tree(X, dst, "X")
                    elif task.startswith("layer:"):
                        key = task.split(":", 1)[1]
                        layer_src = src["layers"][key]
                        layers_dst = _ensure_group(dst, "layers")
                        subset_matrix_entry(
                            layer_src,
                            layers_dst,
                            key,
                            obs_idx,
                            var_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"layer:{key}",
                        )
                    elif task.startswith("obsm:"):
                        key = task.split(":", 1)[1]
                        obsm_dst = _ensure_group(dst, "obsm")
                        obsm_obj = src["obsm"][key]
                        subset_matrix_entry(
                            obsm_obj,
                            obsm_dst,
                            key,
                            obs_idx,
                            None,
                            chunk_rows=chunk_rows,
                            entry_label=f"obsm:{key}",
                        )
                    elif task.startswith("varm:"):
                        key = task.split(":", 1)[1]
                        varm_dst = _ensure_group(dst, "varm")
                        varm_obj = src["varm"][key]
                        subset_matrix_entry(
                            varm_obj,
                            varm_dst,
                            key,
                            var_idx,
                            None,
                            chunk_rows=chunk_rows,
                            entry_label=f"varm:{key}",
                        )
                    elif task.startswith("obsp:"):
                        key = task.split(":", 1)[1]
                        obsp_dst = _ensure_group(dst, "obsp")
                        obsp_obj = src["obsp"][key]
                        subset_matrix_entry(
                            obsp_obj,
                            obsp_dst,
                            key,
                            obs_idx,
                            obs_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"obsp:{key}",
                        )
                    elif task.startswith("varp:"):
                        key = task.split(":", 1)[1]
                        varp_dst = _ensure_group(dst, "varp")
                        varp_obj = src["varp"][key]
                        subset_matrix_entry(
                            varp_obj,
                            varp_dst,
                            key,
                            var_idx,
                            var_idx,
                            chunk_rows=chunk_rows,
                            entry_label=f"varp:{key}",
                        )
                    elif task == "uns":
                        copy_tree(src["uns"], dst, "uns")
                    progress.update(
                        task_id,
                        description=f"[green]Subsetting {task}[/]",
                        completed=1,
                        total=1,
                    )
