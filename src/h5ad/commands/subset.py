"""Subset operations for .h5ad files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict, Any

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from h5ad.read import decode_str_array


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    """
    Copy HDF5 attributes from source to destination.
    Args:
        src (h5py.AttributeManager): Source attributes
        dst (h5py.AttributeManager): Destination attributes
    """
    for k, v in src.items():
        dst[k] = v


def _ds_create_kwargs(src: h5py.Dataset) -> Dict[str, Any]:
    """
    Best-effort carryover of dataset creation properties.
    (h5py doesn't expose everything perfectly; this covers the big ones.)

    Args:
        src (h5py.Dataset): Source dataset
    Returns:
        Dict[str, Any]: Dataset creation keyword arguments
    """
    kw: Dict[str, Any] = {}
    if src.chunks is not None:
        kw["chunks"] = src.chunks
    if src.compression is not None:
        kw["compression"] = src.compression
        kw["compression_opts"] = src.compression_opts
    kw["shuffle"] = bool(src.shuffle)
    kw["fletcher32"] = bool(src.fletcher32)
    if src.scaleoffset is not None:
        kw["scaleoffset"] = src.scaleoffset
    if src.fillvalue is not None:
        kw["fillvalue"] = src.fillvalue
    return kw


def _read_name_file(path: Path) -> Set[str]:
    """
    Read one name per line from a file. Blank lines ignored.
    """
    names: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(line)
    return names


def indices_from_name_set(
    names_ds: h5py.Dataset,
    keep: Set[str],
    *,
    chunk_size: int = 200_000,
) -> Tuple[np.ndarray, Set[str]]:
    """
    Returns (indices_sorted, missing_names).
    Chunked scan so we don't do names_ds[...] for huge datasets.

    Args:
        names_ds (h5py.Dataset): Dataset containing names
        keep (Set[str]): Set of names to find
        chunk_size (int): Number of names to read per chunk

    Returns:
        Tuple[np.ndarray, Set[str]]: (Array of found indices, set of missing names)
    """
    if names_ds.ndim != 1:
        # common h5ad uses 1D obs_names/var_names
        flat_len = int(np.prod(names_ds.shape))
    else:
        flat_len = names_ds.shape[0]

    remaining = set(keep)  # we'll delete as we find
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
    src: h5py.Group,
    dst: h5py.Group,
    indices: Optional[np.ndarray],
) -> None:
    """
    Subset obs/var group:
    - datasets: subset along first axis (obj[indices, ...])
    - categorical groups: copy categories, subset codes
    - unknown groups: copy as-is if indices is None; otherwise copy conservatively

    Args:
        src (h5py.Group): Source axis group
        dst (h5py.Group): Destination axis group
        indices (Optional[np.ndarray]): Indices to keep; if None, copy as-is
    """
    _copy_attrs(src.attrs, dst.attrs)

    for key in src.keys():
        obj = src[key]

        if isinstance(obj, h5py.Dataset):
            if indices is None:
                src.copy(key, dst, name=key)
            else:
                data = obj[indices, ...]
                ds = dst.create_dataset(key, data=data)
                _copy_attrs(obj.attrs, ds.attrs)

        elif isinstance(obj, h5py.Group):
            enc = obj.attrs.get("encoding-type", b"")
            if isinstance(enc, bytes):
                enc = enc.decode("utf-8")

            if enc == "categorical":
                gdst = dst.create_group(key)
                _copy_attrs(obj.attrs, gdst.attrs)
                obj.copy("categories", gdst, name="categories")

                codes = obj["codes"]
                if indices is None:
                    obj.copy("codes", gdst, name="codes")
                else:
                    codes_sub = codes[indices, ...]
                    ds = gdst.create_dataset("codes", data=codes_sub)
                    _copy_attrs(codes.attrs, ds.attrs)
            else:
                if indices is None:
                    src.copy(key, dst, name=key)
                else:
                    src.copy(key, dst, name=key)


def subset_dense_matrix(
    src: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int = 1024,
) -> None:
    """
    Chunked write for dense 2D datasets.
    Args:
        src (h5py.Dataset): Source dense matrix dataset
        dst_parent (h5py.Group): Destination parent group
        name (str): Name for the destination dataset
        obs_idx (Optional[np.ndarray]): Indices of observations to keep
        var_idx (Optional[np.ndarray]): Indices of variables to keep
        chunk_rows (int): Number of rows to read per chunk
    """
    if src.ndim != 2:
        # fallback: copy whole dataset
        src.parent.copy(src.name.split("/")[-1], dst_parent, name=name)
        return

    n_obs, n_var = src.shape
    out_obs = len(obs_idx) if obs_idx is not None else n_obs
    out_var = len(var_idx) if var_idx is not None else n_var

    kw = _ds_create_kwargs(src)
    # adjust chunks to output shape if possible
    if "chunks" in kw and kw["chunks"] is not None:
        c0, c1 = kw["chunks"]
        kw["chunks"] = (min(c0, out_obs), min(c1, out_var))

    dst = dst_parent.create_dataset(
        name, shape=(out_obs, out_var), dtype=src.dtype, **kw
    )
    _copy_attrs(src.attrs, dst.attrs)

    # Write in blocks of output rows
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
    src: h5py.Group,
    dst_parent: h5py.Group,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
) -> None:
    """
    Subset a sparse matrix stored as an h5ad group with datasets:
      - data, indices, indptr
    Supports both CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) formats.

    CSR: rows are compressed, efficient for row-wise operations
    CSC: columns are compressed, efficient for column-wise operations

    Args:
        src (h5py.Group): Source sparse matrix group
        dst_parent (h5py.Group): Destination parent group
        name (str): Name for the destination group
        obs_idx (Optional[np.ndarray]): Indices of observations to keep
        var_idx (Optional[np.ndarray]): Indices of variables to keep
    """
    data = src["data"]
    indices = src["indices"]
    indptr = src["indptr"]

    # Determine format
    encoding = src.attrs.get("encoding-type", b"")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")

    is_csr = encoding == "csr_matrix"
    is_csc = encoding == "csc_matrix"

    if not is_csr and not is_csc:
        raise ValueError(f"Unsupported sparse format: {encoding}")

    # Determine shape
    shape = src.attrs.get("shape", None)
    if shape is None:
        # fallback: infer from indptr len and max index
        major_dim = indptr.shape[0] - 1
        minor_dim = int(indices[...].max()) + 1 if indices.size else 0
        if is_csr:
            n_obs, n_var = major_dim, minor_dim
        else:  # CSC
            n_obs, n_var = minor_dim, major_dim
    else:
        n_obs, n_var = shape

    # For CSR: major axis = obs (rows), minor axis = var (cols)
    # For CSC: major axis = var (cols), minor axis = obs (rows)
    if is_csr:
        major_idx = obs_idx if obs_idx is not None else np.arange(n_obs, dtype=np.int64)
        minor_idx = var_idx
        out_obs = major_idx.shape[0]
        out_var = minor_idx.shape[0] if minor_idx is not None else n_var
    else:  # CSC
        major_idx = var_idx if var_idx is not None else np.arange(n_var, dtype=np.int64)
        minor_idx = obs_idx
        out_obs = minor_idx.shape[0] if minor_idx is not None else n_obs
        out_var = major_idx.shape[0]

    # Build minor axis remap if needed
    minor_map = None
    out_minor_dim = out_var if is_csr else out_obs
    total_minor_dim = n_var if is_csr else n_obs

    if minor_idx is not None:
        # array remap is fastest; if dimension is huge and memory matters, use dict instead
        minor_map = np.full(total_minor_dim, -1, dtype=np.int64)
        minor_map[minor_idx] = np.arange(minor_idx.shape[0], dtype=np.int64)

    # Pass 1: count nnz in output to preallocate
    out_counts = np.zeros(len(major_idx), dtype=np.int64)
    for i, major_pos in enumerate(major_idx):
        s = int(indptr[major_pos])
        e = int(indptr[major_pos + 1])
        if s == e:
            continue
        minor_indices = indices[s:e]
        if minor_map is None:
            out_counts[i] = e - s
        else:
            mask = minor_map[minor_indices] >= 0
            out_counts[i] = mask.sum()

    out_indptr = np.zeros(len(major_idx) + 1, dtype=indptr.dtype)
    np.cumsum(out_counts, out=out_indptr[1:])
    out_nnz = int(out_indptr[-1])

    # Preallocate output arrays
    out_data = np.empty(out_nnz, dtype=data.dtype)
    out_indices = np.empty(out_nnz, dtype=indices.dtype)

    # Pass 2: fill
    cursor = 0
    for i, major_pos in enumerate(major_idx):
        s = int(indptr[major_pos])
        e = int(indptr[major_pos + 1])
        if s == e:
            continue

        minor_indices = indices[s:e]
        vals = data[s:e]

        if minor_map is None:
            length = e - s
            out_indices[cursor : cursor + length] = minor_indices
            out_data[cursor : cursor + length] = vals
            cursor += length
        else:
            mask = minor_map[minor_indices] >= 0
            new_minor = minor_map[minor_indices[mask]]
            new_vals = vals[mask]
            length = len(new_minor)
            out_indices[cursor : cursor + length] = new_minor
            out_data[cursor : cursor + length] = new_vals
            cursor += length

    # Create dst group
    gdst = dst_parent.create_group(name)
    _copy_attrs(src.attrs, gdst.attrs)
    gdst.attrs["shape"] = (out_obs, out_var)
    gdst.attrs["encoding-type"] = encoding

    # Write datasets (best-effort preserve compression/etc.)
    # Adjust chunks to not exceed output size
    data_kw = _ds_create_kwargs(data)
    if "chunks" in data_kw and data_kw["chunks"] is not None:
        data_kw["chunks"] = (min(data_kw["chunks"][0], out_nnz),)
    d_data = gdst.create_dataset("data", data=out_data, **data_kw)
    _copy_attrs(data.attrs, d_data.attrs)

    indices_kw = _ds_create_kwargs(indices)
    if "chunks" in indices_kw and indices_kw["chunks"] is not None:
        indices_kw["chunks"] = (min(indices_kw["chunks"][0], out_nnz),)
    d_indices = gdst.create_dataset("indices", data=out_indices, **indices_kw)
    _copy_attrs(indices.attrs, d_indices.attrs)

    indptr_kw = _ds_create_kwargs(indptr)
    if "chunks" in indptr_kw and indptr_kw["chunks"] is not None:
        indptr_kw["chunks"] = (min(indptr_kw["chunks"][0], len(out_indptr)),)
    d_indptr = gdst.create_dataset("indptr", data=out_indptr, **indptr_kw)
    _copy_attrs(indptr.attrs, d_indptr.attrs)


def subset_matrix_like(
    src_obj: h5py.Dataset | h5py.Group,
    dst_parent: h5py.Group,
    name: str,
    obs_idx: Optional[np.ndarray],
    var_idx: Optional[np.ndarray],
    *,
    chunk_rows: int = 1024,
) -> None:
    """
    Dispatch for dense dataset vs sparse (csr/csc) group.
    Args:
        src_obj (h5py.Dataset | h5py.Group): Source dataset or group
        dst_parent (h5py.Group): Destination parent group
        name (str): Name for the destination dataset/group
        obs_idx (Optional[np.ndarray]): Indices of observations to keep
        var_idx (Optional[np.ndarray]): Indices of variables to keep
        chunk_rows (int): Number of rows to read per chunk when subsetting dense matrices
    """
    if isinstance(src_obj, h5py.Dataset):
        subset_dense_matrix(
            src_obj, dst_parent, name, obs_idx, var_idx, chunk_rows=chunk_rows
        )
        return

    # group
    enc = src_obj.attrs.get("encoding-type", b"")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")

    if enc in ("csr_matrix", "csc_matrix"):
        subset_sparse_matrix_group(src_obj, dst_parent, name, obs_idx, var_idx)
    else:
        # unknown sparse type -> copy as-is (or raise)
        src_obj.file.copy(src_obj, dst_parent, name)


def subset_h5ad(
    file: Path,
    output: Path,
    obs_file: Optional[Path],
    var_file: Optional[Path],
    *,
    chunk_rows: int = 1024,
    console: Console,
) -> None:
    """
    Subset an h5ad file by obs and/or var names.
    Args:
        file (Path): Input h5ad file path
        output (Path): Output h5ad file path
        obs_file (Optional[Path]): File with obs names to keep (one per line)
        var_file (Optional[Path]): File with var names to keep (one per line)
        chunk_rows (int): Number of rows to read per chunk when subsetting dense matrices
        console (Console): Rich console for output
    """
    # ---- Read keep-lists
    obs_keep: Optional[Set[str]] = None
    if obs_file is not None:
        obs_keep = _read_name_file(obs_file)
        console.print(f"[cyan]Found {len(obs_keep)} obs names to keep[/]")

    var_keep: Optional[Set[str]] = None
    if var_file is not None:
        var_keep = _read_name_file(var_file)
        console.print(f"[cyan]Found {len(var_keep)} var names to keep[/]")

    if obs_keep is None and var_keep is None:
        console.print(
            "[bold red]Error:[/] At least one of [cyan]--obs[/] or [cyan]--var[/] must be provided.",
        )
        raise typer.Exit(code=1)

    # ---- Open files
    with console.status("[magenta]Opening files...[/]"):
        src = h5py.File(file, "r")
        dst = h5py.File(output, "w")

    try:
        # ---- Compute indices
        obs_idx = None
        if obs_keep is not None:
            console.print("[cyan]Matching obs names...[/]")
            obs_names_ds = src["obs"].get("obs_names") or src["obs"].get(
                src["obs"].attrs.get("_index", "obs_names")
            )
            if obs_names_ds is None:
                console.print("[bold red]Error:[/] Could not find obs names")
                raise RuntimeError("Could not find obs names")

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
            var_names_ds = src["var"].get("var_names") or src["var"].get(
                src["var"].attrs.get("_index", "var_names")
            )
            if var_names_ds is None:
                console.print("[bold red]Error:[/] Could not find var names")
                raise RuntimeError("Could not find var names")

            var_idx, missing_var = indices_from_name_set(var_names_ds, var_keep)
            if missing_var:
                console.print(
                    f"[yellow]Warning: {len(missing_var)} var names not found in file[/]"
                )
            console.print(
                f"[green]Selected {len(var_idx)} var (of {var_names_ds.shape[0]})[/]"
            )

        # ---- Build task list
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

        # ---- Progress bar for all operations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]Subsetting...", total=len(tasks))
            processed_top: Set[str] = set()

            # obs
            if "obs" in src:
                progress.update(task_id, description="[cyan]Subsetting obs...[/]")
                obs_dst = dst.create_group("obs")
                subset_axis_group(src["obs"], obs_dst, obs_idx)
                processed_top.add("obs")
                progress.advance(task_id)

            # var
            if "var" in src:
                progress.update(task_id, description="[cyan]Subsetting var...[/]")
                var_dst = dst.create_group("var")
                subset_axis_group(src["var"], var_dst, var_idx)
                processed_top.add("var")
                progress.advance(task_id)

            # X
            if "X" in src:
                progress.update(task_id, description="[cyan]Subsetting X...[/]")
                subset_matrix_like(
                    src["X"], dst, "X", obs_idx, var_idx, chunk_rows=chunk_rows
                )
                processed_top.add("X")
                progress.advance(task_id)

            # layers
            if "layers" in src:
                layers_dst = dst.create_group("layers")
                processed_top.add("layers")
                for lname in src["layers"].keys():
                    progress.update(
                        task_id, description=f"[cyan]Subsetting layer: {lname}...[/]"
                    )
                    subset_matrix_like(
                        src["layers"][lname],
                        layers_dst,
                        lname,
                        obs_idx,
                        var_idx,
                        chunk_rows=chunk_rows,
                    )
                    progress.advance(task_id)

            # obsm
            if "obsm" in src:
                obsm_dst = dst.create_group("obsm")
                processed_top.add("obsm")
                for k in src["obsm"].keys():
                    if obs_idx is None:
                        progress.update(
                            task_id, description=f"[cyan]Copying obsm: {k}...[/]"
                        )
                        src["obsm"].copy(k, obsm_dst, name=k)
                    else:
                        progress.update(
                            task_id, description=f"[cyan]Subsetting obsm: {k}...[/]"
                        )
                        obj = src["obsm"][k]
                        if isinstance(obj, h5py.Dataset):
                            data = obj[obs_idx, ...]
                            obsm_dst.create_dataset(k, data=data)
                            for ak, av in obj.attrs.items():
                                obsm_dst[k].attrs[ak] = av
                        else:
                            subset_matrix_like(
                                obj, obsm_dst, k, obs_idx, None, chunk_rows=chunk_rows
                            )
                    progress.advance(task_id)

            # varm
            if "varm" in src:
                varm_dst = dst.create_group("varm")
                processed_top.add("varm")
                for k in src["varm"].keys():
                    if var_idx is None:
                        progress.update(
                            task_id, description=f"[cyan]Copying varm: {k}...[/]"
                        )
                        src["varm"].copy(k, varm_dst, name=k)
                    else:
                        progress.update(
                            task_id, description=f"[cyan]Subsetting varm: {k}...[/]"
                        )
                        obj = src["varm"][k]
                        if isinstance(obj, h5py.Dataset):
                            data = obj[var_idx, ...]
                            varm_dst.create_dataset(k, data=data)
                            for ak, av in obj.attrs.items():
                                varm_dst[k].attrs[ak] = av
                        else:
                            subset_matrix_like(
                                obj, varm_dst, k, var_idx, None, chunk_rows=chunk_rows
                            )
                    progress.advance(task_id)

            # obsp
            if "obsp" in src:
                obsp_dst = dst.create_group("obsp")
                processed_top.add("obsp")
                for k in src["obsp"].keys():
                    if obs_idx is None:
                        progress.update(
                            task_id, description=f"[cyan]Copying obsp: {k}...[/]"
                        )
                        src["obsp"].copy(k, obsp_dst, name=k)
                    else:
                        progress.update(
                            task_id, description=f"[cyan]Subsetting obsp: {k}...[/]"
                        )
                        subset_matrix_like(
                            src["obsp"][k],
                            obsp_dst,
                            k,
                            obs_idx,
                            obs_idx,
                            chunk_rows=chunk_rows,
                        )
                    progress.advance(task_id)

            # varp
            if "varp" in src:
                varp_dst = dst.create_group("varp")
                processed_top.add("varp")
                for k in src["varp"].keys():
                    if var_idx is None:
                        progress.update(
                            task_id, description=f"[cyan]Copying varp: {k}...[/]"
                        )
                        src["varp"].copy(k, varp_dst, name=k)
                    else:
                        progress.update(
                            task_id, description=f"[cyan]Subsetting varp: {k}...[/]"
                        )
                        subset_matrix_like(
                            src["varp"][k],
                            varp_dst,
                            k,
                            var_idx,
                            var_idx,
                            chunk_rows=chunk_rows,
                        )
                    progress.advance(task_id)

            # uns
            if "uns" in src:
                progress.update(task_id, description="[cyan]Copying uns...[/]")
                src.copy("uns", dst)
                processed_top.add("uns")
                progress.advance(task_id)

            # copy any remaining top-level keys
            for key in src.keys():
                if key not in processed_top:
                    src.copy(key, dst)

            # top-level attrs
            for ak, av in src.attrs.items():
                dst.attrs[ak] = av

        console.print(f"[bold green]✓ Successfully created {output}[/]")

    finally:
        dst.close()
        src.close()
