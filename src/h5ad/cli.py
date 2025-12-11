from pathlib import Path
from typing import Optional, Sequence

import rich
import typer
import h5py
import numpy as np

app = typer.Typer(help="Streaming CLI for huge .h5ad files (info, ls, table, matrix, subset-obs-range).")


def open_file(path: Path) -> h5py.File:
    """Open an h5ad file in read-only mode."""
    return h5py.File(path, "r")


def axis_len(file: h5py.File, axis: str) -> Optional[int]:
    """Get the length of the specified axis ('obs' or 'var') in the h5ad file."""
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
        index_name = index_name.decode('utf-8')
    
    if index_name not in group:
        return None
    
    # Return the length of the index dataset
    dataset = group[index_name]
    if not isinstance(dataset, h5py.Dataset):
        return None
    if dataset.shape:
        return int(dataset.shape[0])
    return None


def _decode_str_array(array: np.ndarray) -> np.ndarray:
    """Turn HDF5 bytes/objects into plain unicode strings."""
    if np.issubdtype(array.dtype, np.bytes_):
        return array.astype("U")
    if array.dtype.kind == "O":
        return array.astype(str)
    return array.astype(str)


@app.command()
def info(file: Path = typer.Argument(..., help="Path to the .h5ad file")) -> None:
    with open_file(file) as f:
        # Get n_obs and n_var
        n_obs = axis_len(f, "obs")
        n_var = axis_len(f, "var")
        #rich.print(f"[bold red]File:[/] [white]{file}[/white]")
        #console.rule(f"[bold blue]File[/] • {file}")
        rich.print(f"[bold cyan]An object with n_obs × n_var: {n_obs if n_obs is not None else '?'} × {n_var if n_var is not None else '?'}[/]")
        # List top-level keys and their sub-keys
        for key, obj in sorted(f.items(), key=lambda x: len(x[0])):
            sub_keys = [k for k in obj.keys() if k != "_index"]
            if sub_keys and key != "X":
                rich.print(f"\t[bold yellow]{key}:[/]\t" + ", ".join(f"[bright_white]{sub}[/]" for sub in sub_keys))



def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)