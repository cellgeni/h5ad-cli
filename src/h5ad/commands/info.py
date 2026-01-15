from pathlib import Path

import h5py
import rich
from rich.console import Console
from h5ad.info import axis_len


def show_info(file: Path, console: Console) -> None:
    """
    Show high-level information about the .h5ad file.
    Args:
        file (Path): Path to the .h5ad file
        console (Console): Rich console for output
    """
    with h5py.File(file, "r") as f:
        # Get n_obs and n_var
        n_obs = axis_len(f, "obs")
        n_var = axis_len(f, "var")
        rich.print(
            f"[bold cyan]An object with n_obs × n_var: {n_obs if n_obs is not None else '?'} × {n_var if n_var is not None else '?'}[/]"
        )
        # List top-level keys and their sub-keys
        for key, obj in sorted(f.items(), key=lambda x: len(x[0])):
            # Only process Groups, skip Datasets like X
            if isinstance(obj, h5py.Group):
                sub_keys = [k for k in obj.keys() if k != "_index"]
                if sub_keys and key != "X":
                    rich.print(
                        f"\t[bold yellow]{key}:[/]\t"
                        + ", ".join(f"[bright_white]{sub}[/]" for sub in sub_keys)
                    )
