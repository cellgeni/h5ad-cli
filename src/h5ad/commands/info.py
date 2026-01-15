import h5py
import rich
from rich.console import Console
from pathlib import Path
from h5ad.info import axis_len, get_axis_group

def show_info(file: Path, console: Console) -> None:
    """
    Show high-level information about the .h5ad file.
    Args:
        file (_type_): 
        console (_type_): _description_
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
            sub_keys = [k for k in obj.keys() if k != "_index"]
            if sub_keys and key != "X":
                rich.print(
                    f"\t[bold yellow]{key}:[/]\t"
                    + ", ".join(f"[bright_white]{sub}[/]" for sub in sub_keys)
                )