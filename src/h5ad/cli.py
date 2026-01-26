"""CLI for h5ad files with export and import subcommands."""

from pathlib import Path
from typing import Optional, Sequence, List

from rich.console import Console
import typer

from h5ad.commands import (
    show_info,
    subset_h5ad,
    export_mtx,
    export_npy,
    export_json,
    export_image,
    export_table,
)

app = typer.Typer(
    help="Streaming CLI for huge .h5ad files (info, subset, export, import)."
)
# Use stderr for status/progress to keep stdout clean for data output
# force_terminal=True ensures Rich output is visible even in non-TTY environments
console = Console(stderr=True, force_terminal=True)

# Create sub-apps for export and import
export_app = typer.Typer(help="Export objects from h5ad files.")
import_app = typer.Typer(help="Import objects into h5ad files.")
app.add_typer(export_app, name="export")
app.add_typer(import_app, name="import")


# ============================================================================
# INFO command
# ============================================================================
@app.command()
def info(
    file: Path = typer.Argument(
        ...,
        help="Path to the .h5ad file",
        exists=True,
        readable=True,
    ),
    entry: Optional[str] = typer.Argument(
        None,
        help="Entry path to inspect (e.g., 'obsm/X_pca', 'X', 'uns')",
    ),
    types: bool = typer.Option(
        False,
        "--types",
        "-t",
        help="Show detailed type information for all entries",
    ),
    depth: int = typer.Option(
        None,
        "--depth",
        "-d",
        help="Maximum recursion depth for type display (only with --types)",
    ),
) -> None:
    """
    Show high-level information about the .h5ad file.

    Use --types to see type information for each entry.
    Use --entry to inspect a specific entry in detail.

    Examples:
        h5ad info data.h5ad
        h5ad info --types data.h5ad
        h5ad info obsm/X_pca data.h5ad
    """
    try:
        show_info(file, console, show_types=types, depth=depth, entry_path=entry)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# SUBSET command
# ============================================================================
@app.command()
def subset(
    file: Path = typer.Argument(..., help="Input .h5ad", exists=True, readable=True),
    output: Path = typer.Argument(..., help="Output .h5ad", writable=True),
    obs: Optional[Path] = typer.Option(
        None,
        "--obs",
        help="File with obs names (one per line)",
        exists=True,
        readable=True,
    ),
    var: Optional[Path] = typer.Option(
        None,
        "--var",
        help="File with var names (one per line)",
        exists=True,
        readable=True,
    ),
    chunk_rows: int = typer.Option(
        1024, "--chunk", "-C", help="Row chunk size for dense matrices"
    ),
) -> None:
    """Subset an h5ad by obs and/or var names."""
    if obs is None and var is None:
        console.print(
            "[bold red]Error:[/] At least one of --obs or --var must be provided.",
        )
        raise typer.Exit(code=1)

    try:
        subset_h5ad(
            file=file,
            output=output,
            obs_file=obs,
            var_file=var,
            chunk_rows=chunk_rows,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# EXPORT subcommands
# ============================================================================
@export_app.command("dataframe")
def export_dataframe(
    file: Path = typer.Argument(
        ..., help="Path to the .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(..., help="Entry path to export ('obs' or 'var')"),
    output: Path = typer.Option(
        None, "--output", "-o", writable=True, help="Output CSV file path"
    ),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma separated column names to include",
    ),
    chunk_rows: int = typer.Option(
        10_000, "--chunk", "-C", help="Number of rows to read per chunk"
    ),
    head: Optional[int] = typer.Option(
        None, "--head", "-n", help="Output only the first n entries"
    ),
) -> None:
    """
    Export a dataframe (obs or var) to CSV.

    Examples:
        h5ad export dataframe data.h5ad obs --output obs.csv
        h5ad export dataframe data.h5ad var --output var.csv --columns gene_id,mean
        h5ad export dataframe data.h5ad obs --head 100
    """

    if entry not in ("obs", "var"):
        console.print(
            f"[bold red]Error:[/] Dataframe export is only supported for 'obs' or 'var' at this point, not '{entry}'.",
        )
        raise typer.Exit(code=1)

    col_list: Optional[List[str]] = None
    if columns:
        col_list = [col.strip() for col in columns.split(",") if col.strip()]

    try:
        export_table(
            file=file,
            axis=entry,
            columns=col_list,
            out=output,
            chunk_rows=chunk_rows,
            head=head,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("array")
def export_array(
    file: Path = typer.Argument(
        ..., help="Path to the .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'obsm/X_pca', 'varm/PCs', 'X')"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output .npy file path", writable=True
    ),
    chunk_elements: int = typer.Option(
        100_000,
        "--chunk",
        "-C",
        help="Number of elements to read per chunk",
    ),
) -> None:
    """
    Export a dense array or matrix to NumPy .npy format.

    Examples:
        h5ad export array data.h5ad obsm/X_pca pca.npy
        h5ad export array data.h5ad X matrix.npy
        h5ad export array data.h5ad varm/PCs loadings.npy
    """

    try:
        export_npy(
            file=file,
            obj=entry,
            out=output,
            chunk_elements=chunk_elements,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("sparse")
def export_sparse(
    file: Path = typer.Argument(
        ..., help="Path to the .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'X', 'layers/counts')"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        writable=True,
        help="Output .mtx file path (defaults to stdout)",
    ),
    head: Optional[int] = typer.Option(
        None, "--head", "-n", help="Output only the first n entries of mtx file"
    ),
    chunk_elements: int = typer.Option(
        1_000,
        "--chunk",
        "-C",
        help="Number of rows/columns (depends on compression format) to process per chunk",
    ),
    in_memory: bool = typer.Option(
        False,
        "--in-memory",
        "-m",
        help="Load the entire sparse matrix into memory before exporting (may be faster for small matrices)",
    ),
) -> None:
    """
    Export a sparse matrix (CSR/CSC) to Matrix Market (.mtx) format.

    Examples:
        h5ad export sparse data.h5ad X matrix.mtx
        h5ad export sparse data.h5ad layers/counts counts.mtx
        h5ad export sparse data.h5ad X --head 100
    """

    try:
        export_mtx(
            file=file,
            obj=entry,
            out=output,
            head=head,
            chunk_elements=chunk_elements,
            in_memory=in_memory,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("dict")
def export_dict(
    file: Path = typer.Argument(
        ..., help="Path to the .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to export (e.g., 'uns', 'uns/colors')"
    ),
    out: Path = typer.Argument(..., help="Output .json file path"),
    max_elements: int = typer.Option(
        1_000_000,
        "--max-elements",
        help="Maximum array elements for JSON export",
    ),
    include_attrs: bool = typer.Option(
        False, "--include-attrs", help="Include HDF5 attributes in JSON export"
    ),
) -> None:
    """
    Export a dict/group or scalar to JSON format.

    Examples:
        h5ad export dict data.h5ad uns metadata.json
        h5ad export dict data.h5ad uns/colors colors.json
    """

    try:
        export_json(
            file=file,
            obj=entry,
            out=out,
            max_elements=max_elements,
            include_attrs=include_attrs,
            console=console,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@export_app.command("image")
def export_image(
    file: Path = typer.Argument(
        ..., help="Path to the .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(..., help="Entry path to export (2D or 3D array)"),
    out: Path = typer.Argument(..., help="Output image file (.png, .jpg, .tiff)"),
) -> None:
    """
    Export an image-like array to PNG/JPG/TIFF format.

    The array should be 2D (H,W) or 3D (H,W,C) with C in {1,3,4}.

    Examples:
        h5ad export image data.h5ad uns/spatial/image tissue.png
    """

    try:
        export_image(file=file, obj=entry, out=out, console=console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# IMPORT subcommands
# ============================================================================
def _get_target_file(file: Path, output: Optional[Path], inplace: bool) -> Path:
    """Determine target file and copy if needed."""
    import shutil

    if inplace:
        return file
    if output is None:
        raise ValueError("Output file is required unless --inplace is specified.")
    shutil.copy2(file, output)
    console.print(f"[dim]Copied {file} → {output}[/]")
    return output


@import_app.command("dataframe")
def import_dataframe(
    file: Path = typer.Argument(
        ..., help="Path to the source .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to create/replace ('obs' or 'var')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input CSV file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad file path. Required unless --inplace.",
        writable=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
    index_column: Optional[str] = typer.Option(
        None,
        "--index-column",
        "-i",
        help="Column to use as index. Defaults to first column.",
    ),
) -> None:
    """
    Import a CSV file into obs or var.

    Examples:
        h5ad import dataframe data.h5ad obs cells.csv -o output.h5ad -i cell_id
        h5ad import dataframe data.h5ad var genes.csv --inplace -i gene_id
    """
    from h5ad.commands.import_data import _import_csv

    if entry not in ("obs", "var"):
        console.print(
            f"[bold red]Error:[/] Entry must be 'obs' or 'var', not '{entry}'.",
        )
        raise typer.Exit(code=1)

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_csv(target, entry, input_file, index_column, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("array")
def import_array(
    file: Path = typer.Argument(
        ..., help="Path to the source .h5ad file", exists=True, readable=True
    ),
    entry: str = typer.Argument(
        ..., help="Entry path to create/replace (e.g., 'X', 'obsm/X_pca')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .npy file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad file path. Required unless --inplace.",
        writable=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a NumPy .npy file as a dense array.

    Dimensions are validated against existing obs/var.

    Examples:
        h5ad import array data.h5ad obsm/X_pca pca.npy -o output.h5ad
        h5ad import array data.h5ad X matrix.npy --inplace
    """
    from h5ad.commands.import_data import _import_npy

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_npy(target, entry, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("sparse")
def import_sparse(
    file: Path = typer.Argument(
        ..., help="Path to the source .h5ad file", exists=True, readable=True
    ),
    obj: str = typer.Argument(
        ..., help="Object path to create/replace (e.g., 'X', 'layers/counts')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .mtx file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad file path. Required unless --inplace.",
        writable=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a Matrix Market (.mtx) file as a CSR sparse matrix.

    Dimensions are validated against existing obs/var.

    Examples:
        h5ad import sparse data.h5ad X matrix.mtx -o output.h5ad
        h5ad import sparse data.h5ad layers/counts counts.mtx --inplace
    """
    from h5ad.commands.import_data import _import_mtx

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_mtx(target, obj, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


@import_app.command("dict")
def import_dict(
    file: Path = typer.Argument(
        ..., help="Path to the source .h5ad file", exists=True, readable=True
    ),
    obj: str = typer.Argument(
        ..., help="Object path to create/replace (e.g., 'uns', 'uns/metadata')"
    ),
    input_file: Path = typer.Argument(
        ..., help="Input .json file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .h5ad file path. Required unless --inplace.",
        writable=True,
    ),
    inplace: bool = typer.Option(
        False,
        "--inplace",
        help="Modify source file directly.",
    ),
) -> None:
    """
    Import a JSON file into uns or other dict-like groups.

    Examples:
        h5ad import dict data.h5ad uns/metadata config.json -o output.h5ad
        h5ad import dict data.h5ad uns settings.json --inplace
    """
    from h5ad.commands.import_data import _import_json

    if not inplace and output is None:
        console.print(
            "[bold red]Error:[/] Output file is required. "
            "Use --output/-o or --inplace.",
        )
        raise typer.Exit(code=1)

    try:
        target = _get_target_file(file, output, inplace)
        _import_json(target, obj, input_file, console)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    app(standalone_mode=True)
