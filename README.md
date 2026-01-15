# h5ad CLI

A command-line tool for exploring huge `.h5ad` (AnnData) files without loading them fully into memory. Streams data directly from disk for efficient inspection of structure, metadata, and matrices.

## Features

- **`info`** – Show file structure and dimensions (`n_obs × n_var`)
- **`table`** – Export obs/var metadata to CSV with chunked streaming
- **`subset`** – Filter h5ad files by cell/gene names (supports dense and sparse CSR/CSC matrices)
- Memory-efficient chunked processing for large files
- Rich terminal output with colors and progress bars

## Installation

```bash
uv sync
```

For development and testing:
```bash
uv sync --extra dev
```

See [docs/TESTING.md](docs/TESTING.md) for testing documentation.

## Usage
Invoke any subcommand via `uv run h5ad ...`:

```bash
uv run h5ad --help
```

#### Examples

**Inspect overall structure and axis sizes:**
```bash
uv run h5ad info data.h5ad
```

**Export full obs metadata to CSV:**
```bash
uv run h5ad table data.h5ad --axis obs --out obs_metadata.csv
```

**Export selected obs columns to stdout:**
```bash
uv run h5ad table data.h5ad --axis obs --cols cell_type,donor
```

**Export var metadata with custom chunk size:**
```bash
uv run h5ad table data.h5ad --axis var --chunk-rows 5000 --out var_metadata.csv
```

**Subset by cell names:**
```bash
uv run h5ad subset input.h5ad output.h5ad --obs cells.txt
```

**Subset by both cells and genes:**
```bash
uv run h5ad subset input.h5ad output.h5ad --obs cells.txt --var genes.txt
```

All commands stream from disk, so even multi-GB `.h5ad` files remain responsive.
