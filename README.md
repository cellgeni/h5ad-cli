## h5ad CLI

h5ad is a Typer-powered command-line tool for exploring huge `.h5ad` (AnnData) files without loading them fully into memory. It streams data directly from disk so you can inspect structure, metadata, and matrices on demand.

### Features
- **`info`** – Reports `n_obs × n_var` dimensions and highlights top-level groups with their child keys.
- **`table`** – Exports obs or var metadata tables to CSV in memory-efficient chunks.
- Colored Rich output for clearer, more informative summaries.

### Usage
Install dependencies with `uv sync`, then invoke any subcommand via `uv run h5ad ...`:

```bash
uv sync
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

Each command streams from disk, so even multi-GB `.h5ad` files remain responsive.
