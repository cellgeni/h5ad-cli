## h5ad CLI

h5ad is a Typer-powered command-line tool for exploring huge `.h5ad` (AnnData) files without loading them fully into memory. It streams data directly from disk so you can inspect structure, metadata, and matrices on demand.

### Features
- Fast `info` command that reports `n_obs × n_var` and highlights the top-level groups plus their child keys.
- Additional subcommands (ls, table, matrix, subset-obs-range) designed for incremental, memory-safe inspection of AnnData contents.
- Colored Rich output for clearer, more informative summaries.

### Usage
Install dependencies with `uv sync`, then run any subcommand, for example:

```
uv run h5ad info path/to/data.h5ad
```

This prints a formatted overview of the file, including axis sizes and group structure.
