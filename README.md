# h5ad CLI

A command-line tool for exploring huge AnnData stores (`.h5ad` and `.zarr`) without loading them fully into memory. Streams data directly from disk for efficient inspection of structure, metadata, and matrices.

## Features

- Streaming access to very large `.h5ad` and `.zarr` stores
- Auto-detects `.h5ad` files vs `.zarr` directories
- Chunked processing for dense and sparse matrices (CSR/CSC)
- Rich terminal output with progress indicators

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):
```bash
git clone https://github.com/cellgeni/h5ad-cli.git
cd h5ad-cli
uv sync
```

For development and testing:
```bash
uv sync --extra dev
```

Alternative with pip:
```bash
git clone https://github.com/cellgeni/h5ad-cli.git
cd h5ad-cli
pip install .
```

For development and testing with pip:
```bash
pip install -e ".[dev]"
```

See [docs/TESTING.md](docs/TESTING.md) for testing documentation.

## Commands (Overview)

Run help at any level (e.g. `uv run h5ad --help`, `uv run h5ad export --help`).

- `info` – read-only inspection of store layout, shapes, and type hints; supports drilling into paths like `obsm/X_pca` or `uns`.
- `subset` – stream and write a filtered copy based on obs/var name lists, preserving dense and sparse matrix encodings.
- `export` – extract data from a store; subcommands: `dataframe` (obs/var to CSV), `array` (dense to `.npy`), `sparse` (CSR/CSC to `.mtx`), `dict` (JSON), `image` (PNG).
- `import` – write new data into a store; subcommands: `dataframe` (CSV → obs/var), `array` (`.npy`), `sparse` (`.mtx`), `dict` (JSON).

See [docs/GET_STARTED.md](docs/GET_STARTED.md) for a short tutorial.