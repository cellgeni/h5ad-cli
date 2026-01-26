# AnnData on-disk element specifications — HDF5 (`.h5ad`)

This document describes how *elements* are encoded inside an AnnData **HDF5** container (`.h5ad`).  
It is intended to be GitHub-renderable Markdown (no Sphinx/MyST directives).

> **Scope**
>
> - “Modern” encoding metadata (`encoding-type`, `encoding-version`) is the convention used by **anndata ≥ 0.8**.
> - “Legacy” conventions (notably DataFrame categorical handling) are described for **anndata 0.7.x** files, which are still commonly encountered.

## Table of contents

- [Encoding metadata](#encoding-metadata)
- [AnnData group](#anndata-group)
- [Dense arrays](#dense-arrays)
- [Sparse arrays (CSR/CSC)](#sparse-arrays-csrcsc)
- [DataFrames](#dataframes)
  - [DataFrame v0.2.0](#dataframe-v020)
  - [DataFrame v0.1.0 (legacy: anndata 0.7.x)](#dataframe-v010-legacy-anndata-07x)
  - [Legacy categorical columns (Series-level)](#legacy-categorical-columns-series-level)
- [Mappings / dict](#mappings--dict)
- [Scalars](#scalars)
- [Categorical arrays](#categorical-arrays)
- [String arrays](#string-arrays)
- [Nullable arrays](#nullable-arrays)
  - [Missing value semantics](#missing-value-semantics)
- [Awkward arrays (experimental)](#awkward-arrays-experimental)
- [Sources](#sources)

## Encoding metadata

**Modern convention (anndata ≥ 0.8):**

- Any element (HDF5 *group* or *dataset*) that participates in the element-dispatch system:
  - **MUST** have attribute `encoding-type` (string)
  - **MUST** have attribute `encoding-version` (string, parseable as a version)

Readers should dispatch first on `encoding-type`, then on `encoding-version`.

**Legacy convention (anndata ≤ 0.7.x):**

- Many objects do *not* have `encoding-type`/`encoding-version`.
- Some elements (e.g. CSR/CSC sparse matrices, legacy DataFrames) *do* use `encoding-type`/`encoding-version`.
- Readers typically infer element kinds from:
  - known AnnData keys (`X`, `obs`, `var`, …),
  - group structure, and/or
  - legacy attributes (e.g. the `categories` attribute on categorical columns).

## AnnData group

### `encoding-type: anndata`, `encoding-version: 0.1.0`

An `AnnData` object **MUST** be stored as an HDF5 **group** with attributes:

- `encoding-type: "anndata"`
- `encoding-version: "0.1.0"`

Required members:

- `obs` — a [DataFrame](#dataframes)
- `var` — a [DataFrame](#dataframes)

Optional members (if present, they must satisfy these constraints):

- `X` — dense array or sparse array; shape `(n_obs, n_var)`
- `layers` — mapping; values dense or sparse arrays; each shape `(n_obs, n_var)`
- `obsm` — mapping; values dense arrays, sparse arrays, or dataframes; first dim `n_obs`
- `varm` — mapping; values dense arrays, sparse arrays, or dataframes; first dim `n_var`
- `obsp` — mapping; values dense or sparse arrays; first two dims `n_obs`
- `varp` — mapping; values dense or sparse arrays; first two dims `n_var`
- `uns` — mapping/dict-like container (recursive)

## Dense arrays

### `encoding-type: array`, `encoding-version: 0.2.0`

- A dense array **MUST** be an HDF5 **dataset**.
- The dataset **MUST** have attributes:
  - `encoding-type: "array"`
  - `encoding-version: "0.2.0"`

> **Legacy note**
>
> In anndata 0.7.x, dense arrays were typically stored as plain datasets *without* `encoding-type`/`encoding-version`.

## Sparse arrays (CSR/CSC)

### `encoding-type: csr_matrix|csc_matrix`, `encoding-version: 0.1.0`

A sparse matrix **MUST** be stored as an HDF5 **group**.

- Group attributes:
  - `encoding-type: "csr_matrix"` **or** `"csc_matrix"`
  - `encoding-version: "0.1.0"`
  - `shape`: integer array of length 2 (matrix shape)
- Group members (datasets):
  - `data`
  - `indices`
  - `indptr`

The exact CSR/CSC semantics follow SciPy’s conventions.

## DataFrames

DataFrames are stored column-wise: each column is stored as a dataset (or group, if the column itself is an encoded element).

<a id="dataframe-v020"></a>
### DataFrame v0.2.0

#### `encoding-type: dataframe`, `encoding-version: 0.2.0`

A dataframe **MUST** be stored as an HDF5 **group**.

- Group attributes:
  - `_index`: string — the key of the dataset to be used as the row index
  - `column-order`: array of strings — original column order
  - `encoding-type: "dataframe"`
  - `encoding-version: "0.2.0"`
- Group members:
  - the index dataset (named by `_index`)
  - one member per column
- All column entries **MUST** have the same length in their first dimension.
- Columns **SHOULD** share chunking along the first dimension.

Columns are independently encoded:
- simple numeric/bool columns are commonly `encoding-type: array`
- categorical columns are commonly `encoding-type: categorical`

<a id="dataframe-v010-legacy-anndata-07x"></a>
### DataFrame v0.1.0 (legacy: anndata 0.7.x)

#### `encoding-type: dataframe`, `encoding-version: 0.1.0`

A legacy dataframe is stored as an HDF5 **group** where:

- Group attributes include:
  - `_index`
  - `column-order`
  - `encoding-type: "dataframe"`
  - `encoding-version: "0.1.0"`
- Each column is a dataset.
- Categorical columns are stored as **integer code datasets**, and their category labels are stored in a reserved subgroup named `__categories`.

**Reserved subgroup:**

- `__categories/<colname>` stores the array of category labels for column `<colname>`.

<a id="legacy-categorical-columns-series-level"></a>
### Legacy categorical columns (Series-level)

In v0.1.0 DataFrames, a categorical column dataset (e.g. `obs/cell_type`) can be identified by the presence of an attribute:

- `categories`: an **HDF5 object reference** pointing to the corresponding `__categories/<colname>` dataset.

## Mappings / dict

### `encoding-type: dict`, `encoding-version: 0.1.0`

- A mapping **MUST** be stored as an HDF5 **group**.
- Group attributes:
  - `encoding-type: "dict"`
  - `encoding-version: "0.1.0"`
- Each entry in the group is another element (recursively).

> **Legacy note**
>
> In anndata 0.7.x, groups used as mappings often had **no special attributes**.

## Scalars

### `encoding-version: 0.2.0`

Scalars are stored as **0-dimensional datasets**.

- Numeric scalars:
  - `encoding-type: "numeric-scalar"`
  - `encoding-version: "0.2.0"`
  - value is numeric (including boolean, ints, floats, complex)
- String scalars:
  - `encoding-type: "string"`
  - `encoding-version: "0.2.0"`
  - **HDF5 requirement:** variable-length UTF-8 string dtype

> **Legacy note**
>
> In anndata 0.7.x, scalar strings were commonly stored as `|O` datasets without `encoding-type`/`encoding-version`.

## Categorical arrays

### `encoding-type: categorical`, `encoding-version: 0.2.0`

Categorical arrays are stored as an HDF5 **group** with members:

- `codes`: integer dataset  
  - values are zero-based indices into `categories`
  - signed integer arrays **MAY** use `-1` to denote missing values
- `categories`: array of labels

Group attributes:

- `encoding-type: "categorical"`
- `encoding-version: "0.2.0"`
- `ordered`: boolean (whether the categories are ordered)

## String arrays

### `encoding-type: string-array`, `encoding-version: 0.2.0`

- String arrays **MUST** be stored as HDF5 datasets.
- Dataset attributes:
  - `encoding-type: "string-array"`
  - `encoding-version: "0.2.0"`
- **HDF5 requirement:** variable-length UTF-8 string dtype

## Nullable arrays

These encodings support Pandas nullable integer/boolean/string arrays by storing a `values` array plus a boolean `mask` array.

### `encoding-type: nullable-integer`, `encoding-version: 0.1.0`

- Stored as an HDF5 group with datasets:
  - `values` (integer)
  - `mask` (boolean)

### `encoding-type: nullable-boolean`, `encoding-version: 0.1.0`

- Stored as an HDF5 group with datasets:
  - `values` (boolean)
  - `mask` (boolean)
- `values` and `mask` **MUST** have the same shape.

### `encoding-type: nullable-string-array`, `encoding-version: 0.1.0`

- Stored as an HDF5 group with datasets:
  - `values` (string array)
  - `mask` (boolean)
- Group attributes:
  - `encoding-type: "nullable-string-array"`
  - `encoding-version: "0.1.0"`
  - optional `na-value`: `"NA"` or `"NaN"` (default `"NA"`)

<a id="missing-value-semantics"></a>
#### Missing value semantics

For elements supporting a `na-value` attribute:

- `"NA"`: comparisons propagate missingness (e.g. `"x" == NA` → `NA`)
- `"NaN"`: comparisons yield boolean results (e.g. `"x" == NaN` → `false`)

Readers should preserve semantics when the runtime model supports it.

## Awkward arrays (experimental)

### `encoding-type: awkward-array`, `encoding-version: 0.1.0`

Ragged arrays are stored by decomposing an Awkward Array into constituent buffers (via `ak.to_buffers`), then storing those buffers as datasets within a group.

Group attributes:

- `encoding-type: "awkward-array"`
- `encoding-version: "0.1.0"`
- `form`: string — serialized Awkward “form”
- `length`: integer — logical length

Group members: datasets for the buffers (often named like `nodeX-*`).

> **Experimental**
>
> This encoding is considered experimental in the anndata 0.9.x series and later.

## Sources

- AnnData “on-disk format” prose docs (modern, ≥0.8): https://anndata.readthedocs.io/en/stable/fileformat-prose.html
- AnnData 0.7.8 “on-disk format” prose docs (legacy): https://dokk.org/documentation/anndata/0.7.8/fileformat-prose/
