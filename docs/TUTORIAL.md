# Tutorial: Using h5ad CLI with csvkit

This tutorial demonstrates how to combine `h5ad` CLI with `csvkit` to explore, analyze, and subset large `.h5ad` files efficiently without loading them into memory.

## Introduction

### h5ad CLI
A command-line tool for working with AnnData (`.h5ad`) files. It streams data directly from disk, making it perfect for exploring huge single-cell datasets without memory constraints.

**Key features:**
- `info` - Inspect file structure and dimensions
- `table` - Export metadata to CSV
- `subset` - Filter files by cell/gene names

### csvkit
A suite of command-line tools for working with CSV files. Think of it as `awk`, `sed`, and `grep` but specifically designed for CSV data.

**Key tools we'll use:**
- `csvcut` - Select specific columns
- `csvsql` - Execute SQL queries on CSV files
- `csvgrep` - Filter rows by pattern
- `csvlook` - Pretty-print CSV in terminal

**Installation:**
```bash
pip install csvkit
```

## 1. Inspect File Structure with `info`

First, let's see what's in our `.h5ad` file:

```bash
h5ad info dataset.h5ad
```

**Example output:**
```
File: dataset.h5ad
Dimensions: 50000 obs × 20000 var

Top-level groups:
  obs/
    - cell_type
    - sample_id
    - donor_id
    - tissue
    - n_genes
  var/
    - gene_name
    - highly_variable
  X (sparse matrix)
  layers/
  obsm/
  uns/
```

This shows us that we have 50,000 cells with metadata including cell types, samples, and donor information.

## 2. Export Metadata with `table`

### 2.1 Basic Metadata Export

Export all cell metadata (observations) to CSV:

```bash
h5ad table dataset.h5ad --axis obs --output cell_metadata.csv
```

Export just specific columns:

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type,sample_id,donor_id --output cells.csv
```

Preview the first few rows in a nice table format:

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type,sample_id,donor_id --head 10 | csvlook
```

**Example output:**
```
| obs_names           | cell_type    | sample_id | donor_id |
| ------------------- | ------------ | --------- | -------- |
| AAACCTGAGAAACCAT-1  | T cell       | sample_1  | donor_A  |
| AAACCTGAGACAGACC-1  | B cell       | sample_1  | donor_A  |
| AAACCTGAGGCATGGT-1  | NK cell      | sample_2  | donor_B  |
| AAACCTGCAAGCCGCT-1  | T cell       | sample_2  | donor_B  |
| AAACCTGCACATTAGC-1  | Monocyte     | sample_1  | donor_A  |
```

### 2.2 Calculate Statistics with `csvsql`

Now let's analyze the metadata using SQL queries!

**Count cells per cell type:**

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type | \
  csvsql --query "SELECT cell_type, COUNT(*) as n_cells FROM stdin GROUP BY cell_type ORDER BY n_cells DESC" | \
  csvlook
```

**Example output:**
```
| cell_type    | n_cells |
| ------------ | ------- |
| T cell       | 15234   |
| Monocyte     | 12456   |
| B cell       | 8932    |
| NK cell      | 5621    |
| DC           | 3456    |
| Macrophage   | 2301    |
```

**Count cells per cell type and sample:**

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type,sample_id | \
  csvsql --query "SELECT cell_type, sample_id, COUNT(*) as n_cells 
                  FROM stdin 
                  GROUP BY cell_type, sample_id 
                  ORDER BY cell_type, sample_id" | \
  csvlook
```

**Example output:**
```
| cell_type    | sample_id | n_cells |
| ------------ | --------- | ------- |
| B cell       | sample_1  | 4521    |
| B cell       | sample_2  | 4411    |
| Monocyte     | sample_1  | 6234    |
| Monocyte     | sample_2  | 6222    |
| T cell       | sample_1  | 7645    |
| T cell       | sample_2  | 7589    |
```

**Calculate average gene count per cell type:**

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type,n_genes | \
  csvsql --query "SELECT cell_type, 
                         AVG(n_genes) as avg_genes,
                         MIN(n_genes) as min_genes,
                         MAX(n_genes) as max_genes
                  FROM stdin 
                  GROUP BY cell_type 
                  ORDER BY avg_genes DESC" | \
  csvlook
```

**Find samples with low cell counts:**

```bash
h5ad table dataset.h5ad --axis obs --columns sample_id | \
  csvsql --query "SELECT sample_id, COUNT(*) as n_cells 
                  FROM stdin 
                  GROUP BY sample_id 
                  HAVING COUNT(*) < 1000 
                  ORDER BY n_cells" | \
  csvlook
```

## 3. Filter and Subset Data

### 3.1 Extract Cell Names for a Specific Cell Type

Let's say we want to create a subset containing only T cells.

**Step 1: Export metadata and filter for T cells**

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type --output cell_metadata.csv
```

**Step 2: Use csvgrep to find T cells and extract their names**

```bash
csvgrep -c cell_type -m "T cell" cell_metadata.csv | \
  csvcut -c obs_names | \
  tail -n +2 > tcell_names.txt
```

This creates a file `tcell_names.txt` with one cell barcode per line.

**Alternative: Use csvsql for more complex filters**

Get T cells from a specific donor:

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type,donor_id --output cell_metadata.csv

csvsql --query "SELECT obs_names 
                FROM cell_metadata 
                WHERE cell_type = 'T cell' 
                AND donor_id = 'donor_A'" \
       cell_metadata.csv | \
  tail -n +2 > tcell_donor_A.txt
```

Get cells with high gene counts (>2000 genes):

```bash
h5ad table dataset.h5ad --axis obs --columns n_genes --output cell_metadata.csv

csvsql --query "SELECT obs_names 
                FROM cell_metadata 
                WHERE n_genes > 2000" \
       cell_metadata.csv | \
  tail -n +2 > high_quality_cells.txt
```

### 3.2 Create the Subset

Now use the filtered cell list to create a new `.h5ad` file:

```bash
h5ad subset dataset.h5ad tcells_only.h5ad --obs tcell_names.txt
```

**Verify the subset:**

```bash
h5ad info tcells_only.h5ad
```

**Check the cell type distribution:**

```bash
h5ad table tcells_only.h5ad --axis obs --columns cell_type | \
  csvsql --query "SELECT cell_type, COUNT(*) as n_cells FROM stdin GROUP BY cell_type" | \
  csvlook
```

### 3.3 Advanced: Subset by Both Cells and Genes

Let's create a subset with specific cell types and a curated gene list.

**Step 1: Filter cells (multiple cell types)**

```bash
h5ad table dataset.h5ad --axis obs --columns cell_type --output cell_metadata.csv

csvsql --query "SELECT obs_names 
                FROM cell_metadata 
                WHERE cell_type IN ('T cell', 'NK cell', 'B cell')" \
       cell_metadata.csv | \
  tail -n +2 > lymphocytes.txt
```

**Step 2: Create a gene list**

You might have a predefined list or extract genes from the file:

```bash
# Export all genes
h5ad table dataset.h5ad --axis var --columns gene_name --output genes.csv

# Filter for specific genes (e.g., markers)
echo "CD3D
CD3E
CD4
CD8A
CD8B
CD19
CD20
NCAM1" > marker_genes.txt
```

**Step 3: Create the subset**

```bash
h5ad subset dataset.h5ad lymphocytes_markers.h5ad \
  --obs lymphocytes.txt \
  --var marker_genes.txt
```

**Verify:**

```bash
h5ad info lymphocytes_markers.h5ad
```

## 4. Complete Example Workflow

Here's a complete workflow combining everything:

```bash
# 1. Inspect the file
h5ad info large_dataset.h5ad

# 2. Export and analyze metadata
h5ad table large_dataset.h5ad --axis obs \
  --columns cell_type,sample_id,donor_id,n_genes \
  --output all_metadata.csv

# 3. Generate statistics
echo "Cell type distribution:"
csvsql --query "SELECT cell_type, COUNT(*) as n_cells 
                FROM all_metadata 
                GROUP BY cell_type 
                ORDER BY n_cells DESC" \
       all_metadata.csv | csvlook

echo "Sample distribution:"
csvsql --query "SELECT sample_id, donor_id, COUNT(*) as n_cells 
                FROM all_metadata 
                GROUP BY sample_id, donor_id" \
       all_metadata.csv | csvlook

# 4. Filter for high-quality T cells from a specific donor
csvsql --query "SELECT obs_names 
                FROM all_metadata 
                WHERE cell_type = 'T cell' 
                AND donor_id = 'donor_A' 
                AND n_genes > 1500" \
       all_metadata.csv | \
  tail -n +2 > selected_cells.txt

echo "Selected $(wc -l < selected_cells.txt) cells"

# 5. Create subset
h5ad subset large_dataset.h5ad tcells_subset.h5ad --obs selected_cells.txt

# 6. Verify result
h5ad info tcells_subset.h5ad
h5ad table tcells_subset.h5ad --axis obs --columns cell_type,donor_id | \
  csvsql --query "SELECT cell_type, donor_id, COUNT(*) as n_cells FROM stdin GROUP BY cell_type, donor_id" | \
  csvlook
```

## Tips and Best Practices

1. **Use `--head` for quick previews** before exporting large files:
   ```bash
   h5ad table data.h5ad --axis obs --head 100 | csvlook
   ```

2. **Pipe directly to csvkit** to avoid creating intermediate files:
   ```bash
   h5ad table data.h5ad --axis obs --columns cell_type | csvsql --query "..." 
   ```

3. **Check cell counts** before subsetting:
   ```bash
   wc -l selected_cells.txt  # Should be > 0!
   ```

4. **Use csvstat** for quick summary statistics:
   ```bash
   h5ad table data.h5ad --axis obs --columns n_genes,n_counts | csvstat
   ```

5. **Combine with standard Unix tools**:
   ```bash
   # Get unique cell types
   h5ad table data.h5ad --axis obs --columns cell_type | tail -n +2 | sort -u
   
   # Count samples
   h5ad table data.h5ad --axis obs --columns sample_id | tail -n +2 | sort | uniq -c
   ```

## Conclusion

By combining `h5ad` CLI with `csvkit`, you can:
- ✅ Explore huge datasets without loading them into memory
- ✅ Perform complex queries and aggregations on metadata
- ✅ Create filtered subsets based on sophisticated criteria
- ✅ Work entirely on the command line without Python/R

This workflow is especially powerful for:
- Initial data exploration
- Quality control analysis
- Creating test datasets
- Preparing data for downstream analysis
- Batch processing multiple files

For more information:
- h5ad CLI: [README.md](../README.md)
- csvkit documentation: https://csvkit.readthedocs.io/
