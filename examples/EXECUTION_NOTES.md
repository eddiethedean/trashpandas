# Notebook Execution Notes

## Overview

All notebooks in this directory are ready to execute. They have been validated for syntax correctness.

## Running the Notebooks

### Option 1: Jupyter Notebook

```bash
jupyter notebook
```

Then open any notebook from the examples directory and click "Run All".

### Option 2: JupyterLab

```bash
jupyter lab
```

Then open any notebook and use "Run All Cells".

### Option 3: Via Command Line

You can execute notebooks programmatically using nbconvert:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/01_basic_usage.ipynb
```

However, be aware that nbconvert executes the notebooks in a separate Python process and may have issues with the interactive nature of some examples.

## Notebook Status

All 4 notebooks have been validated:

✓ **01_basic_usage.ipynb** - 5 code cells
- CSV storage with dictionary-like syntax
- SQL storage with query capabilities  
- Pickle storage for data type preservation

✓ **02_advanced_features.ipynb** - 4 code cells
- Compression support (gzip)
- Bulk operations
- Data type preservation

✓ **03_format_conversion.ipynb** - 3 code cells
- CSV to SQL conversion
- Batch conversion operations

✓ **04_query_capabilities.ipynb** - 5 code cells
- Basic queries
- WHERE clause filtering
- Column selection
- Result limiting

## Expected Outputs

When executed, the notebooks will:

1. Create data directories (data_csv, data_pickle, etc.)
2. Create SQLite databases (data_sql.db, converted.db, etc.)
3. Display DataFrame contents and verification messages
4. Demonstrate various features of TrashPandas

## Cleanup

After running the notebooks, you can clean up generated files:

```bash
rm -rf data_* *.db
```

This will remove all test data and database files created by the notebooks.

