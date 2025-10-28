# TrashPandas Examples

This directory contains example Jupyter notebooks demonstrating various features of TrashPandas.

## Available Notebooks

### 1. Basic Usage (`01_basic_usage.ipynb`)
An introduction to TrashPandas covering:
- CSV storage
- SQL storage
- Pickle storage
- Basic operations and dictionary-like interface

### 2. Advanced Features (`02_advanced_features.ipynb`)
Demonstrates advanced features:
- Compression support
- Bulk operations (store_many, load_many, delete_many)
- Data type preservation

### 3. Format Conversion (`03_format_conversion.ipynb`)
Shows how to convert DataFrames between different storage formats:
- Converting from CSV to SQL
- Batch conversion operations

### 4. Query Capabilities (`04_query_capabilities.ipynb`)
Advanced querying with SQL storage:
- Basic queries
- Filtering with WHERE clauses
- Column selection
- Result limiting

## Running the Notebooks

### Prerequisites

Install TrashPandas in development mode:

```bash
pip install -e .
```

### Starting Jupyter

```bash
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

### Executing Notebooks

1. Open any notebook from the examples directory
2. Run all cells sequentially
3. Each notebook is self-contained and will create data files in the current directory

## Note on Data Files

The notebooks create temporary data files (SQL databases, CSV directories, etc.) in the current directory. These can be safely deleted after running the notebooks.

## Creating Additional Notebooks

To create additional example notebooks, modify `create_notebooks.py` and run:

```bash
python examples/create_notebooks.py
```

