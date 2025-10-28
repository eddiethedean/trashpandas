![TrashPandas Logo](https://raw.githubusercontent.com/eddiethedean/trashpandas/main/docs/trashpanda.svg)

# TrashPandas: Persistent Pandas DataFrame Storage and Retrieval

[![PyPI Latest Release](https://img.shields.io/pypi/v/trashpandas.svg)](https://pypi.org/project/trashpandas/)
[![Tests](https://github.com/eddiethedean/trashpandas/actions/workflows/tests.yml/badge.svg)](https://github.com/eddiethedean/trashpandas/actions/workflows/tests.yml)
[![Python Support](https://img.shields.io/pypi/pyversions/trashpandas.svg)](https://pypi.org/project/trashpandas/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What is it?

**TrashPandas** is a modern Python package that provides persistent Pandas DataFrame storage and retrieval using SQL databases, CSV files, HDF5, or pickle files. Version 1.0.2 brings significant improvements including SQLAlchemy 2.x support, comprehensive type hints, modern Python features, enhanced error handling, and improved CI/CD reliability.

## ✨ Main Features

- **Multiple Storage Backends**: SQL databases, CSV files, HDF5, and pickle files
- **Preserve Data Integrity**: Maintains indexes and data types during storage/retrieval
- **Format Conversion**: Transfer DataFrames between different storage formats
- **Modern Python Support**: Full type hints, context managers, and iterator protocol
- **Bulk Operations**: Efficient batch processing with `store_many()`, `load_many()`, `delete_many()`
- **Compression Support**: Optional compression for CSV and pickle storage
- **Comprehensive Error Handling**: Custom exception hierarchy with detailed error messages
- **Schema Validation**: Robust validation for SQL schema names and metadata
- **Cross-Platform Compatibility**: Tested on multiple Python versions (3.8-3.12) and operating systems
- **Comprehensive Testing**: 252+ tests with 76% code coverage

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install trashpandas

# With HDF5 support
pip install trashpandas[hdf5]

# Development dependencies
pip install trashpandas[dev]
```

### Basic Usage

```python
import pandas as pd
import sqlalchemy as sa
import trashpandas as tp

# Create sample data
df = pd.DataFrame({'name': ['Joe', 'Bob', 'John'], 'age': [23, 34, 44]})

# SQL Storage
with tp.SqlStorage('sqlite:///test.db') as storage:
    storage['people'] = df
    loaded_df = storage['people']
    print(f"Stored {len(storage)} tables")

# CSV Storage with compression
csv_storage = tp.CsvStorage('./data', compression='gzip')
csv_storage.store(df, 'people')

# Pickle Storage
pickle_storage = tp.PickleStorage('./pickles', compression='bz2')
pickle_storage.store(df, 'people')
```

## 📖 Example Notebooks

Check out these interactive Jupyter notebooks demonstrating TrashPandas features:

- **[Basic Usage](https://github.com/eddiethedean/trashpandas/blob/main/examples/01_basic_usage.ipynb)** - Introduction to CSV, SQL, and Pickle storage
- **[Advanced Features](https://github.com/eddiethedean/trashpandas/blob/main/examples/02_advanced_features.ipynb)** - Compression, bulk operations, and data type preservation
- **[Format Conversion](https://github.com/eddiethedean/trashpandas/blob/main/examples/03_format_conversion.ipynb)** - Converting DataFrames between different storage formats
- **[Query Capabilities](https://github.com/eddiethedean/trashpandas/blob/main/examples/04_query_capabilities.ipynb)** - Advanced SQL querying with WHERE clauses and filtering

All notebooks are fully executed with outputs included. Click the links above to view them on GitHub or open them in Jupyter Notebook/Lab.

## 📚 API Reference

### Storage Classes

#### SqlStorage
```python
# Create SQL storage
storage = tp.SqlStorage('sqlite:///test.db')
# or with existing engine
engine = sa.create_engine('sqlite:///test.db')
storage = tp.SqlStorage(engine)

# Basic operations
storage.store(df, 'table_name')
df = storage.load('table_name')
storage.delete('table_name')

# Dictionary-like interface
storage['table_name'] = df
df = storage['table_name']
del storage['table_name']

# Bulk operations
storage.store_many({'table1': df1, 'table2': df2})
results = storage.load_many(['table1', 'table2'])
storage.delete_many(['table1', 'table2'])

# Context manager
with storage:
    storage['data'] = df
```

#### CsvStorage
```python
# Basic CSV storage
storage = tp.CsvStorage('./data')

# With compression
storage = tp.CsvStorage('./data', compression='gzip')

# Operations
storage.store(df, 'table_name')
df = storage.load('table_name')
```

#### PickleStorage
```python
# Basic pickle storage
storage = tp.PickleStorage('./pickles')

# With custom extension and compression
storage = tp.PickleStorage('./pickles', file_extension='.pkl', compression='bz2')

# Operations
storage.store(df, 'table_name')
df = storage.load('table_name')
```

#### HdfStorage (Optional)
```python
# Requires: pip install trashpandas[hdf5]
storage = tp.HdfStorage('data.h5')
storage.store(df, 'table_name')
df = storage.load('table_name')
```

### Modern Features

#### Iterator Protocol
```python
storage = tp.SqlStorage('sqlite:///test.db')

# Iterate over table names
for table_name in storage:
    print(f"Table: {table_name}")

# Check if table exists
if 'my_table' in storage:
    df = storage['my_table']

# Get number of tables
print(f"Total tables: {len(storage)}")
```

#### Context Managers
```python
# Automatic resource cleanup
with tp.SqlStorage('sqlite:///test.db') as storage:
    storage['data'] = df
    # Connection automatically closed
```

#### Bulk Operations
```python
# Store multiple DataFrames efficiently
dataframes = {
    'users': users_df,
    'orders': orders_df,
    'products': products_df
}
storage.store_many(dataframes)

# Load multiple tables
tables = ['users', 'orders', 'products']
results = storage.load_many(tables)

# Delete multiple tables
storage.delete_many(tables)
```

#### Compression Support
```python
# CSV with compression
csv_storage = tp.CsvStorage('./data', compression='gzip')

# Pickle with compression
pickle_storage = tp.PickleStorage('./pickles', compression='bz2')

# Supported compression types: 'gzip', 'bz2', 'xz', 'zstd'
```

### Error Handling

```python
from trashpandas.exceptions import TableNotFoundError, MetadataCorruptedError

try:
    df = storage.load('nonexistent_table')
except TableNotFoundError as e:
    print(f"Table not found: {e.table_name}")
except MetadataCorruptedError as e:
    print(f"Metadata corrupted: {e.details}")
```

## 🔄 Migration from 0.x to 1.0.2

### Breaking Changes

1. **SQLAlchemy 2.x Required**: Update your SQLAlchemy version
   ```bash
   pip install "SQLAlchemy>=2.0.0"
   ```

2. **Path Parameters**: Storage classes now accept `pathlib.Path` objects
   ```python
   # Old
   storage = tp.CsvStorage('/path/to/data')
   
   # New (still works)
   storage = tp.CsvStorage('/path/to/data')
   
   # New (recommended)
   from pathlib import Path
   storage = tp.CsvStorage(Path('/path/to/data'))
   ```

3. **Method Signatures**: Some internal methods have updated signatures
   ```python
   # Old
   storage.store(df, 'table')
   
   # New (backward compatible)
   storage.store(df, 'table')
   storage.store(df, 'table', schema='my_schema')  # New optional parameter
   ```

### New Features in 1.0.2

1. **Enhanced Schema Validation**: Improved validation for SQL schema names and metadata
2. **Better Error Handling**: More specific exception types and detailed error messages
3. **Improved Compatibility**: Fixed numpy/PyTables compatibility issues across Python versions
4. **Robust CI/CD**: Comprehensive testing across Python 3.8-3.12 with reliable builds
5. **Context Managers**: Use `with` statements for automatic cleanup
6. **Iterator Protocol**: Iterate over storage objects
7. **Bulk Operations**: Efficient batch processing
8. **Compression**: Optional compression for file-based storage

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/eddiethedean/trashpandas.git
cd trashpandas
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with tox (recommended)
tox

# Run specific Python version
tox -e py311
tox -e py312

# Run linting
tox -e lint

# Run type checking
tox -e mypy

# Run with coverage
tox -e coverage

# Or run pytest directly
pytest
pytest --cov=trashpandas
pytest tests/test_sql.py
```

### Code Quality

```bash
# Linting with ruff
ruff check src tests

# Type checking with mypy
mypy src

# Format code
ruff format src tests
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- SQLAlchemy >= 2.0.0
- h5py >= 3.10.0 (optional, for HDF5 support)
- tables >= 3.8.0 (optional, for HDF5 support)
- numpy >= 1.21.0, < 2.0.0 (for PyTables compatibility)

## 📝 Recent Changes (v1.0.2)

### 🐛 Bug Fixes
- Fixed schema validation in `load_metadata_sql` function
- Resolved numpy/PyTables compatibility issues across Python versions
- Improved error handling for edge cases in metadata operations

### 🔧 Improvements
- Enhanced CI/CD pipeline with comprehensive testing
- Added proper dependency version constraints for stability
- Improved cross-platform compatibility testing
- Better error messages and exception handling

### 🧪 Testing
- All 252 tests passing with 76% code coverage
- Comprehensive testing across Python 3.8-3.12
- Robust CI/CD with reliable builds on GitHub Actions

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/eddiethedean/trashpandas/blob/main/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/eddiethedean/trashpandas/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [pandas](https://pandas.pydata.org/) for the excellent DataFrame library
- [SQLAlchemy](https://www.sqlalchemy.org/) for robust database connectivity
- [h5py](https://docs.h5py.org/) for HDF5 support
- The Python community for inspiration and feedback

---

**TrashPandas** - Making DataFrame persistence simple and reliable! 🐼