![TrashPandas Logo](https://raw.githubusercontent.com/eddiethedean/trashpandas/ef8b3b0b6394ac6be905e423447b0c238faba03d/docs/trashpanda.svg)
-----------------

# TrashPandas: Persistent Pandas DataFrame Storage and Retrieval
[![PyPI Latest Release](https://img.shields.io/pypi/v/trashpandas.svg)](https://pypi.org/project/trashpandas/)

## What is it?

**TrashPandas** is a Python package that provides persistent Pandas DataFrame storage and retrieval using a SQL database, CSV files, HDF5, or pickle files.

## Main Features
Here are just a few of the things that TrashPandas does well:

  - Store Pandas DataFrames in your choice of format. (SQL, CSV, HDF5, pickle)
  - Retrieve the Pandas DataFrame in the same format you stored.
  - Transfer your DataFrames between storage formats.

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/eddiethedean/trashpandas

```sh
# PyPI
pip install trashpandas
```

## Dependencies
- [pandas - a Python package that provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive.](https://pandas.pydata.org/)
- [sqlalchemy - Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL](https://www.sqlalchemy.org/)
- [h5py - a Python package with a Pythonic interface to the HDF5 binary data format.](https://docs.h5py.org/)]



## Example
```sh
import trashpandas as ts
```