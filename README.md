![TrashPandas Logo](https://raw.githubusercontent.com/eddiethedean/trashpandas/main/docs/trashpanda.svg)
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
import pandas as pd
import sqlalchemy as sa
import trashpandas as tp

df = pd.DataFrame({'name': ['Joe', 'Bob', 'John'], 'age': [23, 34, 44]})

# Create SqlStorage object with sqlite database connection string.
storage = tp.SqlStorage('sqlite:///test.db')
# or create an engine with SQLAlchemy and pass it into SqlStorage
engine = sa.create_engine('sqlite:///test.db')
storage = tp.SqlStorage(engine)

# Store DataFrame in database as table named 'people'
# and store metadata as table named '_people_metadata'
storage.store(df, 'people') 
# or assign DataFrame to item
storage['people'] = df

# Retrieve DataFrame from SqlStorage object.
df = storage.load('people')
# or use table name key
df = storage['people']

# Delete stored sql table using SqlStorage delete method.
storage.delete('people')
# or use del on table name key
del storage['people']

# Or use functions instead of SqlStorage class
tp.store_df_sql(df, 'people', engine)

df = tp.load_df_sql('people', engine)

tp.delete_table_sql('people', engine)
```