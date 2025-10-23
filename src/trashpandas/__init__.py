"""TrashPandas: Persistent Pandas DataFrame Storage and Retrieval.

A Python package that provides persistent Pandas DataFrame storage and retrieval
using a SQL database, CSV files, HDF5, or pickle files.
"""

from __future__ import annotations

import importlib.util

__version__ = '1.0.0'
__author__ = 'Odos Matthews'
__email__ = 'odos@example.com'

# Core storage classes
from trashpandas.sql import SqlStorage
from trashpandas.csv import CsvStorage
from trashpandas.pickle import PickleStorage

# Async storage classes
from trashpandas.async_sql import AsyncSqlStorage
from trashpandas.async_file import AsyncCsvStorage, AsyncPickleStorage

# Function interfaces
from trashpandas.sql import store_df_sql, load_df_sql, delete_table_sql, table_names_sql
from trashpandas.csv import store_df_csv, load_df_csv, delete_table_csv, table_names_csv
from trashpandas.pickle import store_df_pickle, load_df_pickle, delete_table_pickle, table_names_pickle

# Optional HDF5 support
if importlib.util.find_spec('h5py'):
    from trashpandas.hdf5 import HdfStorage
    from trashpandas.hdf5 import store_df_hdf5, load_df_hdf5, delete_table_hdf5, table_names_hdf5

# Exception classes
from trashpandas.exceptions import (
    TrashPandasError,
    StorageError,
    TableNotFoundError,
    TableAlreadyExistsError,
    MetadataError,
    MetadataNotFoundError,
    MetadataCorruptedError,
    ConnectionError,
    ValidationError,
    ConversionError,
    CompressionError,
)

# Metadata handling
from trashpandas.metadata import TableMetadata

# Conversion utilities
from trashpandas.conversion import (
    convert_table_storage,
    convert_all_tables_storage,
    csv_to_sql,
    csv_to_sql_all,
    csv_to_hdf,
    csv_to_hdf_all,
    csv_to_pickle,
    csv_to_pickle_all,
    sql_to_csv,
    sql_to_csv_all,
    sql_to_hdf,
    sql_to_hdf_all,
    sql_to_pickle,
    sql_to_pickle_all,
    hdf_to_csv,
    hdf_to_csv_all,
    hdf_to_sql,
    hdf_to_sql_all,
    hdf_to_pickle,
    hdf_to_pickle_all,
    pickle_to_csv,
    pickle_to_csv_all,
    pickle_to_sql,
    pickle_to_sql_all,
    pickle_to_hdf,
    pickle_to_hdf_all,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Storage classes
    'SqlStorage',
    'CsvStorage', 
    'PickleStorage',
    
    # Async storage classes
    'AsyncSqlStorage',
    'AsyncCsvStorage',
    'AsyncPickleStorage',
    
    # Function interfaces
    'store_df_sql',
    'load_df_sql',
    'delete_table_sql',
    'table_names_sql',
    'store_df_csv',
    'load_df_csv',
    'delete_table_csv',
    'table_names_csv',
    'store_df_pickle',
    'load_df_pickle',
    'delete_table_pickle',
    'table_names_pickle',
    
    # Optional HDF5
    'HdfStorage',
    'store_df_hdf5',
    'load_df_hdf5',
    'delete_table_hdf5',
    'table_names_hdf5',
    
    # Exceptions
    'TrashPandasError',
    'StorageError',
    'TableNotFoundError',
    'TableAlreadyExistsError',
    'MetadataError',
    'MetadataNotFoundError',
    'MetadataCorruptedError',
    'ConnectionError',
    'ValidationError',
    'ConversionError',
    'CompressionError',
    
    # Metadata
    'TableMetadata',
    
    # Conversion utilities
    'convert_table_storage',
    'convert_all_tables_storage',
    'csv_to_sql',
    'csv_to_sql_all',
    'csv_to_hdf',
    'csv_to_hdf_all',
    'csv_to_pickle',
    'csv_to_pickle_all',
    'sql_to_csv',
    'sql_to_csv_all',
    'sql_to_hdf',
    'sql_to_hdf_all',
    'sql_to_pickle',
    'sql_to_pickle_all',
    'hdf_to_csv',
    'hdf_to_csv_all',
    'hdf_to_sql',
    'hdf_to_sql_all',
    'hdf_to_pickle',
    'hdf_to_pickle_all',
    'pickle_to_csv',
    'pickle_to_csv_all',
    'pickle_to_sql',
    'pickle_to_sql_all',
    'pickle_to_hdf',
    'pickle_to_hdf_all',
]