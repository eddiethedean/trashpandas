"""CSV storage backend for TrashPandas DataFrames.

This module provides CSV-based storage for Pandas DataFrames with metadata preservation.
It supports optional compression and maintains DataFrame structure including indexes and data types.

Key Features:
    - Store DataFrames as CSV files with metadata preservation
    - Optional compression support (gzip, bz2, xz, zstd)
    - Context manager support for automatic resource cleanup
    - Iterator protocol for easy table enumeration
    - Bulk operations for efficient batch processing
    - Pathlib.Path support for modern path handling

Examples:
    Basic usage:
        >>> import pandas as pd
        >>> import trashpandas as tp
        >>> 
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> 
        >>> # Using context manager (recommended)
        >>> with tp.CsvStorage('./data') as storage:
        ...     storage['users'] = df
        ...     loaded_df = storage['users']
        ...     print(f"Stored {len(storage)} tables")
        
    With compression:
        >>> storage = tp.CsvStorage('./data', compression='gzip')
        >>> storage.store(df, 'users')  # Creates users.csv.gz
        
    Using pathlib.Path:
        >>> from pathlib import Path
        >>> storage = tp.CsvStorage(Path('./data'))
        >>> storage.store(df, 'users')
        
    Bulk operations:
        >>> dataframes = {'users': users_df, 'orders': orders_df}
        >>> storage.store_many(dataframes)
        >>> results = storage.load_many(['users', 'orders'])
        >>> storage.delete_many(['users', 'orders'])

Raises:
    FileNotFoundError: When trying to load a non-existent CSV file
    PermissionError: When unable to write to the specified directory
    ValidationError: When table names contain invalid characters
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Iterator, Union, Optional, Dict

from pandas import DataFrame, read_csv

from trashpandas.interfaces import IStorage, IFileStorage
from trashpandas.utils import cast_type, convert_meta_to_dict, df_metadata, name_no_names, unname_no_names
from trashpandas.exceptions import TableNotFoundError


class CsvStorage(IFileStorage):
    def __init__(self, folder_path: Union[str, Path], compression: Optional[str] = None) -> None:
        """Takes folder path where DataFrames and metadata are stored as csv files.
        
        Args:
            folder_path: Path to directory for CSV storage
            compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
        """
        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def __repr__(self) -> str:
        return f"CsvStorage(path='{self.path}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame and metadata as csv files."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from csv file."""
        return self.load(key)

    def __delitem__(self, key: str) -> None:
        """Delete DataFrame and metadata csv files."""
        self.delete(key)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over table names."""
        return iter(self.table_names())
    
    def __len__(self) -> int:
        """Get number of stored tables."""
        return len(self.table_names())
    
    def __contains__(self, key: str) -> bool:
        """Check if table exists."""
        return key in self.table_names()
    
    def __enter__(self) -> CsvStorage:
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type: Optional[Exception], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Exit context manager."""
        pass

    def store(self, df: DataFrame, table_name: str, schema: Optional[str] = None) -> None:
        """Store DataFrame and metadata as csv files."""
        store_df_csv(df, table_name, str(self.path), self.compression)
    
    def load(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve DataFrame from csv file."""
        return load_df_csv(table_name, str(self.path), self.compression)

    def delete(self, table_name: str, schema: Optional[str] = None) -> None:
        """Delete DataFrame and metadata csv files."""
        delete_table_csv(table_name, str(self.path))

    def load_metadata(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve DataFrame metadata from csv file."""
        return load_metadata_csv(table_name, str(self.path))

    def table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of stored non-metadata table names."""
        return table_names_csv(str(self.path))

    def metadata_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of stored metadata table names."""
        return metadata_names_csv(str(self.path))
    
    def store_many(self, dataframes: Dict[str, DataFrame], schema: Optional[str] = None) -> None:
        """Store multiple DataFrames as CSV files.
        
        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name (ignored for CSV)
        """
        for table_name, df in dataframes.items():
            store_df_csv(df, table_name, str(self.path), self.compression)
    
    def load_many(self, table_names: List[str], schema: Optional[str] = None) -> Dict[str, DataFrame]:
        """Load multiple DataFrames from CSV files.
        
        Args:
            table_names: List of table names to load
            schema: Optional schema name (ignored for CSV)
            
        Returns:
            Dictionary mapping table names to DataFrames
        """
        result = {}
        for table_name in table_names:
            result[table_name] = self.load(table_name, schema=schema)
        return result
    
    def delete_many(self, table_names: List[str], schema: Optional[str] = None) -> None:
        """Delete multiple CSV files.
        
        Args:
            table_names: List of table names to delete
            schema: Optional schema name (ignored for CSV)
        """
        for table_name in table_names:
            delete_table_csv(table_name, str(self.path))


def store_df_csv(df: DataFrame, table_name: str, path: str, compression: Optional[str] = None) -> None:
    """Store DataFrame and metadata as csv files.
    
    Args:
        df: DataFrame to store
        table_name: Name of the table
        path: Directory path for storage
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
    """
    csv_path = _get_csv_path(table_name, path, compression)
    metadata_path = _get_metadata_csv_path(table_name, path, compression)
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df.to_csv(csv_path, compression=compression)
    metadata.to_csv(metadata_path, index=False, compression=compression)


def load_df_csv(table_name: str, path: str, compression: Optional[str] = None) -> DataFrame:
    """Retrieve DataFrame from csv file.
    
    Args:
        table_name: Name of the table
        path: Directory path for storage
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
        
    Returns:
        Loaded DataFrame
    """
    csv_path = _get_csv_path(table_name, path, compression)
    metadata_path = _get_metadata_csv_path(table_name, path, compression)

    if not os.path.exists(metadata_path):
        return _first_load_df_csv(table_name, path, compression)

    metadata = _read_cast_metadata_csv(table_name, path, compression)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata['column'][metadata['index']==True])
    df = read_csv(csv_path, compression=compression).astype(types).set_index(indexes)
    unname_no_names(df)
    return df


def delete_table_csv(table_name: str, path: str) -> None:
    """Delete DataFrame and metadata csv files."""
    csv_path = _get_csv_path(table_name, path)
    metadata_path = _get_metadata_csv_path(table_name, path)
    os.remove(csv_path)
    os.remove(metadata_path)


def load_metadata_csv(table_name: str, path: str) -> DataFrame:
    meta_name = f'_{table_name}_metadata'
    return _read_cast_metadata_csv(meta_name, path)


def table_names_csv(path: str) -> List[str]:
    """Get list of stored non-metadata table names."""
    filenames = os.listdir(path)
    return [filename.split('.csv')[0] for filename in filenames
                if filename.endswith('.csv') and '_metadata' not in filename]


def metadata_names_csv(path: str) -> List[str]:
    """Get list of stored metadata table names."""
    filenames = os.listdir(path)
    return [filename.split('.csv')[0] for filename in filenames
                if filename.endswith('.csv') and '_metadata' in filename]


def _get_csv_path(table_name: str, path: str, compression: Optional[str] = None) -> str:
    """Return joined folder path and csv file name for DataFrame."""
    filename = f'{table_name}.csv'
    if compression:
        filename += f'.{compression}'
    return os.path.join(path, filename)


def _get_metadata_csv_path(table_name: str, path: str, compression: Optional[str] = None) -> str:
    """Return joined folder path and csv file name for DataFrame metadata."""
    filename = f'_{table_name}_metadata.csv'
    if compression:
        filename += f'.{compression}'
    return os.path.join(path, filename)


def _read_cast_metadata_csv(table_name: str, path: str, compression: Optional[str] = None) -> DataFrame:
    """Load metadata csv and cast column datatypes column."""
    metadata_path = _get_metadata_csv_path(table_name, path, compression)
    meta = read_csv(metadata_path, compression=compression)
    meta['datatype'] = cast_type(meta['datatype'])
    return meta


def _first_load_df_csv(table_name: str, path: str, compression: Optional[str] = None) -> DataFrame:
    """Load a csv that has no metadata stored, create and store metadata"""
    csv_path = _get_csv_path(table_name, path, compression)
    df = read_csv(csv_path, compression=compression)
    store_df_csv(df, table_name, path, compression)
    return df