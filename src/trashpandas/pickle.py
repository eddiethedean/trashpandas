"""Pickle storage backend for TrashPandas DataFrames.

This module provides pickle-based storage for Pandas DataFrames with full data type preservation.
Pickle storage is the most efficient for preserving exact DataFrame structure and data types.

Key Features:
    - Store DataFrames as pickle files with full data type preservation
    - Optional compression support (gzip, bz2, xz, zstd)
    - Custom file extensions support
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
        >>> with tp.PickleStorage('./data') as storage:
        ...     storage['users'] = df
        ...     loaded_df = storage['users']
        ...     print(f"Stored {len(storage)} tables")
        
    With compression:
        >>> storage = tp.PickleStorage('./data', compression='bz2')
        >>> storage.store(df, 'users')  # Creates users.pickle.bz2
        
    With custom file extension:
        >>> storage = tp.PickleStorage('./data', file_extension='.pkl')
        >>> storage.store(df, 'users')  # Creates users.pkl
        
    Using pathlib.Path:
        >>> from pathlib import Path
        >>> storage = tp.PickleStorage(Path('./data'))
        >>> storage.store(df, 'users')
        
    Bulk operations:
        >>> dataframes = {'users': users_df, 'orders': orders_df}
        >>> storage.store_many(dataframes)
        >>> results = storage.load_many(['users', 'orders'])
        >>> storage.delete_many(['users', 'orders'])

Raises:
    FileNotFoundError: When trying to load a non-existent pickle file
    PermissionError: When unable to write to the specified directory
    ValidationError: When table names contain invalid characters
    CompressionError: When compression/decompression fails
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Iterator, Union, Optional, Dict

from pandas import DataFrame, read_pickle

from trashpandas.interfaces import IStorage, IFileStorage
from trashpandas.exceptions import TableNotFoundError


class PickleStorage(IFileStorage):
    def __init__(self, folder_path: Union[str, Path], file_extension: str = '.pickle', compression: Optional[str] = None) -> None:
        """Takes folder path where DataFrames are stored as pickle files.
        
        Args:
            folder_path: Path to directory for pickle storage
            file_extension: File extension for pickle files
            compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
        """
        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension
        self.compression = compression

    def __repr__(self) -> str:
        return f"PickleStorage(path='{self.path}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame pickle file."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from pickle file."""
        return self.load(key)

    def __delitem__(self, key: str) -> None:
        """Delete DataFrame pickle file."""
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
    
    def __enter__(self) -> PickleStorage:
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type: Optional[Exception], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Exit context manager."""
        pass

    def store(self, df: DataFrame, table_name: str, schema: Optional[str] = None) -> None:
        """Store DataFrame pickle file."""
        store_df_pickle(df, table_name, str(self.path), self.file_extension, self.compression)

    def load(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve DataFrame from pickle file."""
        return load_df_pickle(table_name, str(self.path), self.file_extension, self.compression)

    def delete(self, table_name: str, schema: Optional[str] = None) -> None:
        """Delete DataFrame pickle file."""
        delete_table_pickle(table_name, str(self.path), self.file_extension)

    def table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of stored table names."""
        return table_names_pickle(str(self.path))
    
    def store_many(self, dataframes: Dict[str, DataFrame], schema: Optional[str] = None) -> None:
        """Store multiple DataFrames as pickle files.
        
        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name (ignored for pickle)
        """
        for table_name, df in dataframes.items():
            store_df_pickle(df, table_name, str(self.path), self.file_extension, self.compression)
    
    def load_many(self, table_names: List[str], schema: Optional[str] = None) -> Dict[str, DataFrame]:
        """Load multiple DataFrames from pickle files.
        
        Args:
            table_names: List of table names to load
            schema: Optional schema name (ignored for pickle)
            
        Returns:
            Dictionary mapping table names to DataFrames
        """
        result = {}
        for table_name in table_names:
            result[table_name] = self.load(table_name, schema=schema)
        return result
    
    def delete_many(self, table_names: List[str], schema: Optional[str] = None) -> None:
        """Delete multiple pickle files.
        
        Args:
            table_names: List of table names to delete
            schema: Optional schema name (ignored for pickle)
        """
        for table_name in table_names:
            delete_table_pickle(table_name, str(self.path), self.file_extension)


def store_df_pickle(df: DataFrame, table_name: str, path: str, file_extension: str = '.pickle', compression: Optional[str] = None) -> None:
    """Store DataFrame as pickle file.
    
    Args:
        df: DataFrame to store
        table_name: Name of the table
        path: Directory path for storage
        file_extension: File extension for pickle files
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
    """
    pickle_path = _get_pickle_path(table_name, path, file_extension, compression)
    df.to_pickle(pickle_path, compression=compression)


def load_df_pickle(table_name: str, path: str, file_extension: str = '.pickle', compression: Optional[str] = None) -> DataFrame:
    """Retrieve DataFrame from pickle file.
    
    Args:
        table_name: Name of the table
        path: Directory path for storage
        file_extension: File extension for pickle files
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')
        
    Returns:
        Loaded DataFrame
    """
    pickle_path = _get_pickle_path(table_name, path, file_extension, compression)
    return read_pickle(pickle_path, compression=compression)


def delete_table_pickle(table_name: str, path: str, file_extension: str = '.pickle') -> None:
    """Delete DataFrame pickle file."""
    pickle_path = _get_pickle_path(table_name, path)
    os.remove(pickle_path)


def table_names_pickle(path: str, file_extension: str = '.pickle') -> List[str]:
    """Get list of stored table names."""
    filenames = os.listdir(path)
    return [filename.split(file_extension)[0] for filename in filenames
                if filename.endswith(file_extension) and '_metadata' not in filename]


def _get_pickle_path(table_name: str, path: str, file_extension: str = '.pickle', compression: Optional[str] = None) -> str:
    """Return joined folder path and pickle file name for DataFrame."""
    filename = f'{table_name}{file_extension}'
    if compression:
        filename += f'.{compression}'
    return os.path.join(path, filename)