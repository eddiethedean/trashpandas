"""
Store Pandas DataFrames as pickle files in a specified directory path.
Also retriave Pandas DataFrames from pickle files.

Example
-------
import pandas as pd
import trashpandas as tp

df = pd.DataFrame({'name': ['Joe', 'Bob', 'John'], 'age': [23, 34, 44]})

# Create PickleStorage object with current directory path.
storage = tp.PickleStorage('')

# Store DataFrame as pickle file named 'people.pickle'
# and store metadata as pickle file named '_people_metadata.pickle'
storage.store(df, 'people') 
# or assign DataFrame to item
storage['people'] = df

# Retrieve DataFrame using PickleStorage load method.
df = storage.load('people')
# or use table name key
df = storage['people']

# Delete stored pickle files using PickleStorage delete method.
storage.delete('people')
# or use del on table name key
del storage['people']

# Or use functions instead of PickleStorage class
tp.store_df_pickle(df, 'people', '')

df = tp.load_df_pickle('people', '')

tp.delete_table_pickle('people', '')
"""

import os
from typing import List

from pandas import DataFrame, read_pickle

from trashpandas.interfaces import IStorage


class PickleStorage(IStorage):
    def __init__(self, folder_path: str, file_extension: str = '.pickle') -> None:
        """Takes folder path where DataFrames are stored as pickle files."""
        self.path = folder_path
        self.file_extension = file_extension

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

    def store(self, df: DataFrame, table_name: str) -> None:
        """Store DataFrame pickle file."""
        store_df_pickle(df, table_name, self.path, self.file_extension)

    def load(self, table_name: str) -> DataFrame:
        """Retrieve DataFrame from pickle file."""
        return load_df_pickle(table_name, self.path, self.file_extension)

    def delete(self, table_name: str) -> None:
        """Delete DataFrame pickle file."""
        delete_table_pickle(table_name, self.path, self.file_extension)

    def table_names(self) -> List[str]:
        """Get list of stored table names."""
        return table_names_pickle(self.path)


def store_df_pickle(df: DataFrame, table_name: str, path: str, file_extension: str = '.pickle') -> None:
    """Store DataFrame as pickle file."""
    pickle_path = _get_pickle_path(table_name, path, file_extension)
    df.to_pickle(pickle_path)


def load_df_pickle(table_name: str, path: str, file_extension: str = '.pickle') -> DataFrame:
    """Retrieve DataFrame from pickle file."""
    pickle_path = _get_pickle_path(table_name, path)
    return read_pickle(pickle_path)


def delete_table_pickle(table_name: str, path: str, file_extension: str = '.pickle') -> None:
    """Delete DataFrame pickle file."""
    pickle_path = _get_pickle_path(table_name, path)
    os.remove(pickle_path)


def table_names_pickle(path: str, file_extension: str = '.pickle') -> List[str]:
    """Get list of stored table names."""
    filenames = os.listdir(path)
    return [filename.split(file_extension)[0] for filename in filenames
                if filename.endswith(file_extension) and '_metadata' not in filename]


def _get_pickle_path(table_name: str, path: str, file_extension: str = '.pickle') -> str:
    """Return joined folder path and pickle file name for DataFrame."""
    return os.path.join(path, f'{table_name}{file_extension}')