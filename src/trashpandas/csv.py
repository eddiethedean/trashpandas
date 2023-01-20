"""
Store Pandas DataFrames as CSV files in a specified directory path.
Also saves the DataFrame metadata in CSV files.

When saved CSV is loaded back into a DataFrame,
the indexes and datatypes are converted back from saved metadata.

Example
-------
import pandas as pd
import trashpandas as tp

df = pd.DataFrame({'name': ['Joe', 'Bob', 'John'], 'age': [23, 34, 44]})

# Create CsvStorage object with current directory path.
storage = tp.CsvStorage('')

# Store DataFrame as csv named 'people.csv'
# and store metadata as csv named '_people_metadata.csv'
storage.store(df, 'people') 
# or assign DataFrame to item
storage['people'] = df

# Retrieve DataFrame from CsvStorage object.
df = storage.load('people')
# or use table name key
df = storage['people']

# Delete stored csv files using CsvStorage delete method.
storage.delete('people')
# or use del on table name key
del storage['people']

# Or use functions instead of CsvStorage class
tp.store_df_csv(df, 'people', '')

df = tp.load_df_csv('people', '')

tp.delete_table_csv('people', '')
"""

import os
from typing import List

from pandas import DataFrame, read_csv

from trashpandas.interfaces import IStorage
from trashpandas.utils import cast_type, convert_meta_to_dict, df_metadata, name_no_names, unname_no_names


class CsvStorage(IStorage):
    def __init__(self, folder_path: str) -> None:
        """Takes folder path where DataFrames and metadata are stored as csv files."""
        self.path = folder_path

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

    def store(self, df: DataFrame, table_name: str) -> None:
        """Store DataFrame and metadata as csv files."""
        store_df_csv(df, table_name, self.path)
    
    def load(self, table_name: str) -> DataFrame:
        """Retrieve DataFrame from csv file."""
        return load_df_csv(table_name, self.path)

    def delete(self, table_name: str) -> None:
        """Delete DataFrame and metadata csv files."""
        delete_table_csv(table_name, self.path)

    def load_metadata(self, table_name: str) -> DataFrame:
        """Retrieve DataFrame metadata from csv file."""
        return load_metadata_csv(table_name, self.path)

    def table_names(self) -> List[str]:
        """Get list of stored non-metadata table names."""
        return table_names_csv(self.path)

    def metdata_names(self) -> List[str]:
        """Get list of stored metadata table names."""
        return metadata_names_csv(self.path)


def store_df_csv(df: DataFrame, table_name: str, path: str) -> None:
    """Store DataFrame and metadata as csv files."""
    csv_path = _get_csv_path(table_name, path)
    metadata_path = _get_metadata_csv_path(table_name, path)
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df.to_csv(csv_path)
    metadata.to_csv(metadata_path, index=False)


def load_df_csv(table_name: str, path: str) -> DataFrame:
    """Retrieve DataFrame from csv file."""
    csv_path = _get_csv_path(table_name, path)
    metadata_path = _get_metadata_csv_path(table_name, path)

    if not os.path.exists(metadata_path):
        return _first_load_df_csv(table_name, path)

    metadata = _read_cast_metadata_csv(table_name, path)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata['column'][metadata['index']==True])
    df = read_csv(csv_path).astype(types).set_index(indexes)
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


def _get_csv_path(table_name: str, path: str) -> str:
    """Return joined folder path and csv file name for DataFrame."""
    return os.path.join(path, f'{table_name}.csv')


def _get_metadata_csv_path(table_name: str, path: str) -> str:
    """Return joined folder path and csv file name for DataFrame metadata."""
    return os.path.join(path, f'_{table_name}_metadata.csv')


def _read_cast_metadata_csv(table_name: str, path: str) -> DataFrame:
    """Load metadata csv and cast column datatypes column."""
    metadata_path = _get_metadata_csv_path(table_name, path)
    meta = read_csv(metadata_path)
    meta['datatype'] = cast_type(meta['datatype'])
    return meta


def _first_load_df_csv(table_name: str, path: str) -> DataFrame:
    """Load a csv that has no metadata stored, create and store metadata"""
    csv_path = _get_csv_path(table_name, path)
    df = read_csv(csv_path)
    store_df_csv(df, table_name, path)
    return df