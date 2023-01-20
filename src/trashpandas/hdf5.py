"""
Store Pandas DataFrames in a hdf5 file in a specified directory path.
Also saves the DataFrame metadata in the hdf5 file.

When saved table is loaded back into a DataFrame,
the indexes and datatypes are converted back from saved metadata.

Example
-------
import pandas as pd
import trashpandas as tp

df = pd.DataFrame({'name': ['Joe', 'Bob', 'John'], 'age': [23, 34, 44]})

# Create HdfStorage object with hdf5 file in current directory path.
# TrashPandas will create the hdf5 file if it doesn't exist.'
storage = tp.HdfStorage('data.h5')

# Store DataFrame in hdf5 table named 'people'
# and store metadata as hdf5 table named '_people_metadata'
storage.store(df, 'people') 
# or assign DataFrame to item
storage['people'] = df

# Retrieve DataFrame using HdfStorage load method.
df = storage.load('people')
# or use table name key
df = storage['people']

# Delete stored table using HdfStorage delete method.
storage.delete('people')
# or use del on table name key
del storage['people']

# Or use functions instead of HdfStorage class
tp.store_df_hdf5(df, 'people', 'data.h5')

df = tp.load_df_hdf5('people', 'data.h5')

tp.delete_table_hdf5('people', 'data.h5')
"""

import os
from typing import List

from h5py import File
from pandas import DataFrame, read_hdf

from trashpandas.interfaces import IStorage
from trashpandas.utils import cast_type, convert_meta_to_dict, df_metadata, name_no_names, unname_no_names
from trashpandas.utils import df_cols_to_numpy


class HdfStorage(IStorage):
    def __init__(self, hdf5_path: str) -> None:
        """Takes folder path to hdf5 file where DataFrames and metadata are stored."""
        self.path = hdf5_path
        # create hdf5 file if it doesn't exist
        create_hdf5_file(hdf5_path)

    def __repr__(self) -> str:
        return f"HdfStorage('{self.path}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame and metadata in hdf5 file."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from hdf5 file."""
        return self.load(key)

    def __delitem__(self, key: str) -> None:
        """Delete DataFrame and metadata from hdf5 file."""
        self.delete(key)

    def store(self, df: DataFrame, table_name: str) -> None:
        """Store DataFrame and metadata in hdf5 file."""
        store_df_hdf5(df, table_name, self.path)

    def load(self, table_name: str) -> DataFrame:
        """Retrieve DataFrame from hdf5 file."""
        return load_df_hdf5(table_name, self.path)

    def delete(self, table_name: str) -> None:
        """Delete DataFrame and metadata from hdf5 file."""
        delete_table_hdf5(table_name, self.path)

    def load_metadata(self, table_name: str, schema=None) -> DataFrame:
        """Retrieve DataFrame metadata from hdf5 file."""
        return load_metadata_hdf5(table_name, self.path)

    def table_names(self) -> List[str]:
        """Get list of stored non-metadata table names."""
        return table_names_hdf5(self.path)

    def metadata_names(self) -> List[str]:
        """Get list of stored metadata table names."""
        return metadata_names_hdf5(self.path)


def create_hdf5_file(path: str) -> None:
    """Create hdf5 file if it doesn't exist"""
    if not os.path.exists(path):
        hf = File(path, 'w')
        hf.close()


def store_df_hdf5(df: DataFrame, table_name: str, path: str) -> None:
    """Store DataFrame and metadata in hdf5 file."""
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df_cols_to_numpy(df)
    create_hdf5_file(path)
    df.to_hdf(path, key=table_name)
    metadata.to_hdf(path, key=f'_{table_name}_metadata', index=False)


def load_df_hdf5(table_name: str, path: str) -> DataFrame:
    """Retrieve DataFrame from hdf5 file."""
    meta_name = f'_{table_name}_metadata'

    if meta_name not in metadata_names_hdf5(path):
        return _first_load_df_hdf5(table_name, path)

    metadata = _read_cast_metadata_hdf5(meta_name, path)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata['column'][metadata['index']==True])
    df = read_hdf(path, key=table_name).reset_index().astype(types).set_index(indexes)
    unname_no_names(df)
    return df


def delete_table_hdf5(table_name: str, path: str) -> None:
    """Delete DataFrame and metadata from hdf5 file."""
    meta_name = f'_{table_name}_metadata'
    with File(path,  "a") as hf:
        del hf[table_name]
        del hf[meta_name]


def load_metadata_hdf5(table_name: str, path: str) -> DataFrame:
    """Retrieve DataFrame metadata from hdf5 file."""
    meta_name = f'_{table_name}_metadata'
    return _read_cast_metadata_hdf5(meta_name, path)


def table_names_hdf5(path: str) -> List[str]:
    """Get list of stored non-metadata table names."""
    table_names = _hdf5_keys(path)
    return [name for name in table_names
                if '_metadata' not in name and name[0]!='_']


def metadata_names_hdf5(path: str) -> List[str]:
    """Get list of stored metadata table names."""
    table_names = _hdf5_keys(path)
    return [name for name in table_names
                if '_metadata' in name and name[0]=='_']


def _hdf5_keys(path: str) -> List[str]:
    """Get list of all stored table names."""
    with File(path, 'r') as hf:
        names = list(hf.keys())
    return names


def _read_cast_metadata_hdf5(table_name: str, path: str) -> DataFrame:
    """Load metadata table and cast column datatypes column."""
    meta = read_hdf(path, key=table_name)
    meta['datatype'] = cast_type(meta['datatype'])
    return DataFrame(meta)


def _first_load_df_hdf5(table_name: str, path: str) -> DataFrame:
    """Load a sql table that has no metadata stored, create and store metadata"""
    # Load a table that has no metadata stored, create and store metadata
    df = DataFrame(read_hdf(path, key=table_name))
    store_df_hdf5(df, table_name, path)
    return df