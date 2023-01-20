"""
Store Pandas DataFrames in database tables.
Also saves the DataFrame metadata in database tables.

When stored table is loaded back into a DataFrame,
the indexes and datatypes are converted back from saved metadata.

Example
-------
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
"""

from typing import List, Union, Optional

from pandas import DataFrame, read_sql_table
from sqlalchemy import inspect, Table, MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.schema import DropTable

from trashpandas.interfaces import IStorage
from trashpandas.utils import cast_type, convert_meta_to_dict, df_metadata, name_no_names, unname_no_names


class SqlStorage(IStorage):
    def __init__(self, engine: Union[Engine, str]) -> None:
        """Takes SQLAlchemy Engine or database connection string."""
        if isinstance(engine, str):
            engine = create_engine(engine)
        self.engine = engine

    def __repr__(self):
        return f"SqlStorage('{self.engine.url}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame and metadata in database."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from database."""
        return self.load(key)

    def __delitem__(self, key) -> None:
        """Delete DataFrame and metadata from database."""
        self.delete(key)
        
    def store(self, df: DataFrame, table_name: str, schema: Optional[str] = None) -> None:
        """Store DataFrame and metadata in database."""
        store_df_sql(df, table_name, self.engine, schema=schema)
    
    def load(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve DataFrame from database."""
        return load_df_sql(table_name, self.engine, schema=schema)

    def delete(self, table_name: str, schema: Optional[str] = None) -> None:
        """Delete DataFrame and metadata from database."""
        delete_table_sql(table_name, self.engine, schema)

    def load_metadata(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve DataFrame metadata from database."""
        return load_metadata_sql(table_name, self.engine, schema=schema)

    def table_names(self, schema: Optional[str] = None) -> List[str]:
        """Query database for list of non-metadata table names."""
        return table_names_sql(self.engine, schema=schema)

    def metadata_names(self, schema: Optional[str] = None) -> List[str]:
        """Query database for list of metadata table names."""
        return metadata_names_sql(self.engine, schema=schema)


def store_df_sql(df: DataFrame, table_name: str, engine: Engine, schema=None) -> None:
    """Store DataFrame and metadata in database."""
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df.to_sql(table_name, engine, if_exists='replace', schema=schema)
    metadata.to_sql(f'_{table_name}_metadata', engine, if_exists='replace', index=False, schema=schema)


def load_df_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    """Retrieve DataFrame from database."""
    meta_name = f'_{table_name}_metadata'

    if meta_name not in metadata_names_sql(engine, schema=schema):
        return _first_load_df_sql(table_name, engine, schema=schema)

    metadata = _read_cast_metadata_sql(meta_name, engine, schema)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata['column'][metadata['index']==True])
    df = read_sql_table(table_name, engine, schema=schema).astype(types).set_index(indexes)
    unname_no_names(df)
    return df


def delete_table_sql(table_name: str, engine: Engine, schema=None) -> None:
    """Delete DataFrame and metadata from database."""
    table = _get_table(table_name, engine, schema)
    metadata = _get_table(f'_{table_name}_metadata', engine, schema)
    engine.execute(DropTable(table))
    engine.execute(DropTable(metadata))


def load_metadata_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    """Retrieve DataFrame metadata from database."""
    meta_name = f'_{table_name}_metadata'
    return _read_cast_metadata_sql(meta_name, engine, schema=schema)


def table_names_sql(engine: Engine, schema=None) -> List[str]:
    """Query database for list of non-metadata table names."""
    table_names = inspect(engine).get_table_names(schema=schema)
    return [name for name in table_names if '_metadata' not in name and name[0]!='_']


def metadata_names_sql(engine: Engine, schema=None) -> List[str]:
    """Query database for list of metadata table names."""
    table_names = inspect(engine).get_table_names(schema=schema)
    return [name for name in table_names if '_metadata' in name and name[0]=='_']


def _read_cast_metadata_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    """Load metadata table and cast column datatypes column."""
    meta = read_sql_table(table_name, engine, schema=schema)
    meta['datatype'] = cast_type(meta['datatype'])
    return meta


def _first_load_df_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    """Load a sql table that has no metadata stored, create and store metadata"""
    df = read_sql_table(table_name, engine, schema=schema)
    store_df_sql(df, table_name, engine)
    return df


def _get_table(table_name: str, engine: Engine, schema=None) -> Table:
    """Get SQLAlchemy Table mapped to database table."""
    metadata = MetaData(bind=engine, schema=schema)
    return Table(table_name,
                 metadata,
                 autoload=True,
                 schema=schema)