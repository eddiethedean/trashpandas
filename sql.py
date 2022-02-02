from pandas import DataFrame, read_sql_table
from sqlalchemy.engine import Engine

from interfaces import IStorage
from utils import cast_type, convert_meta_to_dict, df_metadata, name_no_names, unname_no_names


class SqlStorage(IStorage):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def __getitem__(self, key: str) -> DataFrame:
        return self.load(key)

    def __setitem__(self, key: str, other: DataFrame) -> None:
        self.store(other, key)
        
    def store(self, df: DataFrame, name: str, schema=None) -> None:
        store_df_sql(df, name, self.engine, schema=schema)
    
    def load(self, name: str, schema=None) -> DataFrame:
        return load_df_sql(name, self.engine, schema=schema)
    
    
def store_df_sql(df: DataFrame, table_name: str, engine: Engine, schema=None) -> None:
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df.to_sql(table_name, engine, if_exists='replace', schema=schema)
    metadata.to_sql(f'_{table_name}_metadata', engine, if_exists='replace', index=False, schema=schema)


def read_cast_metadata_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    meta = read_sql_table(table_name, engine, schema=schema)
    meta['datatype'] = cast_type(meta['datatype'])
    return meta


def load_df_sql(table_name: str, engine: Engine, schema=None) -> DataFrame:
    meta_name = f'_{table_name}_metadata'
    metadata = read_cast_metadata_sql(meta_name, engine, schema)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata['column'][metadata['index']==True])
    df = read_sql_table(table_name, engine, schema=schema).astype(types).set_index(indexes)
    unname_no_names(df)
    return df