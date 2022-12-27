from pandas import DataFrame
from pandas.core.dtypes.common import pandas_dtype


def df_metadata(df: DataFrame) -> DataFrame:
    columns = DataFrame({'column': df.columns,
                            'index': False,
                            'datatype': [str(dt) for dt in df.dtypes]})
    indexes = DataFrame({'column': [f'_no_name_{i}' if name==None else name for i, name in enumerate(df.index.names)],
                            'index': True,
                            'datatype': str(df.index.dtype) if len(df.index.names)==1
                                            else [str(dt) for dt in df.index.dtypes]})  # type: ignore
    return indexes.append(columns).reset_index(drop=True)


def unname_no_names(df) -> None:
    if len(df.index.names)==1:
        if df.index.name == '_no_name':
            df.index.name = None
    else:
        df.index.names = [None if name==f'_no_name_{i}' else name for i, name in enumerate(df.index.names)]
        

def name_no_names(df) -> None:
    if len(df.index.names)==1:
        if df.index.name == None:
            df.index.name = '_no_name'
    else:
        df.index.names = [f'_no_name_{i}' if name==None else '_no_name_' for i, name in enumerate(df.index.names)]

        
def cast_type(series):
    return [pandas_dtype(n) for n in series]


def convert_meta_to_dict(meta: DataFrame) -> dict:
    return {col: typ for col, typ in zip(meta['column'], meta['datatype'])}


def df_cols_to_numpy(df: DataFrame) -> None:
    """Convert each column to numpy data type"""
    for name in df.columns:
        df[name] = df[name].to_numpy()