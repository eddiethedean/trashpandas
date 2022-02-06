import os

from pandas import DataFrame


def create_df() -> DataFrame:
    return DataFrame({'id': [1, 2, 3],
                      'name': ['Joe', 'Bob', 'John'],
                      'age': [23, 34, 44]})


def create_named_index_df() -> DataFrame:
    df = create_df()
    df.index.name = 'x'
    return df


def create_two_named_index_df() -> DataFrame:
    df = create_df()
    return df.set_index(['id', 'age'])


def create_two_unnamed_index_df() -> DataFrame:
    df = create_two_named_index_df()
    df.index.names = [None, None]
    return df


def create_two_index_one_named_df() -> DataFrame:
    df = create_two_named_index_df()
    df.index.names = ['id', None]
    return df


def create_df_string() -> DataFrame:
    df = create_df()
    df['name'] = df['name'].astype('string')
    return df


def delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
