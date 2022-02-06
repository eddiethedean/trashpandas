import unittest
import os

from pandas import DataFrame, read_sql_table
from sqlalchemy import create_engine, inspect

from trashpandas.sql import SqlStorage
from trashpandas.sql import store_df_sql, load_df_sql, delete_table_sql, table_names_sql

from setup_test import create_df
from setup_test import create_named_index_df
from setup_test import create_two_unnamed_index_df
from setup_test import create_two_index_one_named_df
from setup_test import create_df_string
from setup_test import delete_all_files


path = os.getcwd() + '/tests/data'


class TestSqlStorage(unittest.TestCase):
    def test_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        storage = SqlStorage(engine)
        df1 = create_df()
        storage.store(df1, 'people')
        df2 = storage.load('people')
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_delete(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        storage = SqlStorage(engine)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.delete('people')
        table_names = inspect(engine).get_table_names()
        self.assertFalse('people' in table_names)
        self.assertFalse('_people_metadata' in table_names)
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        storage = SqlStorage(engine)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.store(df1, 'peoples')
        names = set(storage.table_names())
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)


class TestSql(unittest.TestCase):
    def test_basic_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_df()
        store_df_sql(df1, 'people', engine)
        df2 = load_df_sql('people', engine)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_named_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_named_index_df()
        store_df_sql(df1, 'people', engine)
        df2 = load_df_sql('people', engine)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_two_unnamed_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_two_unnamed_index_df()
        store_df_sql(df1, 'people', engine)
        df2 = load_df_sql('people', engine)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_one_named_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_two_index_one_named_df()
        store_df_sql(df1, 'people', engine)
        df2 = load_df_sql('people', engine)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_string_store_load(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_df_string()
        store_df_sql(df1, 'people', engine)
        df2 = load_df_sql('people', engine)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_delete_table(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_df()
        store_df_sql(df1, 'people', engine)
        delete_table_sql('people', engine)
        table_names = inspect(engine).get_table_names()
        self.assertFalse('people' in table_names)
        self.assertFalse('_people_metadata' in table_names)
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        engine = create_engine(f'sqlite:///{path}/test.db')
        df1 = create_df()
        store_df_sql(df1, 'people', engine)
        store_df_sql(df1, 'peoples', engine)
        names = set(table_names_sql(engine))
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)