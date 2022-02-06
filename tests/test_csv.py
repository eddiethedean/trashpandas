import unittest
import os

from pandas import DataFrame, read_csv

from trashpandas.csv import CsvStorage
from trashpandas.csv import store_df_csv, load_df_csv, delete_table_csv, table_names_csv

from setup_test import create_df
from setup_test import create_named_index_df
from setup_test import create_two_unnamed_index_df
from setup_test import create_two_index_one_named_df
from setup_test import create_df_string
from setup_test import delete_all_files


path = os.getcwd() + '/tests/data'


class TestCsvStorage(unittest.TestCase):
    def test_store_load(self):
        delete_all_files(path)
        storage = CsvStorage(path)
        df1 = create_df()
        storage.store(df1, 'people')
        df2 = storage.load('people')
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_delete(self):
        delete_all_files(path)
        storage = CsvStorage(path)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.delete('people')
        self.assertFalse(os.path.exists(os.path.join(path, 'people.csv'))) 
        self.assertFalse(os.path.exists(os.path.join(path, '_people_metadata.csv')))
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        storage = CsvStorage(path)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.store(df1, 'peoples')
        names = set(storage.table_names())
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)


class TestCsvFunctions(unittest.TestCase):
    def test_basic_store_load(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_csv(df1, 'people', path)
        df2 = load_df_csv('people', path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)
        self.assertFalse(os.path.exists(os.path.join(path, 'people.csv'))) 
        self.assertFalse(os.path.exists(os.path.join(path, '_people_metadata.csv')))
        delete_all_files(path)

    def test_named_store_load(self):
        delete_all_files(path)
        df1 = create_named_index_df()
        store_df_csv(df1, 'people', path)
        df2 = load_df_csv('people', path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_two_unnamed_store_load(self):
        delete_all_files(path)
        df1 = create_two_unnamed_index_df()
        store_df_csv(df1, 'people', path)
        df2 = load_df_csv('people', path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_one_named_store_load(self):
        delete_all_files(path)
        df1 = create_two_index_one_named_df()
        store_df_csv(df1, 'people', path)
        df2 = load_df_csv('people', path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_string_store_load(self):
        delete_all_files(path)
        df1 = create_df_string()
        store_df_csv(df1, 'people', path)
        df2 = load_df_csv('people', path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_delete_table(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_csv(df1, 'people', path)
        delete_table_csv('people', path)
        self.assertFalse(os.path.exists(os.path.join(path, 'people.csv'))) 
        self.assertFalse(os.path.exists(os.path.join(path, '_people_metadata.csv')))
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_csv(df1, 'people', path)
        store_df_csv(df1, 'peoples', path)
        names = set(table_names_csv(path))
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)