import unittest
import os

from pandas import DataFrame, read_hdf

from trashpandas.hdf5 import HdfStorage
from trashpandas.hdf5 import store_df_hdf5, load_df_hdf5, delete_table_hdf5, table_names_hdf5

from setup_test import create_df
from setup_test import create_named_index_df
from setup_test import create_two_unnamed_index_df
from setup_test import create_two_index_one_named_df
from setup_test import create_df_string
from setup_test import delete_all_files


path = os.getcwd() + '/tests/data'
file_path = os.path.join(path, 'test.h5')


class TestHdfStorage(unittest.TestCase):
    def test_store_load(self):
        delete_all_files(path)
        storage = HdfStorage(file_path)
        df1 = create_df()
        storage.store(df1, 'people')
        df2 = storage.load('people')
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_delete(self):
        delete_all_files(path)
        storage = HdfStorage(file_path)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.delete('people')
        names = set(table_names_hdf5(file_path))
        expected = set()
        self.assertSetEqual(names, expected)
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        storage = HdfStorage(file_path)
        df1 = create_df()
        storage.store(df1, 'people')
        storage.store(df1, 'peoples')
        names = set(storage.table_names())
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)


class TestHdf5Functions(unittest.TestCase):
    def test_basic_store_load(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_hdf5(df1, 'people', file_path)
        df2 = load_df_hdf5('people', file_path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_named_store_load(self):
        delete_all_files(path)
        df1 = create_named_index_df()
        store_df_hdf5(df1, 'people', file_path)
        df2 = load_df_hdf5('people', file_path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_two_unnamed_store_load(self):
        delete_all_files(path)
        df1 = create_two_unnamed_index_df()
        store_df_hdf5(df1, 'people', file_path)
        df2 = load_df_hdf5('people', file_path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_one_named_store_load(self):
        delete_all_files(path)
        df1 = create_two_index_one_named_df()
        store_df_hdf5(df1, 'people', file_path)
        df2 = load_df_hdf5('people', file_path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_string_store_load(self):
        delete_all_files(path)
        df1 = create_df_string()
        store_df_hdf5(df1, 'people', file_path)
        df2 = load_df_hdf5('people', file_path)
        self.assertTrue(df1.equals(df2))
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_hdf5(df1, 'people', file_path)
        store_df_hdf5(df1, 'peoples', file_path)
        names = set(table_names_hdf5(file_path))
        expected = {'people', 'peoples'}
        self.assertSetEqual(names, expected)
        delete_all_files(path)

    def test_delete_table(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_hdf5(df1, 'people', file_path)
        delete_table_hdf5('people', file_path)
        names = set(table_names_hdf5(file_path))
        expected = set()
        self.assertSetEqual(names, expected)
        delete_all_files(path)

