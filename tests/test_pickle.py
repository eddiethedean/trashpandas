import os
import unittest

from setup_test import (
    create_df,
    create_df_string,
    create_named_index_df,
    create_two_index_one_named_df,
    create_two_unnamed_index_df,
    delete_all_files,
)

from trashpandas.pickle import (
    PickleStorage,
    delete_table_pickle,
    load_df_pickle,
    store_df_pickle,
    table_names_pickle,
)

path = os.getcwd() + "/tests/data"


class TestPickleStorage(unittest.TestCase):
    def test_store_load(self):
        delete_all_files(path)
        storage = PickleStorage(path)
        df1 = create_df()
        storage.store(df1, "people")
        df2 = storage.load("people")
        assert df1.equals(df2)
        delete_all_files(path)

    def test_delete(self):
        delete_all_files(path)
        storage = PickleStorage(path)
        df1 = create_df()
        storage.store(df1, "people")
        storage.delete("people")
        assert not os.path.exists(os.path.join(path, "people.pickle"))
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        storage = PickleStorage(path)
        df1 = create_df()
        storage.store(df1, "people")
        storage.store(df1, "peoples")
        names = set(storage.table_names())
        expected = {"people", "peoples"}
        self.assertSetEqual(names, expected)
        delete_all_files(path)


class TestPickleFunctions(unittest.TestCase):
    def test_basic_store_load(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_pickle(df1, "people", path)
        df2 = load_df_pickle("people", path)
        assert df1.equals(df2)
        delete_all_files(path)

    def test_named_store_load(self):
        delete_all_files(path)
        df1 = create_named_index_df()
        store_df_pickle(df1, "people", path)
        df2 = load_df_pickle("people", path)
        assert df1.equals(df2)
        delete_all_files(path)

    def test_two_unnamed_store_load(self):
        delete_all_files(path)
        df1 = create_two_unnamed_index_df()
        store_df_pickle(df1, "people", path)
        df2 = load_df_pickle("people", path)
        assert df1.equals(df2)
        delete_all_files(path)

    def test_one_named_store_load(self):
        delete_all_files(path)
        df1 = create_two_index_one_named_df()
        store_df_pickle(df1, "people", path)
        df2 = load_df_pickle("people", path)
        assert df1.equals(df2)
        delete_all_files(path)

    def test_string_store_load(self):
        delete_all_files(path)
        df1 = create_df_string()
        store_df_pickle(df1, "people", path)
        df2 = load_df_pickle("people", path)
        assert df1.equals(df2)
        delete_all_files(path)

    def test_delete_table(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_pickle(df1, "people", path)
        delete_table_pickle("people", path)
        assert not os.path.exists(os.path.join(path, "people.pickle"))
        delete_all_files(path)

    def test_table_names(self):
        delete_all_files(path)
        df1 = create_df()
        store_df_pickle(df1, "people", path)
        store_df_pickle(df1, "peoples", path)
        names = set(table_names_pickle(path))
        expected = {"people", "peoples"}
        self.assertSetEqual(names, expected)
        delete_all_files(path)
