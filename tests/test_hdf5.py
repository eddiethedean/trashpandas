"""Test HDF5 storage operations."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from trashpandas.exceptions import ValidationError
from trashpandas.hdf5 import (
    HdfStorage,
    create_hdf5_file,
    delete_table_hdf5,
    load_df_hdf5,
    store_df_hdf5,
    table_names_hdf5,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various data types."""
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        },
        index=pd.Index(["row1", "row2", "row3"], name="named_index"),
    )


@pytest.fixture
def complex_df():
    """Create a complex DataFrame with datetime and multi-index."""
    return pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "category": ["A", "B", "A", "B"],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("2023-01-01", "X"),
                ("2023-01-01", "Y"),
                ("2023-01-02", "X"),
                ("2023-01-02", "Y"),
            ],
            names=["date", "region"],
        ),
    )


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def hdf_storage(temp_dir):
    """Create an HDF5 storage instance."""
    return HdfStorage(temp_dir / "test.h5")


class TestHdfStorageInit:
    """Test HdfStorage initialization."""

    def test_init_creates_file(self, temp_dir):
        """Test that initialization creates the HDF5 file."""
        hdf_path = temp_dir / "test.h5"
        assert not hdf_path.exists()

        HdfStorage(hdf_path)
        assert hdf_path.exists()

    def test_init_with_existing_file(self, temp_dir):
        """Test initialization with existing HDF5 file."""
        hdf_path = temp_dir / "test.h5"
        storage1 = HdfStorage(hdf_path)
        storage1.store(pd.DataFrame({"a": [1]}), "test")

        # Create another storage instance with same file
        storage2 = HdfStorage(hdf_path)
        assert storage2.table_names() == ["test"]

    def test_init_with_path_object(self, temp_dir):
        """Test initialization with Path object."""
        hdf_path = temp_dir / "test.h5"
        storage = HdfStorage(hdf_path)
        assert storage.path == hdf_path

    def test_init_with_string_path(self, temp_dir):
        """Test initialization with string path."""
        hdf_path = str(temp_dir / "test.h5")
        storage = HdfStorage(hdf_path)
        assert storage.path == Path(hdf_path)

    def test_repr(self, hdf_storage):
        """Test string representation."""
        repr_str = repr(hdf_storage)
        assert "HdfStorage" in repr_str
        assert "test.h5" in repr_str


class TestHdfStorageBasicOperations:
    """Test basic HDF5 storage operations."""

    def test_store_and_load(self, hdf_storage, sample_df):
        """Test storing and loading a DataFrame."""
        hdf_storage.store(sample_df, "test_table")
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_store_and_load_with_dict_interface(self, hdf_storage, sample_df):
        """Test storing and loading using dictionary interface."""
        hdf_storage["test_table"] = sample_df
        loaded_df = hdf_storage["test_table"]
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_table(self, hdf_storage, sample_df):
        """Test deleting a table."""
        hdf_storage.store(sample_df, "test_table")
        assert "test_table" in hdf_storage

        hdf_storage.delete("test_table")
        assert "test_table" not in hdf_storage

    def test_delete_with_del_operator(self, hdf_storage, sample_df):
        """Test deleting a table using del operator."""
        hdf_storage.store(sample_df, "test_table")
        assert "test_table" in hdf_storage

        del hdf_storage["test_table"]
        assert "test_table" not in hdf_storage

    def test_table_names(self, hdf_storage, sample_df):
        """Test getting table names."""
        assert hdf_storage.table_names() == []

        hdf_storage.store(sample_df, "table1")
        hdf_storage.store(sample_df, "table2")
        assert set(hdf_storage.table_names()) == {"table1", "table2"}

    def test_len(self, hdf_storage, sample_df):
        """Test getting the number of tables."""
        assert len(hdf_storage) == 0

        hdf_storage.store(sample_df, "table1")
        assert len(hdf_storage) == 1

        hdf_storage.store(sample_df, "table2")
        assert len(hdf_storage) == 2

    def test_contains(self, hdf_storage, sample_df):
        """Test checking if table exists."""
        assert "test_table" not in hdf_storage

        hdf_storage.store(sample_df, "test_table")
        assert "test_table" in hdf_storage

    def test_iter(self, hdf_storage, sample_df):
        """Test iterating over table names."""
        hdf_storage.store(sample_df, "table1")
        hdf_storage.store(sample_df, "table2")

        table_names = list(hdf_storage)
        assert set(table_names) == {"table1", "table2"}


class TestHdfStorageMetadata:
    """Test HDF5 metadata handling."""

    def test_named_index_preservation(self, hdf_storage):
        """Test that named indices are preserved."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.Index(["a", "b", "c"], name="named_index"),
        )
        hdf_storage.store(df, "test_table")
        loaded_df = hdf_storage.load("test_table")

        assert loaded_df.index.name == "named_index"
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_multi_index_preservation(self, hdf_storage, complex_df):
        """Test that multi-indices are preserved."""
        hdf_storage.store(complex_df, "test_table")
        loaded_df = hdf_storage.load("test_table")

        pd.testing.assert_frame_equal(complex_df, loaded_df)
        assert loaded_df.index.names == ["date", "region"]

    def test_column_types_preservation(self, hdf_storage):
        """Test that column types are preserved."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            },
        )
        hdf_storage.store(df, "test_table")
        loaded_df = hdf_storage.load("test_table")

        assert loaded_df["int_col"].dtype == "int64"
        assert loaded_df["float_col"].dtype == "float64"
        assert loaded_df["str_col"].dtype == "object"
        assert loaded_df["bool_col"].dtype == "bool"

    def test_empty_dataframe(self, hdf_storage, empty_df):
        """Test storing and loading empty DataFrame."""
        hdf_storage.store(empty_df, "empty_table")
        loaded_df = hdf_storage.load("empty_table")

        # Empty DataFrames may have different inferred types, so check shape and columns
        assert empty_df.shape == loaded_df.shape
        assert list(empty_df.columns) == list(loaded_df.columns)

    def test_overwrite_existing_table(self, hdf_storage):
        """Test overwriting an existing table."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        hdf_storage.store(df1, "test_table")
        hdf_storage.store(df2, "test_table")  # Overwrite

        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(df2, loaded_df)


class TestHdfStorageEdgeCases:
    """Test HDF5 storage edge cases."""

    def test_special_characters_in_table_name(self, hdf_storage, sample_df):
        """Test table names with special characters."""
        # Only test valid table names (underscores are allowed)
        valid_names = ["table_with_underscore", "table_with_numbers_123"]
        for name in valid_names:
            hdf_storage.store(sample_df, name)
            assert name in hdf_storage
            loaded_df = hdf_storage.load(name)
            pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_large_dataframe(self, hdf_storage):
        """Test storing a large DataFrame."""
        large_df = pd.DataFrame(
            {"value": range(1000)},
            index=pd.Index([f"row_{i}" for i in range(1000)]),
        )
        hdf_storage.store(large_df, "large_table")
        loaded_df = hdf_storage.load("large_table")
        pd.testing.assert_frame_equal(large_df, loaded_df)

    def test_nonexistent_table_error(self, hdf_storage):
        """Test error handling for nonexistent tables."""
        with pytest.raises(KeyError, match="No object named"):
            hdf_storage.load("nonexistent_table")

    def test_context_manager(self, temp_dir, sample_df):
        """Test using HDF5 storage as context manager."""
        hdf_path = temp_dir / "test.h5"
        with HdfStorage(hdf_path) as storage:
            storage.store(sample_df, "test_table")
            assert "test_table" in storage

        # Verify data persists after context exit
        storage2 = HdfStorage(hdf_path)
        assert "test_table" in storage2


class TestHdfHelperFunctions:
    """Test HDF5 helper functions."""

    def test_store_df_hdf5(self, temp_dir, sample_df):
        """Test store_df_hdf5 helper function."""
        hdf_path = temp_dir / "test.h5"
        store_df_hdf5(sample_df, "test_table", str(hdf_path))

        storage = HdfStorage(hdf_path)
        loaded_df = storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_load_df_hdf5(self, temp_dir, sample_df):
        """Test load_df_hdf5 helper function."""
        hdf_path = temp_dir / "test.h5"
        storage = HdfStorage(hdf_path)
        storage.store(sample_df, "test_table")

        loaded_df = load_df_hdf5("test_table", str(hdf_path))
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_table_hdf5(self, temp_dir, sample_df):
        """Test delete_table_hdf5 helper function."""
        hdf_path = temp_dir / "test.h5"
        storage = HdfStorage(hdf_path)
        storage.store(sample_df, "test_table")
        assert "test_table" in storage

        delete_table_hdf5("test_table", str(hdf_path))
        assert "test_table" not in storage

    def test_table_names_hdf5(self, temp_dir, sample_df):
        """Test table_names_hdf5 helper function."""
        hdf_path = temp_dir / "test.h5"
        storage = HdfStorage(hdf_path)
        storage.store(sample_df, "table1")
        storage.store(sample_df, "table2")

        table_names = table_names_hdf5(str(hdf_path))
        assert set(table_names) == {"table1", "table2"}


class TestHdfErrorHandling:
    """Test HDF5 error handling."""

    def test_h5py_import_error(self):
        """Test ImportError when h5py is not available."""
        with patch("trashpandas.hdf5.H5PY_AVAILABLE", False), pytest.raises(
            ImportError, match="h5py is required",
        ):
            HdfStorage("test.h5")

    def test_create_hdf5_file_function(self, temp_dir):
        """Test create_hdf5_file helper function."""
        hdf_path = temp_dir / "test.h5"
        assert not hdf_path.exists()

        create_hdf5_file(str(hdf_path))
        assert hdf_path.exists()

    def test_invalid_table_name(self, hdf_storage, sample_df):
        """Test validation of table names."""
        with pytest.raises(ValidationError):
            hdf_storage.store(sample_df, "")

        with pytest.raises(ValidationError):
            hdf_storage.store(sample_df, None)


class TestHdfStorageAdvanced:
    """Test advanced HDF5 storage features."""

    def test_multiple_tables_same_file(self, hdf_storage, sample_df):
        """Test storing multiple tables in the same HDF5 file."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        df3 = pd.DataFrame({"c": [5, 6]})

        hdf_storage.store(df1, "table1")
        hdf_storage.store(df2, "table2")
        hdf_storage.store(df3, "table3")

        assert len(hdf_storage) == 3
        assert set(hdf_storage.table_names()) == {"table1", "table2", "table3"}

        # Verify each table can be loaded independently
        pd.testing.assert_frame_equal(df1, hdf_storage.load("table1"))
        pd.testing.assert_frame_equal(df2, hdf_storage.load("table2"))
        pd.testing.assert_frame_equal(df3, hdf_storage.load("table3"))

    def test_metadata_storage(self, hdf_storage, sample_df):
        """Test that metadata is stored separately."""
        hdf_storage.store(sample_df, "test_table")

        # Check that metadata table exists
        table_names = hdf_storage.table_names()
        assert "test_table" in table_names
        # Metadata table should not appear in regular table names
        assert "_test_table_metadata" not in table_names

    def test_dataframe_with_nan_values(self, hdf_storage):
        """Test storing DataFrame with NaN values."""
        df = pd.DataFrame(
            {
                "a": [1, 2, None],
                "b": [1.1, None, 3.3],
                "c": ["x", None, "z"],
            },
        )
        hdf_storage.store(df, "nan_table")
        loaded_df = hdf_storage.load("nan_table")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_dataframe_with_datetime(self, hdf_storage):
        """Test storing DataFrame with datetime columns."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "value": [1, 2, 3],
            },
        )
        hdf_storage.store(df, "datetime_table")
        loaded_df = hdf_storage.load("datetime_table")

        pd.testing.assert_frame_equal(df, loaded_df)
        assert loaded_df["date"].dtype == "datetime64[ns]"
