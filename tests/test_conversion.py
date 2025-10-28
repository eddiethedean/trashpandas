"""Test conversion functions between storage types."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from trashpandas.conversion import (
    convert_all_tables_storage,
    convert_table_storage,
    csv_to_hdf,
    csv_to_hdf_all,
    csv_to_pickle,
    csv_to_pickle_all,
    csv_to_sql,
    csv_to_sql_all,
    hdf_to_csv,
    hdf_to_csv_all,
    hdf_to_pickle,
    hdf_to_pickle_all,
    hdf_to_sql,
    hdf_to_sql_all,
    pickle_to_csv,
    pickle_to_csv_all,
    pickle_to_hdf,
    pickle_to_hdf_all,
    pickle_to_sql,
    pickle_to_sql_all,
    sql_to_csv,
    sql_to_csv_all,
    sql_to_hdf,
    sql_to_hdf_all,
    sql_to_pickle,
    sql_to_pickle_all,
)
from trashpandas.csv import CsvStorage
from trashpandas.hdf5 import HdfStorage
from trashpandas.pickle import PickleStorage
from trashpandas.sql import SqlStorage


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
            [("2023-01-01", "X"), ("2023-01-01", "Y"), ("2023-01-02", "X"), ("2023-01-02", "Y")],
            names=["date", "region"],
        ),
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sqlite_engine():
    """Create an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


class TestConvertTableStorage:
    """Test the generic convert_table_storage function."""

    def test_csv_to_sql_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from CSV to SQL storage."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert to SQL
        sql_storage = SqlStorage(sqlite_engine)
        convert_table_storage("test_table", csv_storage, sql_storage)

        # Verify conversion
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_csv_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from SQL to CSV storage."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert to CSV
        csv_storage = CsvStorage(temp_dir)
        convert_table_storage("test_table", sql_storage, csv_storage)

        # Verify conversion
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_csv_to_pickle_conversion(self, sample_df, temp_dir):
        """Test converting a table from CSV to Pickle storage."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert to Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        convert_table_storage("test_table", csv_storage, pickle_storage)

        # Verify conversion
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_csv_conversion(self, sample_df, temp_dir):
        """Test converting a table from Pickle to CSV storage."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert to CSV
        csv_storage = CsvStorage(temp_dir / "csv")
        convert_table_storage("test_table", pickle_storage, csv_storage)

        # Verify conversion
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_csv_to_hdf_conversion(self, sample_df, temp_dir):
        """Test converting a table from CSV to HDF5 storage."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert to HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        convert_table_storage("test_table", csv_storage, hdf_storage)

        # Verify conversion
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_csv_conversion(self, sample_df, temp_dir):
        """Test converting a table from HDF5 to CSV storage."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert to CSV
        csv_storage = CsvStorage(temp_dir / "csv")
        convert_table_storage("test_table", hdf_storage, csv_storage)

        # Verify conversion
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_pickle_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from SQL to Pickle storage."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert to Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        convert_table_storage("test_table", sql_storage, pickle_storage)

        # Verify conversion
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_sql_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from Pickle to SQL storage."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert to SQL
        sql_storage = SqlStorage(sqlite_engine)
        convert_table_storage("test_table", pickle_storage, sql_storage)

        # Verify conversion
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_hdf_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from SQL to HDF5 storage."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert to HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        convert_table_storage("test_table", sql_storage, hdf_storage)

        # Verify conversion
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_sql_conversion(self, sample_df, temp_dir, sqlite_engine):
        """Test converting a table from HDF5 to SQL storage."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert to SQL
        sql_storage = SqlStorage(sqlite_engine)
        convert_table_storage("test_table", hdf_storage, sql_storage)

        # Verify conversion
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_hdf_conversion(self, sample_df, temp_dir):
        """Test converting a table from Pickle to HDF5 storage."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert to HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        convert_table_storage("test_table", pickle_storage, hdf_storage)

        # Verify conversion
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_pickle_conversion(self, sample_df, temp_dir):
        """Test converting a table from HDF5 to Pickle storage."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert to Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        convert_table_storage("test_table", hdf_storage, pickle_storage)

        # Verify conversion
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)


class TestConvertAllTablesStorage:
    """Test the convert_all_tables_storage function."""

    def test_convert_all_csv_to_sql(self, temp_dir, sqlite_engine):
        """Test converting all tables from CSV to SQL storage."""
        # Create multiple DataFrames in CSV
        csv_storage = CsvStorage(temp_dir)
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        csv_storage.store(df1, "table1")
        csv_storage.store(df2, "table2")

        # Convert all to SQL
        sql_storage = SqlStorage(sqlite_engine)
        convert_all_tables_storage(csv_storage, sql_storage)

        # Verify all tables were converted
        assert sql_storage.table_names() == ["table1", "table2"]
        pd.testing.assert_frame_equal(df1, sql_storage.load("table1"))
        pd.testing.assert_frame_equal(df2, sql_storage.load("table2"))

    def test_convert_all_sql_to_csv(self, temp_dir, sqlite_engine):
        """Test converting all tables from SQL to CSV storage."""
        # Create multiple DataFrames in SQL
        sql_storage = SqlStorage(sqlite_engine)
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        sql_storage.store(df1, "table1")
        sql_storage.store(df2, "table2")

        # Convert all to CSV
        csv_storage = CsvStorage(temp_dir)
        convert_all_tables_storage(sql_storage, csv_storage)

        # Verify all tables were converted
        assert set(csv_storage.table_names()) == {"table1", "table2"}
        pd.testing.assert_frame_equal(df1, csv_storage.load("table1"))
        pd.testing.assert_frame_equal(df2, csv_storage.load("table2"))


class TestCsvConversions:
    """Test CSV-specific conversion functions."""

    def test_csv_to_sql_function(self, sample_df, temp_dir, sqlite_engine):
        """Test csv_to_sql function."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert using function
        csv_to_sql("test_table", str(temp_dir), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_csv_to_sql_all_function(self, temp_dir, sqlite_engine):
        """Test csv_to_sql_all function."""
        # Create multiple DataFrames in CSV
        csv_storage = CsvStorage(temp_dir)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        csv_storage.store(df1, "table1")
        csv_storage.store(df2, "table2")

        # Convert all using function
        csv_to_sql_all(str(temp_dir), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        assert sql_storage.table_names() == ["table1", "table2"]

    def test_csv_to_hdf_function(self, sample_df, temp_dir):
        """Test csv_to_hdf function."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert using function
        csv_to_hdf("test_table", str(temp_dir), str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_csv_to_hdf_all_function(self, temp_dir):
        """Test csv_to_hdf_all function."""
        # Create multiple DataFrames in CSV
        csv_storage = CsvStorage(temp_dir)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        csv_storage.store(df1, "table1")
        csv_storage.store(df2, "table2")

        # Convert all using function
        csv_to_hdf_all(str(temp_dir), str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        assert hdf_storage.table_names() == ["table1", "table2"]

    def test_csv_to_pickle_function(self, sample_df, temp_dir):
        """Test csv_to_pickle function."""
        # Store DataFrame in CSV
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(sample_df, "test_table")

        # Convert using function
        csv_to_pickle("test_table", str(temp_dir), str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_csv_to_pickle_all_function(self, temp_dir):
        """Test csv_to_pickle_all function."""
        # Create multiple DataFrames in CSV
        csv_storage = CsvStorage(temp_dir)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        csv_storage.store(df1, "table1")
        csv_storage.store(df2, "table2")

        # Convert all using function
        csv_to_pickle_all(str(temp_dir), str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        assert set(pickle_storage.table_names()) == {"table1", "table2"}


class TestSqlConversions:
    """Test SQL-specific conversion functions."""

    def test_sql_to_csv_function(self, sample_df, temp_dir, sqlite_engine):
        """Test sql_to_csv function."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert using function
        sql_to_csv("test_table", sqlite_engine, str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_csv_all_function(self, temp_dir, sqlite_engine):
        """Test sql_to_csv_all function."""
        # Create multiple DataFrames in SQL
        sql_storage = SqlStorage(sqlite_engine)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        sql_storage.store(df1, "table1")
        sql_storage.store(df2, "table2")

        # Convert all using function
        sql_to_csv_all(sqlite_engine, str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        assert set(csv_storage.table_names()) == {"table1", "table2"}

    def test_sql_to_hdf_function(self, sample_df, temp_dir, sqlite_engine):
        """Test sql_to_hdf function."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert using function
        sql_to_hdf("test_table", sqlite_engine, str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_hdf_all_function(self, temp_dir, sqlite_engine):
        """Test sql_to_hdf_all function."""
        # Create multiple DataFrames in SQL
        sql_storage = SqlStorage(sqlite_engine)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        sql_storage.store(df1, "table1")
        sql_storage.store(df2, "table2")

        # Convert all using function
        sql_to_hdf_all(sqlite_engine, str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        assert hdf_storage.table_names() == ["table1", "table2"]

    def test_sql_to_pickle_function(self, sample_df, temp_dir, sqlite_engine):
        """Test sql_to_pickle function."""
        # Store DataFrame in SQL
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(sample_df, "test_table")

        # Convert using function
        sql_to_pickle("test_table", sqlite_engine, str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_sql_to_pickle_all_function(self, temp_dir, sqlite_engine):
        """Test sql_to_pickle_all function."""
        # Create multiple DataFrames in SQL
        sql_storage = SqlStorage(sqlite_engine)
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        sql_storage.store(df1, "table1")
        sql_storage.store(df2, "table2")

        # Convert all using function
        sql_to_pickle_all(sqlite_engine, str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        assert set(pickle_storage.table_names()) == {"table1", "table2"}


class TestHdfConversions:
    """Test HDF5-specific conversion functions."""

    def test_hdf_to_csv_function(self, sample_df, temp_dir):
        """Test hdf_to_csv function."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert using function
        hdf_to_csv("test_table", str(temp_dir / "test.h5"), str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_csv_all_function(self, temp_dir):
        """Test hdf_to_csv_all function."""
        # Create multiple DataFrames in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        hdf_storage.store(df1, "table1")
        hdf_storage.store(df2, "table2")

        # Convert all using function
        hdf_to_csv_all(str(temp_dir / "test.h5"), str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        assert set(csv_storage.table_names()) == {"table1", "table2"}

    def test_hdf_to_sql_function(self, sample_df, temp_dir, sqlite_engine):
        """Test hdf_to_sql function."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert using function
        hdf_to_sql("test_table", str(temp_dir / "test.h5"), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_sql_all_function(self, temp_dir, sqlite_engine):
        """Test hdf_to_sql_all function."""
        # Create multiple DataFrames in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        hdf_storage.store(df1, "table1")
        hdf_storage.store(df2, "table2")

        # Convert all using function
        hdf_to_sql_all(str(temp_dir / "test.h5"), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        assert sql_storage.table_names() == ["table1", "table2"]

    def test_hdf_to_pickle_function(self, sample_df, temp_dir):
        """Test hdf_to_pickle function."""
        # Store DataFrame in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(sample_df, "test_table")

        # Convert using function
        hdf_to_pickle("test_table", str(temp_dir / "test.h5"), str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        loaded_df = pickle_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_hdf_to_pickle_all_function(self, temp_dir):
        """Test hdf_to_pickle_all function."""
        # Create multiple DataFrames in HDF5
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        hdf_storage.store(df1, "table1")
        hdf_storage.store(df2, "table2")

        # Convert all using function
        hdf_to_pickle_all(str(temp_dir / "test.h5"), str(temp_dir / "pickle"))

        # Verify conversion
        pickle_storage = PickleStorage(temp_dir / "pickle")
        assert set(pickle_storage.table_names()) == {"table1", "table2"}


class TestPickleConversions:
    """Test Pickle-specific conversion functions."""

    def test_pickle_to_csv_function(self, sample_df, temp_dir):
        """Test pickle_to_csv function."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert using function
        pickle_to_csv("test_table", str(temp_dir / "pickle"), str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        loaded_df = csv_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_csv_all_function(self, temp_dir):
        """Test pickle_to_csv_all function."""
        # Create multiple DataFrames in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        pickle_storage.store(df1, "table1")
        pickle_storage.store(df2, "table2")

        # Convert all using function
        pickle_to_csv_all(str(temp_dir / "pickle"), str(temp_dir))

        # Verify conversion
        csv_storage = CsvStorage(temp_dir)
        assert set(csv_storage.table_names()) == {"table1", "table2"}

    def test_pickle_to_sql_function(self, sample_df, temp_dir, sqlite_engine):
        """Test pickle_to_sql function."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert using function
        pickle_to_sql("test_table", str(temp_dir / "pickle"), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        loaded_df = sql_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_sql_all_function(self, temp_dir, sqlite_engine):
        """Test pickle_to_sql_all function."""
        # Create multiple DataFrames in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        pickle_storage.store(df1, "table1")
        pickle_storage.store(df2, "table2")

        # Convert all using function
        pickle_to_sql_all(str(temp_dir / "pickle"), sqlite_engine)

        # Verify conversion
        sql_storage = SqlStorage(sqlite_engine)
        assert sql_storage.table_names() == ["table1", "table2"]

    def test_pickle_to_hdf_function(self, sample_df, temp_dir):
        """Test pickle_to_hdf function."""
        # Store DataFrame in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        pickle_storage.store(sample_df, "test_table")

        # Convert using function
        pickle_to_hdf("test_table", str(temp_dir / "pickle"), str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        loaded_df = hdf_storage.load("test_table")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_pickle_to_hdf_all_function(self, temp_dir):
        """Test pickle_to_hdf_all function."""
        # Create multiple DataFrames in Pickle
        pickle_storage = PickleStorage(temp_dir / "pickle")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        pickle_storage.store(df1, "table1")
        pickle_storage.store(df2, "table2")

        # Convert all using function
        pickle_to_hdf_all(str(temp_dir / "pickle"), str(temp_dir / "test.h5"))

        # Verify conversion
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        assert hdf_storage.table_names() == ["table1", "table2"]


class TestComplexDataConversions:
    """Test conversions with complex DataFrames."""

    def test_complex_df_conversion_csv_to_sql(
        self, complex_df, temp_dir, sqlite_engine,
    ):
        """Test converting complex DataFrame from CSV to SQL."""
        csv_storage = CsvStorage(temp_dir)
        csv_storage.store(complex_df, "complex_table")

        sql_storage = SqlStorage(sqlite_engine)
        convert_table_storage("complex_table", csv_storage, sql_storage)

        loaded_df = sql_storage.load("complex_table")
        pd.testing.assert_frame_equal(complex_df, loaded_df)

    def test_complex_df_conversion_sql_to_hdf(
        self, complex_df, temp_dir, sqlite_engine,
    ):
        """Test converting complex DataFrame from SQL to HDF5."""
        sql_storage = SqlStorage(sqlite_engine)
        sql_storage.store(complex_df, "complex_table")

        hdf_storage = HdfStorage(temp_dir / "test.h5")
        convert_table_storage("complex_table", sql_storage, hdf_storage)

        loaded_df = hdf_storage.load("complex_table")
        pd.testing.assert_frame_equal(complex_df, loaded_df)

    def test_complex_df_conversion_hdf_to_pickle(self, complex_df, temp_dir):
        """Test converting complex DataFrame from HDF5 to Pickle."""
        hdf_storage = HdfStorage(temp_dir / "test.h5")
        hdf_storage.store(complex_df, "complex_table")

        pickle_storage = PickleStorage(temp_dir / "pickle")
        convert_table_storage("complex_table", hdf_storage, pickle_storage)

        loaded_df = pickle_storage.load("complex_table")
        pd.testing.assert_frame_equal(complex_df, loaded_df)
