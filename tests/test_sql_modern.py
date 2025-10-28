"""Modern pytest-style tests for SQL storage functionality."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

from trashpandas.sql import (
    SqlStorage,
    delete_table_sql,
    load_df_sql,
    store_df_sql,
    table_names_sql,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sqlite_engine():
    """Create a SQLite engine for testing using in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture
def storage(sqlite_engine):
    """Create a SqlStorage instance for testing."""
    return SqlStorage(sqlite_engine)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"],
    })


@pytest.fixture
def named_index_df() -> pd.DataFrame:
    """Create a DataFrame with named index for testing."""
    df = pd.DataFrame({
        "value": [1, 2, 3, 4],
        "category": ["A", "B", "A", "B"],
    })
    df.index.name = "id"
    return df


@pytest.fixture
def multi_index_df() -> pd.DataFrame:
    """Create a DataFrame with multi-index for testing."""
    data = {
        "value": [1, 2, 3, 4, 5, 6],
        "category": ["A", "B", "A", "B", "A", "B"],
    }
    index = pd.MultiIndex.from_tuples([
        ("2023", "Q1"), ("2023", "Q2"), ("2023", "Q3"),
        ("2024", "Q1"), ("2024", "Q2"), ("2024", "Q3"),
    ], names=["year", "quarter"])
    return pd.DataFrame(data, index=index)


class TestSqlStorage:
    """Test cases for SqlStorage class."""

    def test_store_and_load(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test storing and loading a DataFrame."""
        storage.store(sample_df, "people")
        loaded_df = storage.load("people")

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_dictionary_interface(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test dictionary-like interface."""
        storage["people"] = sample_df
        loaded_df = storage["people"]

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_table(self, storage: SqlStorage, sample_df: pd.DataFrame, sqlite_engine):
        """Test deleting a table."""
        storage.store(sample_df, "people")
        storage.delete("people")

        table_names = inspect(sqlite_engine).get_table_names()
        assert "people" not in table_names
        assert "_people_metadata" not in table_names

    def test_delete_with_del_operator(
        self, storage: SqlStorage, sample_df: pd.DataFrame, sqlite_engine,
    ):
        """Test deleting a table using del operator."""
        storage["people"] = sample_df
        del storage["people"]

        table_names = inspect(sqlite_engine).get_table_names()
        assert "people" not in table_names
        assert "_people_metadata" not in table_names

    def test_table_names(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test getting table names."""
        storage.store(sample_df, "people")
        storage.store(sample_df, "users")

        table_names = set(storage.table_names())
        expected = {"people", "users"}
        assert table_names == expected

    def test_metadata_names(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test getting metadata table names."""
        storage.store(sample_df, "people")

        metadata_names = storage.metadata_names()
        assert "_people_metadata" in metadata_names

    def test_context_manager(self, sqlite_engine, sample_df: pd.DataFrame):
        """Test context manager functionality."""
        with SqlStorage(sqlite_engine) as storage:
            storage["people"] = sample_df
            loaded_df = storage["people"]
            pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_iterator_protocol(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test iterator protocol."""
        storage.store(sample_df, "people")
        storage.store(sample_df, "users")

        # Test iteration
        table_names = list(storage)
        assert set(table_names) == {"people", "users"}

        # Test length
        assert len(storage) == 2

        # Test contains
        assert "people" in storage
        assert "users" in storage
        assert "nonexistent" not in storage

    def test_bulk_operations(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test bulk operations."""
        # Store multiple DataFrames
        dataframes = {
            "users": sample_df,
            "orders": sample_df.copy(),
            "products": sample_df.copy(),
        }
        storage.store_many(dataframes)

        # Load multiple DataFrames
        table_names = ["users", "orders", "products"]
        results = storage.load_many(table_names)

        assert set(results.keys()) == set(table_names)
        for df in results.values():
            pd.testing.assert_frame_equal(sample_df, df)

        # Delete multiple tables
        storage.delete_many(table_names)
        assert len(storage) == 0

    def test_query_functionality(self, storage: SqlStorage, sample_df: pd.DataFrame):
        """Test query functionality."""
        storage.store(sample_df, "people")

        # Test basic query
        result = storage.query("people")
        pd.testing.assert_frame_equal(sample_df, result)

        # Test query with WHERE clause
        result = storage.query("people", where_clause="age > 25")
        expected = sample_df[sample_df["age"] > 25]
        pd.testing.assert_frame_equal(expected, result)

        # Test query with column selection
        result = storage.query("people", columns=["name", "age"])
        expected = sample_df[["name", "age"]]
        pd.testing.assert_frame_equal(expected, result)

        # Test query with limit
        result = storage.query("people", limit=2)
        assert len(result) == 2

    def test_named_index_preservation(
        self, storage: SqlStorage, named_index_df: pd.DataFrame,
    ):
        """Test that named indexes are preserved."""
        storage.store(named_index_df, "data")
        loaded_df = storage.load("data")

        pd.testing.assert_frame_equal(named_index_df, loaded_df)
        assert loaded_df.index.name == named_index_df.index.name

    def test_multi_index_preservation(
        self, storage: SqlStorage, multi_index_df: pd.DataFrame,
    ):
        """Test that multi-indexes are preserved."""
        storage.store(multi_index_df, "data")
        loaded_df = storage.load("data")

        pd.testing.assert_frame_equal(multi_index_df, loaded_df)
        assert loaded_df.index.names == multi_index_df.index.names

    def test_string_data_preservation(self, storage: SqlStorage):
        """Test that string data types are preserved."""
        df = pd.DataFrame({
            "text": ["hello", "world", "test"],
            "number": [1, 2, 3],
        })
        df["text"] = df["text"].astype("string")

        storage.store(df, "text_data")
        loaded_df = storage.load("text_data")

        pd.testing.assert_frame_equal(df, loaded_df)
        assert loaded_df["text"].dtype == "string"

    def test_nonexistent_table_error(self, storage: SqlStorage):
        """Test error handling for nonexistent tables."""
        with pytest.raises(ValueError, match="Table.*not found"):
            storage.load("nonexistent_table")


class TestSqlFunctions:
    """Test cases for SQL function interfaces."""

    def test_store_load_functions(self, sqlite_engine, sample_df: pd.DataFrame):
        """Test store and load functions."""
        store_df_sql(sample_df, "people", sqlite_engine)
        loaded_df = load_df_sql("people", sqlite_engine)

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_function(self, sqlite_engine, sample_df: pd.DataFrame):
        """Test delete function."""
        store_df_sql(sample_df, "people", sqlite_engine)
        delete_table_sql("people", sqlite_engine)

        table_names = inspect(sqlite_engine).get_table_names()
        assert "people" not in table_names
        assert "_people_metadata" not in table_names

    def test_table_names_function(self, sqlite_engine, sample_df: pd.DataFrame):
        """Test table_names function."""
        store_df_sql(sample_df, "people", sqlite_engine)
        store_df_sql(sample_df, "users", sqlite_engine)

        table_names = set(table_names_sql(sqlite_engine))
        expected = {"people", "users"}
        assert table_names == expected


@pytest.mark.parametrize("compression", [None, "gzip", "bz2"])
def test_compression_support(compression):
    """Test compression support (if implemented)."""
    # This is a placeholder for compression testing
    # In a full implementation, this would test CSV/pickle compression
    pass


@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations (placeholder)."""
    # This is a placeholder for async testing
    # In a full implementation, this would test AsyncSqlStorage
    pass


class TestSecurity:
    """Test security features and SQL injection prevention."""

    def test_query_with_valid_where_clause(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that valid WHERE clauses work correctly."""
        storage.store(sample_df, "test_table")

        # Test with valid WHERE clause
        result = storage.query("test_table", where_clause="age > 25")
        assert len(result) > 0
        assert all(result["age"] > 25)

    def test_query_with_sql_injection_detection(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that SQL injection attempts are detected and rejected."""
        from trashpandas.exceptions import ValidationError

        storage.store(sample_df, "test_table")

        # Test dangerous SQL patterns
        dangerous_queries = [
            "1 = 1; DROP TABLE test_table",
            "age > 25; DELETE FROM test_table",
            "1=1 UNION SELECT * FROM users",
            "age > 25 -- malicious comment",
            "age > 25/* comment */",
        ]

        for dangerous_query in dangerous_queries:
            with pytest.raises(ValidationError) as exc_info:
                storage.query("test_table", where_clause=dangerous_query)

            assert "WHERE clause contains dangerous SQL patterns" in str(exc_info.value)

    def test_query_with_invalid_column_names(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that invalid column names are rejected."""
        from trashpandas.exceptions import ValidationError

        storage.store(sample_df, "test_table")

        # Test with invalid column names
        invalid_columns = [
            ["name; DROP TABLE", "age"],
            ["name;", "age"],
            ["name -- comment", "age"],
        ]

        for invalid_cols in invalid_columns:
            with pytest.raises(ValidationError) as exc_info:
                storage.query("test_table", columns=invalid_cols)

            assert "Invalid characters in column name" in str(exc_info.value)

    def test_query_with_valid_column_names(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that valid column names work correctly."""
        storage.store(sample_df, "test_table")

        # Test with valid column names
        result = storage.query("test_table", columns=["name", "age"])
        assert list(result.columns) == ["name", "age"]

    def test_query_prevents_code_injection_via_pandas_query(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that the new implementation prevents code injection via DataFrame.query()."""

        # Create a file to detect code execution
        test_file = "/tmp/trashpandas_security_test.tmp"  # noqa: S108
        if os.path.exists(test_file):
            os.remove(test_file)

        storage.store(sample_df, "test_table")

        # Attempt code execution via pandas query syntax (should fail)
        # Note: The old implementation allowed this, the new one uses SQL directly
        malicious_query = (
            "__import__('os').system('touch /tmp/trashpandas_security_test.tmp')"
        )

        from trashpandas.exceptions import ValidationError

        try:
            # The new implementation should reject this as dangerous SQL
            storage.query("test_table", where_clause=malicious_query)
            # If we get here, the query was executed but shouldn't have been
            # In the best case, it would have been rejected as invalid SQL
            assert not os.path.exists(test_file), (
                "Security test failed - code was executed!"
            )
        except ValidationError:
            # This is expected - the query should fail with a specific error
            pass
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_query_preserves_metadata_and_types(
        self, storage: SqlStorage, sample_df: pd.DataFrame,
    ):
        """Test that queries preserve DataFrame metadata and types."""
        storage.store(sample_df, "test_table")

        # Original should have proper types
        original = storage.load("test_table")

        # Query result should also have proper types
        result = storage.query("test_table", where_clause="age > 25")

        # Check that types are preserved
        assert result["name"].dtype == original["name"].dtype
        assert result["age"].dtype == original["age"].dtype
