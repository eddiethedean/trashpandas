"""Modern pytest-style tests for CSV storage functionality."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from trashpandas.csv import (
    CsvStorage,
    delete_table_csv,
    load_df_csv,
    store_df_csv,
    table_names_csv,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def csv_storage(temp_dir: Path) -> CsvStorage:
    """Create a CsvStorage instance for testing."""
    return CsvStorage(temp_dir)


@pytest.fixture
def csv_storage_compressed(temp_dir: Path) -> CsvStorage:
    """Create a compressed CsvStorage instance for testing."""
    return CsvStorage(temp_dir, compression="gzip")


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


class TestCsvStorage:
    """Test cases for CsvStorage class."""

    def test_store_and_load(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test storing and loading a DataFrame."""
        csv_storage.store(sample_df, "people")
        loaded_df = csv_storage.load("people")

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_dictionary_interface(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test dictionary-like interface."""
        csv_storage["people"] = sample_df
        loaded_df = csv_storage["people"]

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_table(self, csv_storage: CsvStorage, sample_df: pd.DataFrame, temp_dir: Path):
        """Test deleting a table."""
        csv_storage.store(sample_df, "people")
        csv_storage.delete("people")

        # Check that files are deleted
        csv_file = temp_dir / "people.csv"
        metadata_file = temp_dir / "_people_metadata.csv"
        assert not csv_file.exists()
        assert not metadata_file.exists()

    def test_delete_with_del_operator(self, csv_storage: CsvStorage, sample_df: pd.DataFrame, temp_dir: Path):
        """Test deleting a table using del operator."""
        csv_storage["people"] = sample_df
        del csv_storage["people"]

        # Check that files are deleted
        csv_file = temp_dir / "people.csv"
        metadata_file = temp_dir / "_people_metadata.csv"
        assert not csv_file.exists()
        assert not metadata_file.exists()

    def test_table_names(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test getting table names."""
        csv_storage.store(sample_df, "people")
        csv_storage.store(sample_df, "users")

        table_names = set(csv_storage.table_names())
        expected = {"people", "users"}
        assert table_names == expected

    def test_metadata_names(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test getting metadata table names."""
        csv_storage.store(sample_df, "people")

        metadata_names = csv_storage.metadata_names()
        assert "_people_metadata" in metadata_names

    def test_context_manager(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test context manager functionality."""
        with CsvStorage(temp_dir) as storage:
            storage["people"] = sample_df
            loaded_df = storage["people"]
            pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_iterator_protocol(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test iterator protocol."""
        csv_storage.store(sample_df, "people")
        csv_storage.store(sample_df, "users")

        # Test iteration
        table_names = list(csv_storage)
        assert set(table_names) == {"people", "users"}

        # Test length
        assert len(csv_storage) == 2

        # Test contains
        assert "people" in csv_storage
        assert "users" in csv_storage
        assert "nonexistent" not in csv_storage

    def test_bulk_operations(self, csv_storage: CsvStorage, sample_df: pd.DataFrame):
        """Test bulk operations."""
        # Store multiple DataFrames
        dataframes = {
            "users": sample_df,
            "orders": sample_df.copy(),
            "products": sample_df.copy(),
        }
        csv_storage.store_many(dataframes)

        # Load multiple DataFrames
        table_names = ["users", "orders", "products"]
        results = csv_storage.load_many(table_names)

        assert set(results.keys()) == set(table_names)
        for df in results.values():
            pd.testing.assert_frame_equal(sample_df, df)

        # Delete multiple tables
        csv_storage.delete_many(table_names)
        assert len(csv_storage) == 0

    def test_compression_support(self, csv_storage_compressed: CsvStorage, sample_df: pd.DataFrame, temp_dir: Path):
        """Test compression support."""
        csv_storage_compressed.store(sample_df, "people")

        # Check that compressed files exist
        csv_file = temp_dir / "people.csv.gzip"
        metadata_file = temp_dir / "_people_metadata.csv.gzip"
        assert csv_file.exists()
        assert metadata_file.exists()

        # Test loading compressed data
        loaded_df = csv_storage_compressed.load("people")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_named_index_preservation(self, csv_storage: CsvStorage, named_index_df: pd.DataFrame):
        """Test that named indexes are preserved."""
        csv_storage.store(named_index_df, "data")
        loaded_df = csv_storage.load("data")

        pd.testing.assert_frame_equal(named_index_df, loaded_df)
        assert loaded_df.index.name == named_index_df.index.name

    def test_string_data_preservation(self, csv_storage: CsvStorage):
        """Test that string data types are preserved."""
        df = pd.DataFrame({
            "text": ["hello", "world", "test"],
            "number": [1, 2, 3],
        })
        df["text"] = df["text"].astype("string")

        csv_storage.store(df, "text_data")
        loaded_df = csv_storage.load("text_data")

        pd.testing.assert_frame_equal(df, loaded_df)
        # Note: CSV preserves string dtypes in modern pandas
        assert loaded_df["text"].dtype == "string"  # CSV preserves string dtype

    def test_pathlib_support(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test pathlib.Path support."""
        storage = CsvStorage(temp_dir)
        storage.store(sample_df, "people")
        loaded_df = storage.load("people")

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_nonexistent_table_error(self, csv_storage: CsvStorage):
        """Test error handling for nonexistent tables."""
        with pytest.raises(FileNotFoundError):
            csv_storage.load("nonexistent_table")


class TestCsvFunctions:
    """Test cases for CSV function interfaces."""

    def test_store_load_functions(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test store and load functions."""
        store_df_csv(sample_df, "people", str(temp_dir))
        loaded_df = load_df_csv("people", str(temp_dir))

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_function(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test delete function."""
        store_df_csv(sample_df, "people", str(temp_dir))
        delete_table_csv("people", str(temp_dir))

        # Check that files are deleted
        csv_file = temp_dir / "people.csv"
        metadata_file = temp_dir / "_people_metadata.csv"
        assert not csv_file.exists()
        assert not metadata_file.exists()

    def test_table_names_function(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test table_names function."""
        store_df_csv(sample_df, "people", str(temp_dir))
        store_df_csv(sample_df, "users", str(temp_dir))

        table_names = set(table_names_csv(str(temp_dir)))
        expected = {"people", "users"}
        assert table_names == expected


@pytest.mark.parametrize("compression", [None, "gzip", "bz2", "xz"])
def test_compression_types(compression):
    """Test different compression types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = CsvStorage(tmpdir, compression=compression)
        df = pd.DataFrame({"test": [1, 2, 3]})
        storage.store(df, "test")
        loaded_df = storage.load("test")
        pd.testing.assert_frame_equal(df, loaded_df)


@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations (placeholder)."""
    # This is a placeholder for async testing
    # In a full implementation, this would test AsyncCsvStorage
    pass


def test_delete_with_compression():
    """Test that deletion works correctly with compressed CSV files."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Test with gzip compression
        storage = CsvStorage(temp_dir, compression="gzip")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Store with compression
        storage.store(df, "test_data")
        assert "test_data" in storage.table_names()

        # Verify compressed files exist
        csv_file = temp_dir / "test_data.csv.gzip"
        csv_file_gz = temp_dir / "test_data.csv.gz"
        metadata_file = temp_dir / "_test_data_metadata.csv.gzip"
        metadata_file_gz = temp_dir / "_test_data_metadata.csv.gz"

        assert csv_file.exists() or csv_file_gz.exists(), "Compressed CSV should exist"
        assert metadata_file.exists() or metadata_file_gz.exists(), "Compressed metadata should exist"

        # Delete should work with compression
        storage.delete("test_data")
        assert "test_data" not in storage.table_names()

        # Verify compressed files are deleted
        assert not csv_file.exists() and not csv_file_gz.exists(), "CSV should be deleted"
        assert not metadata_file.exists() and not metadata_file_gz.exists(), "Metadata should be deleted"
