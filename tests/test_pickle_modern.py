"""Modern pytest-style tests for pickle storage functionality."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from trashpandas.pickle import (
    PickleStorage,
    delete_table_pickle,
    load_df_pickle,
    store_df_pickle,
    table_names_pickle,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pickle_storage(temp_dir: Path) -> PickleStorage:
    """Create a PickleStorage instance for testing."""
    return PickleStorage(temp_dir)


@pytest.fixture
def pickle_storage_compressed(temp_dir: Path) -> PickleStorage:
    """Create a compressed PickleStorage instance for testing."""
    return PickleStorage(temp_dir, compression="bz2")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"],
    })


@pytest.fixture
def complex_df() -> pd.DataFrame:
    """Create a complex DataFrame for testing."""
    df = pd.DataFrame({
        "text": ["hello", "world", "test"],
        "number": [1, 2, 3],
        "float": [1.1, 2.2, 3.3],
        "bool": [True, False, True],
        "datetime": pd.date_range("2023-01-01", periods=3),
    })
    df["text"] = df["text"].astype("string")
    df.index.name = "id"
    return df


class TestPickleStorage:
    """Test cases for PickleStorage class."""

    def test_store_and_load(self, pickle_storage: PickleStorage, sample_df: pd.DataFrame):
        """Test storing and loading a DataFrame."""
        pickle_storage.store(sample_df, "people")
        loaded_df = pickle_storage.load("people")

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_dictionary_interface(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame,
    ):
        """Test dictionary-like interface."""
        pickle_storage["people"] = sample_df
        loaded_df = pickle_storage["people"]

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_table(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame, temp_dir: Path,
    ):
        """Test deleting a table."""
        pickle_storage.store(sample_df, "people")
        pickle_storage.delete("people")

        # Check that file is deleted
        pickle_file = temp_dir / "people.pickle"
        assert not pickle_file.exists()

    def test_delete_with_del_operator(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame, temp_dir: Path,
    ):
        """Test deleting a table using del operator."""
        pickle_storage["people"] = sample_df
        del pickle_storage["people"]

        # Check that file is deleted
        pickle_file = temp_dir / "people.pickle"
        assert not pickle_file.exists()

    def test_table_names(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame,
    ):
        """Test getting table names."""
        pickle_storage.store(sample_df, "people")
        pickle_storage.store(sample_df, "users")

        table_names = set(pickle_storage.table_names())
        expected = {"people", "users"}
        assert table_names == expected

    def test_context_manager(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test context manager functionality."""
        with PickleStorage(temp_dir) as storage:
            storage["people"] = sample_df
            loaded_df = storage["people"]
            pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_iterator_protocol(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame,
    ):
        """Test iterator protocol."""
        pickle_storage.store(sample_df, "people")
        pickle_storage.store(sample_df, "users")

        # Test iteration
        table_names = list(pickle_storage)
        assert set(table_names) == {"people", "users"}

        # Test length
        assert len(pickle_storage) == 2

        # Test contains
        assert "people" in pickle_storage
        assert "users" in pickle_storage
        assert "nonexistent" not in pickle_storage

    def test_bulk_operations(
        self, pickle_storage: PickleStorage, sample_df: pd.DataFrame,
    ):
        """Test bulk operations."""
        # Store multiple DataFrames
        dataframes = {
            "users": sample_df,
            "orders": sample_df.copy(),
            "products": sample_df.copy(),
        }
        pickle_storage.store_many(dataframes)

        # Load multiple DataFrames
        table_names = ["users", "orders", "products"]
        results = pickle_storage.load_many(table_names)

        assert set(results.keys()) == set(table_names)
        for df in results.values():
            pd.testing.assert_frame_equal(sample_df, df)

        # Delete multiple tables
        pickle_storage.delete_many(table_names)
        assert len(pickle_storage) == 0

    def test_compression_support(
        self, pickle_storage_compressed: PickleStorage, sample_df: pd.DataFrame, temp_dir: Path,
    ):
        """Test compression support."""
        pickle_storage_compressed.store(sample_df, "people")

        # Check that compressed file exists
        pickle_file = temp_dir / "people.pickle.bz2"
        assert pickle_file.exists()

        # Test loading compressed data
        loaded_df = pickle_storage_compressed.load("people")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_custom_file_extension(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test custom file extension."""
        storage = PickleStorage(temp_dir, file_extension=".pkl")
        storage.store(sample_df, "people")

        # Check that file with custom extension exists
        pickle_file = temp_dir / "people.pkl"
        assert pickle_file.exists()

        # Test loading
        loaded_df = storage.load("people")
        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_complex_data_preservation(
        self, pickle_storage: PickleStorage, complex_df: pd.DataFrame,
    ):
        """Test that complex data types are preserved."""
        pickle_storage.store(complex_df, "complex_data")
        loaded_df = pickle_storage.load("complex_data")

        pd.testing.assert_frame_equal(complex_df, loaded_df)
        # Check that dtypes are preserved
        assert loaded_df["text"].dtype == complex_df["text"].dtype
        assert loaded_df["bool"].dtype == complex_df["bool"].dtype
        assert loaded_df["datetime"].dtype == complex_df["datetime"].dtype

    def test_named_index_preservation(
        self, pickle_storage: PickleStorage, complex_df: pd.DataFrame,
    ):
        """Test that named indexes are preserved."""
        pickle_storage.store(complex_df, "data")
        loaded_df = pickle_storage.load("data")

        pd.testing.assert_frame_equal(complex_df, loaded_df)
        assert loaded_df.index.name == complex_df.index.name

    def test_pathlib_support(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test pathlib.Path support."""
        storage = PickleStorage(temp_dir)
        storage.store(sample_df, "people")
        loaded_df = storage.load("people")

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_nonexistent_table_error(self, pickle_storage: PickleStorage):
        """Test error handling for nonexistent tables."""
        with pytest.raises(FileNotFoundError):
            pickle_storage.load("nonexistent_table")


class TestPickleFunctions:
    """Test cases for pickle function interfaces."""

    def test_store_load_functions(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test store and load functions."""
        store_df_pickle(sample_df, "people", str(temp_dir))
        loaded_df = load_df_pickle("people", str(temp_dir))

        pd.testing.assert_frame_equal(sample_df, loaded_df)

    def test_delete_function(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test delete function."""
        store_df_pickle(sample_df, "people", str(temp_dir))
        delete_table_pickle("people", str(temp_dir))

        # Check that file is deleted
        pickle_file = temp_dir / "people.pickle"
        assert not pickle_file.exists()

    def test_table_names_function(self, temp_dir: Path, sample_df: pd.DataFrame):
        """Test table_names function."""
        store_df_pickle(sample_df, "people", str(temp_dir))
        store_df_pickle(sample_df, "users", str(temp_dir))

        table_names = set(table_names_pickle(str(temp_dir)))
        expected = {"people", "users"}
        assert table_names == expected


@pytest.mark.parametrize("compression", [None, "gzip", "bz2", "xz"])
def test_compression_types(compression):
    """Test different compression types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PickleStorage(tmpdir, compression=compression)
        df = pd.DataFrame({"test": [1, 2, 3]})
        storage.store(df, "test")
        loaded_df = storage.load("test")
        pd.testing.assert_frame_equal(df, loaded_df)


@pytest.mark.parametrize("file_extension", [".pickle", ".pkl", ".p"])
def test_file_extensions(file_extension):
    """Test different file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PickleStorage(tmpdir, file_extension=file_extension)
        df = pd.DataFrame({"test": [1, 2, 3]})
        storage.store(df, "test")
        loaded_df = storage.load("test")
        pd.testing.assert_frame_equal(df, loaded_df)


@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations (placeholder)."""
    # This is a placeholder for async testing
    # In a full implementation, this would test AsyncPickleStorage
    pass
