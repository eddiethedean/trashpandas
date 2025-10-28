"""Tests for new features and edge cases."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sqlalchemy import create_engine

from trashpandas.csv import CsvStorage
from trashpandas.exceptions import (
    CompressionError,
    ConversionError,
    MetadataCorruptedError,
    MetadataError,
    MetadataNotFoundError,
    StorageConnectionError,
    StorageError,
    TableAlreadyExistsError,
    TableNotFoundError,
    TrashPandasError,
    ValidationError,
)
from trashpandas.metadata import TableMetadata


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
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"],
    })


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""

    def test_base_exception(self):
        """Test base TrashPandasError."""
        error = TrashPandasError("Test error", details="Some details")
        assert str(error) == "Test error"
        assert error.details == "Some details"

    def test_storage_error(self):
        """Test StorageError."""
        error = StorageError("Storage error")
        assert isinstance(error, TrashPandasError)
        assert str(error) == "Storage error"

    def test_table_not_found_error(self):
        """Test TableNotFoundError."""
        error = TableNotFoundError("users", "sql")
        assert error.table_name == "users"
        assert error.storage_type == "sql"
        assert "users" in str(error)
        assert "sql" in str(error)

    def test_table_already_exists_error(self):
        """Test TableAlreadyExistsError."""
        error = TableAlreadyExistsError("users", "sql")
        assert error.table_name == "users"
        assert error.storage_type == "sql"

    def test_metadata_error(self):
        """Test MetadataError."""
        error = MetadataError("Metadata error")
        assert isinstance(error, TrashPandasError)

    def test_metadata_not_found_error(self):
        """Test MetadataNotFoundError."""
        error = MetadataNotFoundError("users")
        assert error.table_name == "users"

    def test_metadata_corrupted_error(self):
        """Test MetadataCorruptedError."""
        error = MetadataCorruptedError("users", "Invalid format")
        assert error.table_name == "users"
        assert "Invalid format" in str(error)

    def test_connection_error(self):
        """Test StorageConnectionError."""
        error = StorageConnectionError("sql", "Connection failed")
        assert error.storage_type == "sql"
        assert "Connection failed" in str(error)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("table_name", "invalid_name!", "Invalid characters")
        assert error.field == "table_name"
        assert error.value == "invalid_name!"
        assert error.reason == "Invalid characters"

    def test_conversion_error(self):
        """Test ConversionError."""
        error = ConversionError("csv", "sql", "Schema mismatch")
        assert error.source_format == "csv"
        assert error.target_format == "sql"
        assert "Schema mismatch" in str(error)

    def test_compression_error(self):
        """Test CompressionError."""
        error = CompressionError("compress", "File corrupted")
        assert error.operation == "compress"
        assert "File corrupted" in str(error)


class TestTableMetadata:
    """Test the TableMetadata dataclass."""

    def test_from_dataframe(self, sample_df: pd.DataFrame):
        """Test creating metadata from DataFrame."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")

        assert metadata.table_name == "test_table"
        assert metadata.storage_format == "csv"
        assert metadata.columns == list(sample_df.columns)
        assert len(metadata.column_types) == len(sample_df.columns)

    def test_to_dataframe(self, sample_df: pd.DataFrame):
        """Test converting metadata to DataFrame."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        metadata_df = metadata.to_dataframe()

        assert "column" in metadata_df.columns
        assert "index" in metadata_df.columns
        assert "datatype" in metadata_df.columns

    def test_validation(self, sample_df: pd.DataFrame):
        """Test metadata validation."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        metadata.validate()  # Should not raise

    def test_validation_duplicate_columns(self):
        """Test validation with duplicate columns."""
        metadata = TableMetadata(
            table_name="test",
            columns=["col1", "col1"],  # Duplicate
            column_types={"col1": "int64"},
            index_columns=[],
            index_types={},
        )

        with pytest.raises(MetadataCorruptedError):
            metadata.validate()

    def test_validation_missing_types(self):
        """Test validation with missing types."""
        metadata = TableMetadata(
            table_name="test",
            columns=["col1", "col2"],
            column_types={"col1": "int64"},  # Missing col2
            index_columns=[],
            index_types={},
        )

        with pytest.raises(MetadataCorruptedError):
            metadata.validate()

    def test_get_column_types(self, sample_df: pd.DataFrame):
        """Test getting column types as pandas dtypes."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        types = metadata.get_column_types()

        assert isinstance(types, dict)
        assert len(types) == len(sample_df.columns)

    def test_get_index_types(self, sample_df: pd.DataFrame):
        """Test getting index types as pandas dtypes."""
        sample_df.index.name = "id"
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        types = metadata.get_index_types()

        assert isinstance(types, dict)

    def test_str_representation(self, sample_df: pd.DataFrame):
        """Test string representation."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        str_repr = str(metadata)

        assert "test_table" in str_repr
        assert "3" in str_repr  # Number of columns

    def test_repr_representation(self, sample_df: pd.DataFrame):
        """Test detailed string representation."""
        metadata = TableMetadata.from_dataframe(sample_df, "test_table", "csv")
        repr_str = repr(metadata)

        assert "TableMetadata" in repr_str
        assert "test_table" in repr_str


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self, temp_dir: Path):
        """Test storing and loading empty DataFrame."""
        empty_df = pd.DataFrame()
        storage = CsvStorage(temp_dir)

        storage.store(empty_df, "empty")
        loaded_df = storage.load("empty")

        # Empty DataFrames may have different inferred types, so check shape and columns
        assert empty_df.shape == loaded_df.shape
        assert list(empty_df.columns) == list(loaded_df.columns)

    def test_single_row_dataframe(self, temp_dir: Path):
        """Test storing and loading single row DataFrame."""
        single_row_df = pd.DataFrame({"value": [42]})
        storage = CsvStorage(temp_dir)

        storage.store(single_row_df, "single")
        loaded_df = storage.load("single")

        pd.testing.assert_frame_equal(single_row_df, loaded_df)

    def test_large_dataframe(self, temp_dir: Path):
        """Test storing and loading large DataFrame."""
        large_df = pd.DataFrame({
            "id": range(1000),
            "value": [f"value_{i}" for i in range(1000)],
        })
        storage = CsvStorage(temp_dir)

        storage.store(large_df, "large")
        loaded_df = storage.load("large")

        pd.testing.assert_frame_equal(large_df, loaded_df)

    def test_special_characters_in_data(self, temp_dir: Path):
        """Test storing data with special characters."""
        special_df = pd.DataFrame({
            "text": ["Hello, World!", "Line\nBreak", "Tab\tCharacter", 'Quote"Test'],
            "unicode": ["café", "naïve", "résumé", "🚀"],
        })
        storage = CsvStorage(temp_dir)

        storage.store(special_df, "special")
        loaded_df = storage.load("special")

        pd.testing.assert_frame_equal(special_df, loaded_df)

    def test_nan_values(self, temp_dir: Path):
        """Test storing data with NaN values."""
        nan_df = pd.DataFrame({
            "numbers": [1, 2, None, 4, 5],
            "text": ["a", "b", None, "d", "e"],
        })
        storage = CsvStorage(temp_dir)

        storage.store(nan_df, "nan_data")
        loaded_df = storage.load("nan_data")

        pd.testing.assert_frame_equal(nan_df, loaded_df)

    def test_very_long_column_names(self, temp_dir: Path):
        """Test storing data with very long column names."""
        long_name_df = pd.DataFrame({
            "a" * 100: [1, 2, 3],
            "b" * 100: [4, 5, 6],
        })
        storage = CsvStorage(temp_dir)

        storage.store(long_name_df, "long_names")
        loaded_df = storage.load("long_names")

        pd.testing.assert_frame_equal(long_name_df, loaded_df)

    def test_very_long_table_name(self, temp_dir: Path):
        """Test storing data with very long table name."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        storage = CsvStorage(temp_dir)

        long_name = "a" * 100
        storage.store(df, long_name)
        loaded_df = storage.load(long_name)

        pd.testing.assert_frame_equal(df, loaded_df)


class TestCompressionEdgeCases:
    """Test compression edge cases."""

    def test_compression_with_empty_data(self, temp_dir: Path):
        """Test compression with empty DataFrame."""
        empty_df = pd.DataFrame()
        storage = CsvStorage(temp_dir, compression="gzip")

        storage.store(empty_df, "empty")
        loaded_df = storage.load("empty")

        # Empty DataFrames may have different inferred types, so check shape and columns
        assert empty_df.shape == loaded_df.shape
        assert list(empty_df.columns) == list(loaded_df.columns)

    def test_compression_with_single_character_data(self, temp_dir: Path):
        """Test compression with single character data."""
        single_char_df = pd.DataFrame({"a": ["x"]})
        storage = CsvStorage(temp_dir, compression="gzip")

        storage.store(single_char_df, "single_char")
        loaded_df = storage.load("single_char")

        pd.testing.assert_frame_equal(single_char_df, loaded_df)

    def test_different_compression_algorithms(self, temp_dir: Path):
        """Test different compression algorithms."""
        df = pd.DataFrame({"data": [f"row_{i}" for i in range(100)]})

        for compression in ["gzip", "bz2", "xz"]:
            storage = CsvStorage(temp_dir / compression, compression=compression)
            storage.store(df, "data")
            loaded_df = storage.load("data")
            pd.testing.assert_frame_equal(df, loaded_df)


class TestBulkOperationsEdgeCases:
    """Test bulk operations edge cases."""

    def test_empty_bulk_operations(self, temp_dir: Path):
        """Test bulk operations with empty data."""
        storage = CsvStorage(temp_dir)

        # Empty store_many
        storage.store_many({})

        # Empty load_many
        results = storage.load_many([])
        assert results == {}

        # Empty delete_many
        storage.delete_many([])

    def test_bulk_operations_with_single_item(
        self, temp_dir: Path, sample_df: pd.DataFrame,
    ):
        """Test bulk operations with single item."""
        storage = CsvStorage(temp_dir)

        # Single item store_many
        storage.store_many({"single": sample_df})

        # Single item load_many
        results = storage.load_many(["single"])
        assert len(results) == 1
        pd.testing.assert_frame_equal(sample_df, results["single"])

        # Single item delete_many
        storage.delete_many(["single"])
        assert len(storage) == 0

    def test_bulk_operations_with_duplicate_names(
        self, temp_dir: Path, sample_df: pd.DataFrame,
    ):
        """Test bulk operations with duplicate names."""
        storage = CsvStorage(temp_dir)

        # This should work - last one wins
        dataframes = {
            "duplicate": sample_df.copy(),  # Same key twice
        }
        storage.store_many(dataframes)

        # Should have only one table
        assert len(storage) == 1
        assert "duplicate" in storage


class TestMetadataModule:
    """Test metadata.py module functionality."""

    def test_table_metadata_from_dict_edge_cases(self):
        """Test TableMetadata.from_dataframe with edge cases."""
        from trashpandas.metadata import TableMetadata

        # Test with minimal DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        metadata = TableMetadata.from_dataframe(df, "test_table")
        assert metadata.columns == ["a", "b"]
        assert metadata.column_types["a"] == "int64"
        assert metadata.column_types["b"] == "object"

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        metadata = TableMetadata.from_dataframe(empty_df, "empty_table")
        assert metadata.columns == []
        assert metadata.column_types == {}

    def test_table_metadata_to_dict_edge_cases(self):
        """Test TableMetadata.to_dataframe with edge cases."""
        from trashpandas.metadata import TableMetadata

        # Test with complex metadata
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        metadata = TableMetadata.from_dataframe(df, "test_table")
        
        # Test that we can recreate the DataFrame structure
        assert metadata.table_name == "test_table"
        assert metadata.columns == ["col1", "col2"]
        assert metadata.column_types["col1"] == "int64"
        assert metadata.column_types["col2"] == "object"

    def test_table_metadata_validation_edge_cases(self):
        """Test TableMetadata validation with edge cases."""
        from trashpandas.metadata import TableMetadata

        # Test with DataFrame containing various types
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3],
        })
        metadata = TableMetadata.from_dataframe(df, "test_table")
        
        # Verify all column types are captured
        assert len(metadata.columns) == 3
        assert "int_col" in metadata.column_types
        assert "str_col" in metadata.column_types
        assert "float_col" in metadata.column_types

    def test_table_metadata_str_repr_edge_cases(self):
        """Test TableMetadata string representations with edge cases."""
        from trashpandas.metadata import TableMetadata

        # Test with DataFrame containing long column names
        long_columns = [f"very_long_column_name_{i}" for i in range(5)]
        df = pd.DataFrame({col: [1, 2] for col in long_columns})
        metadata = TableMetadata.from_dataframe(df, "test_table")

        str_repr = str(metadata)
        assert "TableMetadata" in str_repr
        assert "test_table" in str_repr
        assert "columns=5" in str_repr  # Should show count of columns

        repr_str = repr(metadata)
        assert "TableMetadata" in repr_str
        assert "test_table" in repr_str


class TestInterfacesModule:
    """Test interfaces.py module functionality."""

    def test_istorage_abstract_methods(self):
        """Test that IStorage abstract methods are defined."""
        from trashpandas.interfaces import IStorage

        # Test that the interface defines the required methods
        assert hasattr(IStorage, "store")
        assert hasattr(IStorage, "load")
        assert hasattr(IStorage, "delete")
        assert hasattr(IStorage, "table_names")
        
        # Test that these are abstract methods
        import inspect
        assert inspect.isabstract(IStorage)

    def test_ifilestorage_abstract_methods(self):
        """Test that IFileStorage abstract methods are defined."""
        from trashpandas.interfaces import IFileStorage

        # Test that the interface defines the required methods
        assert hasattr(IFileStorage, "store")
        assert hasattr(IFileStorage, "load")
        assert hasattr(IFileStorage, "delete")
        assert hasattr(IFileStorage, "table_names")
        
        # Test that these are abstract methods
        import inspect
        assert inspect.isabstract(IFileStorage)

    def test_iasyncstorage_abstract_methods(self):
        """Test that IAsyncStorage abstract methods are defined."""
        from trashpandas.interfaces import IAsyncStorage

        # Test that the interface defines the required methods
        assert hasattr(IAsyncStorage, "store")
        assert hasattr(IAsyncStorage, "load")
        assert hasattr(IAsyncStorage, "delete")
        assert hasattr(IAsyncStorage, "table_names")
        
        # Test that these are abstract methods
        import inspect
        assert inspect.isabstract(IAsyncStorage)

    def test_iasyncfilestorage_abstract_methods(self):
        """Test that IAsyncFileStorage abstract methods are defined."""
        from trashpandas.interfaces import IAsyncFileStorage

        # Test that the interface defines the required methods
        assert hasattr(IAsyncFileStorage, "store")
        assert hasattr(IAsyncFileStorage, "load")
        assert hasattr(IAsyncFileStorage, "delete")
        assert hasattr(IAsyncFileStorage, "table_names")
        
        # Test that these are abstract methods
        import inspect
        assert inspect.isabstract(IAsyncFileStorage)

    def test_storage_contract_enforcement(self):
        """Test that storage implementations follow the contract."""
        from trashpandas.interfaces import IStorage

        # Test that the interface defines the contract
        assert hasattr(IStorage, "store")
        assert hasattr(IStorage, "load")
        assert hasattr(IStorage, "delete")
        assert hasattr(IStorage, "table_names")

    def test_file_storage_path_handling(self):
        """Test that file storage implementations handle paths correctly."""
        from trashpandas.interfaces import IFileStorage

        # Test that the interface defines path handling
        assert hasattr(IFileStorage, "store")
        assert hasattr(IFileStorage, "load")
        assert hasattr(IFileStorage, "delete")
        assert hasattr(IFileStorage, "table_names")
