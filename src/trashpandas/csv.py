"""CSV storage backend for TrashPandas DataFrames.

This module provides CSV-based storage for Pandas DataFrames with
metadata preservation.
It supports optional compression and maintains DataFrame structure
including indexes and data types.

Key Features:
    - Store DataFrames as CSV files with metadata preservation
    - Optional compression support (gzip, bz2, xz, zstd)
    - Context manager support for automatic resource cleanup
    - Iterator protocol for easy table enumeration
    - Bulk operations for efficient batch processing
    - Pathlib.Path support for modern path handling

Examples:
    Basic usage:
        >>> import pandas as pd
        >>> import trashpandas as tp
        >>>
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>>
        >>> # Using context manager (recommended)
        >>> with tp.CsvStorage('./data') as storage:
        ...     storage['users'] = df
        ...     loaded_df = storage['users']
        ...     print(f"Stored {len(storage)} tables")

    With compression:
        >>> storage = tp.CsvStorage('./data', compression='gzip')
        >>> storage.store(df, 'users')  # Creates users.csv.gz

    Using pathlib.Path:
        >>> from pathlib import Path
        >>> storage = tp.CsvStorage(Path('./data'))
        >>> storage.store(df, 'users')

    Bulk operations:
        >>> dataframes = {'users': users_df, 'orders': orders_df}
        >>> storage.store_many(dataframes)
        >>> results = storage.load_many(['users', 'orders'])
        >>> storage.delete_many(['users', 'orders'])

Raises:
    FileNotFoundError: When trying to load a non-existent CSV file
    PermissionError: When unable to write to the specified directory
    ValidationError: When table names contain invalid characters

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Literal, Optional, cast

from pandas import DataFrame, read_csv

from trashpandas.interfaces import IFileStorage
from trashpandas.utils import (
    cast_type,
    convert_meta_to_dict,
    df_metadata,
    name_no_names,
    unname_no_names,
)
from trashpandas.validation import validate_table_name


class CsvStorage(IFileStorage):
    def __init__(
        self, folder_path: str | Path, compression: str | None = None,
    ) -> None:
        """Initialize CSV storage with a folder path for DataFrames.

        DataFrames are stored as CSV files.

        Args:
            folder_path: Path to directory for CSV storage
            compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')

        """
        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def __repr__(self) -> str:
        return f"CsvStorage(path='{self.path}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame and metadata as csv files."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from csv file."""
        return self.load(key)

    def __delitem__(self, key: str) -> None:
        """Delete DataFrame and metadata csv files."""
        self.delete(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over table names."""
        return iter(self.table_names())

    def __len__(self) -> int:
        """Get number of stored tables."""
        return len(self.table_names())

    def __contains__(self, key: str) -> bool:
        """Check if table exists."""
        return key in self.table_names()

    def __enter__(self) -> CsvStorage:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit context manager."""
        pass

    def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame and metadata as csv files."""
        store_df_csv(df, table_name, str(self.path), self.compression)

    def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame from csv file."""
        return load_df_csv(table_name, str(self.path), self.compression)

    def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete DataFrame and metadata csv files."""
        delete_table_csv(table_name, str(self.path), self.compression)

    def load_metadata(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame metadata from csv file."""
        return load_metadata_csv(table_name, str(self.path), self.compression)

    def table_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored non-metadata table names."""
        return table_names_csv(str(self.path))

    def metadata_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored metadata table names."""
        return metadata_names_csv(str(self.path))

    def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames as CSV files.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name (ignored for CSV)

        """
        for table_name, df in dataframes.items():
            store_df_csv(df, table_name, str(self.path), self.compression)

    def load_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames from CSV files.

        Args:
            table_names: List of table names to load
            schema: Optional schema name (ignored for CSV)

        Returns:
            Dictionary mapping table names to DataFrames

        """
        result = {}
        for table_name in table_names:
            result[table_name] = self.load(table_name, schema=schema)
        return result

    def delete_many(self, table_names: list[str], schema: str | None = None) -> None:
        """Delete multiple CSV files.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name (ignored for CSV)

        """
        for table_name in table_names:
            delete_table_csv(table_name, str(self.path), self.compression)


def store_df_csv(
    df: DataFrame, table_name: str, path: str, compression: str | None = None,
) -> None:
    """Store DataFrame and metadata as csv files.

    Args:
        df: DataFrame to store
        table_name: Name of the table
        path: Directory path for storage
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')

    """
    validate_table_name(table_name, storage_type="csv")
    csv_path = _get_csv_path(table_name, path, compression)
    metadata_path = _get_metadata_csv_path(table_name, path, compression)
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    # Cast compression to proper Literal type expected by pandas
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    df.to_csv(csv_path, compression=compression_literal)
    metadata.to_csv(metadata_path, index=False, compression=compression_literal)


def load_df_csv(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame from csv file.

    Args:
        table_name: Name of the table
        path: Directory path for storage
        compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')

    Returns:
        Loaded DataFrame

    """
    validate_table_name(table_name, storage_type="csv")
    csv_path = _get_csv_path(table_name, path, compression)
    metadata_path = _get_metadata_csv_path(table_name, path, compression)

    if not os.path.exists(metadata_path):
        return _first_load_df_csv(table_name, path, compression)

    metadata = _read_cast_metadata_csv(table_name, path, compression)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata["column"][metadata["index"]])
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    df = (
        read_csv(csv_path, compression=compression_literal)
        .astype(types)
        .set_index(indexes)
    )
    unname_no_names(df)
    return df


def delete_table_csv(
    table_name: str, path: str, compression: str | None = None,
) -> None:
    """Delete DataFrame and metadata csv files.

    Args:
        table_name: Name of the table
        path: Directory path
        compression: Optional compression type (must match what was used during storage)

    """
    validate_table_name(table_name, storage_type="csv")
    csv_path = _get_csv_path(table_name, path, compression)
    metadata_path = _get_metadata_csv_path(table_name, path, compression)

    # Remove files if they exist
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)


def load_metadata_csv(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame metadata from CSV file.

    Args:
        table_name: Name of the table
        path: Directory path for storage
        compression: Optional compression type (must match storage compression)

    Returns:
        DataFrame containing metadata

    """
    meta_name = f"_{table_name}_metadata"
    return _read_cast_metadata_csv(meta_name, path, compression)


def table_names_csv(path: str) -> list[str]:
    """Get list of stored non-metadata table names."""
    filenames = os.listdir(path)
    table_names = []
    for filename in filenames:
        # Check for both uncompressed and compressed CSV files
        if "_metadata" not in filename and ".csv" in filename:
            # Extract table name (everything before .csv)
            table_name = filename.split(".csv")[0]
            if table_name and table_name not in table_names:
                table_names.append(table_name)
    return table_names


def metadata_names_csv(path: str) -> list[str]:
    """Get list of stored metadata table names."""
    filenames = os.listdir(path)
    metadata_names = []
    for filename in filenames:
        # Check for both uncompressed and compressed metadata CSV files
        if "_metadata" in filename and ".csv" in filename:
            # Extract metadata name (everything before .csv)
            metadata_name = filename.split(".csv")[0]
            if metadata_name and metadata_name not in metadata_names:
                metadata_names.append(metadata_name)
    return metadata_names


def _get_csv_path(table_name: str, path: str, compression: str | None = None) -> str:
    """Return joined folder path and csv file name for DataFrame."""
    filename = f"{table_name}.csv"
    if compression:
        filename += f".{compression}"
    return os.path.join(path, filename)


def _get_metadata_csv_path(
    table_name: str, path: str, compression: str | None = None,
) -> str:
    """Return joined folder path and csv file name for DataFrame metadata."""
    filename = f"_{table_name}_metadata.csv"
    if compression:
        filename += f".{compression}"
    return os.path.join(path, filename)


def _read_cast_metadata_csv(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Load metadata csv and cast column datatypes column."""
    metadata_path = _get_metadata_csv_path(table_name, path, compression)
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    meta = read_csv(metadata_path, compression=compression_literal)
    meta["datatype"] = cast_type(meta["datatype"])
    return meta


def _first_load_df_csv(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Load a csv that has no metadata stored, create and store metadata."""
    csv_path = _get_csv_path(table_name, path, compression)
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    df = read_csv(csv_path, compression=compression_literal)
    store_df_csv(df, table_name, path, compression)
    return df
