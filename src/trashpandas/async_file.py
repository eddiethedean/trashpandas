"""Async file storage operations for TrashPandas.

This module provides async versions of file-based storage operations
for CSV, pickle, and HDF5 backends using asyncio.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal, Optional, cast

from pandas import DataFrame, read_csv

from trashpandas.interfaces import IAsyncFileStorage
from trashpandas.pickle import _safe_read_pickle
from trashpandas.utils import (
    cast_type,
    convert_meta_to_dict,
    df_metadata,
    name_no_names,
    unname_no_names,
)


class AsyncCsvStorage(IAsyncFileStorage):
    """Async CSV storage backend for DataFrames.

    This class provides async versions of CSV storage operations,
    allowing for concurrent file operations and better performance
    in async applications.
    """

    def __init__(
        self, folder_path: str | Path, compression: str | None = None,
    ) -> None:
        """Initialize async CSV storage.

        Args:
            folder_path: Path to directory for CSV storage
            compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')

        """
        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def __repr__(self) -> str:
        return f"AsyncCsvStorage(path='{self.path}')"

    async def __aenter__(self) -> AsyncCsvStorage:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager."""
        pass

    async def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame and metadata as CSV files asynchronously.

        Args:
            df: DataFrame to store
            table_name: Name of the table
            schema: Optional schema name (ignored for CSV)

        """
        await store_df_csv_async(df, table_name, str(self.path), self.compression)

    async def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame from CSV file asynchronously.

        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name (ignored for CSV)

        Returns:
            The stored DataFrame

        """
        return await load_df_csv_async(table_name, str(self.path), self.compression)

    async def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete DataFrame and metadata CSV files asynchronously.

        Args:
            table_name: Name of the table to delete
            schema: Optional schema name (ignored for CSV)

        """
        await delete_table_csv_async(table_name, str(self.path), self.compression)

    async def load_metadata(
        self, table_name: str, schema: str | None = None,
    ) -> DataFrame:
        """Retrieve DataFrame metadata from CSV file asynchronously.

        Args:
            table_name: Name of the table
            schema: Optional schema name (ignored for CSV)

        Returns:
            DataFrame containing metadata

        """
        return await load_metadata_csv_async(
            table_name, str(self.path), self.compression,
        )

    async def table_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored non-metadata table names asynchronously.

        Args:
            schema: Optional schema name (ignored for CSV)

        Returns:
            List of table names

        """
        return await table_names_csv_async(str(self.path))

    async def metadata_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored metadata table names asynchronously.

        Args:
            schema: Optional schema name (ignored for CSV)

        Returns:
            List of metadata table names

        """
        return await metadata_names_csv_async(str(self.path))

    async def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames as CSV files asynchronously.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name (ignored for CSV)

        """
        tasks = [
            self.store(df, table_name, schema) for table_name, df in dataframes.items()
        ]
        await asyncio.gather(*tasks)

    async def load_many(
        self,
        table_names: list[str],
        schema: str | None = None,
        max_concurrent: int = 10,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames from CSV files asynchronously.

        Includes concurrency control.

        Args:
            table_names: List of table names to load
            schema: Optional schema name (ignored for CSV)
            max_concurrent: Maximum number of concurrent operations (default: 10)

        Returns:
            Dictionary mapping table names to DataFrames

        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def load_with_semaphore(table_name: str) -> tuple[str, DataFrame]:
            async with semaphore:
                return (table_name, await self.load(table_name, schema))

        tasks = [load_with_semaphore(name) for name in table_names]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def delete_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> None:
        """Delete multiple CSV files asynchronously.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name (ignored for CSV)

        """
        tasks = [self.delete(table_name, schema) for table_name in table_names]
        await asyncio.gather(*tasks)


class AsyncPickleStorage(IAsyncFileStorage):
    """Async pickle storage backend for DataFrames.

    This class provides async versions of pickle storage operations,
    allowing for concurrent file operations and better performance
    in async applications.
    """

    def __init__(
        self,
        folder_path: str | Path,
        file_extension: str = ".pickle",
        compression: str | None = None,
    ) -> None:
        """Initialize async pickle storage.

        Args:
            folder_path: Path to directory for pickle storage
            file_extension: File extension for pickle files
            compression: Optional compression type ('gzip', 'bz2', 'xz', 'zstd')

        """
        self.path = Path(folder_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension
        self.compression = compression

    def __repr__(self) -> str:
        return f"AsyncPickleStorage(path='{self.path}')"

    async def __aenter__(self) -> AsyncPickleStorage:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager."""
        pass

    async def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame as pickle file asynchronously.

        Args:
            df: DataFrame to store
            table_name: Name of the table
            schema: Optional schema name (ignored for pickle)

        """
        await store_df_pickle_async(
            df, table_name, str(self.path), self.file_extension, self.compression,
        )

    async def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame from pickle file asynchronously.

        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name (ignored for pickle)

        Returns:
            The stored DataFrame

        """
        return await load_df_pickle_async(
            table_name, str(self.path), self.file_extension, self.compression,
        )

    async def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete DataFrame pickle file asynchronously.

        Args:
            table_name: Name of the table to delete
            schema: Optional schema name (ignored for pickle)

        """
        await delete_table_pickle_async(table_name, str(self.path), self.file_extension)

    async def table_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored table names asynchronously.

        Args:
            schema: Optional schema name (ignored for pickle)

        Returns:
            List of table names

        """
        return await table_names_pickle_async(str(self.path), self.file_extension)

    async def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames as pickle files asynchronously.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name (ignored for pickle)

        """
        tasks = [
            self.store(df, table_name, schema) for table_name, df in dataframes.items()
        ]
        await asyncio.gather(*tasks)

    async def load_many(
        self,
        table_names: list[str],
        schema: str | None = None,
        max_concurrent: int = 10,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames from pickle files asynchronously.

        Includes concurrency control.

        Args:
            table_names: List of table names to load
            schema: Optional schema name (ignored for pickle)
            max_concurrent: Maximum number of concurrent operations (default: 10)

        Returns:
            Dictionary mapping table names to DataFrames

        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def load_with_semaphore(table_name: str) -> tuple[str, DataFrame]:
            async with semaphore:
                return (table_name, await self.load(table_name, schema))

        tasks = [load_with_semaphore(name) for name in table_names]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def delete_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> None:
        """Delete multiple pickle files asynchronously.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name (ignored for pickle)

        """
        tasks = [self.delete(table_name, schema) for table_name in table_names]
        await asyncio.gather(*tasks)


# Async CSV functions
async def store_df_csv_async(
    df: DataFrame, table_name: str, path: str, compression: str | None = None,
) -> None:
    """Store DataFrame and metadata as CSV files asynchronously."""

    def _store() -> None:
        csv_path = _get_csv_path_async(table_name, path, compression)
        metadata_path = _get_metadata_csv_path_async(table_name, path, compression)
        df_copy = df.copy()
        name_no_names(df_copy)
        metadata = df_metadata(df_copy)
        compression_literal = cast(
            "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
        )
        df_copy.to_csv(csv_path, compression=compression_literal)
        metadata.to_csv(metadata_path, index=False, compression=compression_literal)

    await asyncio.get_event_loop().run_in_executor(None, _store)


async def load_df_csv_async(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame from CSV file asynchronously."""

    def _load() -> DataFrame:
        csv_path = _get_csv_path_async(table_name, path, compression)
        metadata_path = _get_metadata_csv_path_async(table_name, path, compression)

        if not os.path.exists(metadata_path):
            return _first_load_df_csv_async_sync(table_name, path, compression)

        metadata = _read_cast_metadata_csv_async_sync(table_name, path, compression)
        types = convert_meta_to_dict(metadata)
        indexes = list(metadata["column"][metadata["index"]])
        compression_literal = cast(
            "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
        )
        df = read_csv(csv_path, compression=compression_literal).astype(types).set_index(indexes)
        unname_no_names(df)
        return df

    return await asyncio.get_event_loop().run_in_executor(None, _load)


async def delete_table_csv_async(
    table_name: str, path: str, compression: str | None = None,
) -> None:
    """Delete DataFrame and metadata CSV files asynchronously."""

    def _delete() -> None:
        csv_path = _get_csv_path_async(table_name, path, compression)
        metadata_path = _get_metadata_csv_path_async(table_name, path, compression)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    await asyncio.get_event_loop().run_in_executor(None, _delete)


async def load_metadata_csv_async(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame metadata from CSV file asynchronously."""

    def _load() -> DataFrame:
        return _read_cast_metadata_csv_async_sync(table_name, path, compression)

    return await asyncio.get_event_loop().run_in_executor(None, _load)


async def table_names_csv_async(path: str) -> list[str]:
    """Get list of stored non-metadata table names asynchronously."""

    def _get_names() -> list[str]:
        filenames = os.listdir(path)
        return [
            filename.split(".csv")[0]
            for filename in filenames
            if filename.endswith(".csv") and "_metadata" not in filename
        ]

    return await asyncio.get_event_loop().run_in_executor(None, _get_names)


async def metadata_names_csv_async(path: str) -> list[str]:
    """Get list of stored metadata table names asynchronously."""

    def _get_names() -> list[str]:
        filenames = os.listdir(path)
        return [
            filename.split(".csv")[0]
            for filename in filenames
            if filename.endswith(".csv") and "_metadata" in filename
        ]

    return await asyncio.get_event_loop().run_in_executor(None, _get_names)


# Async pickle functions
async def store_df_pickle_async(
    df: DataFrame,
    table_name: str,
    path: str,
    file_extension: str = ".pickle",
    compression: str | None = None,
) -> None:
    """Store DataFrame as pickle file asynchronously."""

    def _store() -> None:
        pickle_path = _get_pickle_path_async(
            table_name, path, file_extension, compression,
        )
        compression_literal = cast(
            "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
        )
        df.to_pickle(pickle_path, compression=compression_literal)

    await asyncio.get_event_loop().run_in_executor(None, _store)


async def load_df_pickle_async(
    table_name: str,
    path: str,
    file_extension: str = ".pickle",
    compression: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame from pickle file asynchronously."""

    def _load() -> DataFrame:
        pickle_path = _get_pickle_path_async(
            table_name, path, file_extension, compression,
        )
        compression_literal = cast(
            "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
        )
        return _safe_read_pickle(pickle_path, compression_literal)

    return await asyncio.get_event_loop().run_in_executor(None, _load)


async def delete_table_pickle_async(
    table_name: str, path: str, file_extension: str = ".pickle",
) -> None:
    """Delete DataFrame pickle file asynchronously."""

    def _delete() -> None:
        pickle_path = _get_pickle_path_async(table_name, path, file_extension)
        if os.path.exists(pickle_path):
            os.remove(pickle_path)

    await asyncio.get_event_loop().run_in_executor(None, _delete)


async def table_names_pickle_async(
    path: str, file_extension: str = ".pickle",
) -> list[str]:
    """Get list of stored table names asynchronously."""

    def _get_names() -> list[str]:
        filenames = os.listdir(path)
        return [
            filename.split(file_extension)[0]
            for filename in filenames
            if filename.endswith(file_extension) and "_metadata" not in filename
        ]

    return await asyncio.get_event_loop().run_in_executor(None, _get_names)


# Helper functions
def _get_csv_path_async(
    table_name: str, path: str, compression: str | None = None,
) -> str:
    """Return joined folder path and CSV file name for DataFrame."""
    filename = f"{table_name}.csv"
    if compression:
        filename += f".{compression}"
    return os.path.join(path, filename)


def _get_metadata_csv_path_async(
    table_name: str, path: str, compression: str | None = None,
) -> str:
    """Return joined folder path and CSV file name for DataFrame metadata."""
    filename = f"_{table_name}_metadata.csv"
    if compression:
        filename += f".{compression}"
    return os.path.join(path, filename)


def _get_pickle_path_async(
    table_name: str,
    path: str,
    file_extension: str = ".pickle",
    compression: str | None = None,
) -> str:
    """Return joined folder path and pickle file name for DataFrame."""
    filename = f"{table_name}{file_extension}"
    if compression:
        filename += f".{compression}"
    return os.path.join(path, filename)


def _read_cast_metadata_csv_async_sync(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Load metadata CSV and cast column datatypes (sync version for async)."""
    metadata_path = _get_metadata_csv_path_async(table_name, path, compression)
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    meta = read_csv(metadata_path, compression=compression_literal)
    meta["datatype"] = cast_type(meta["datatype"])
    return meta


def _first_load_df_csv_async_sync(
    table_name: str, path: str, compression: str | None = None,
) -> DataFrame:
    """Load a CSV without metadata, create and store metadata.

    Sync version for async.
    """
    csv_path = _get_csv_path_async(table_name, path, compression)
    compression_literal = cast(
        "Optional[Literal['infer', 'gzip', 'bz2', 'xz', 'zstd']]", compression,
    )
    df = read_csv(csv_path, compression=compression_literal)
    # Store metadata synchronously
    csv_path_meta = _get_metadata_csv_path_async(table_name, path, compression)
    df_copy = df.copy()
    name_no_names(df_copy)
    metadata = df_metadata(df_copy)
    metadata.to_csv(csv_path_meta, index=False, compression=compression_literal)
    return df
