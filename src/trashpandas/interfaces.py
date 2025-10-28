"""Storage interface definitions for TrashPandas."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from pandas import DataFrame


class IStorage(abc.ABC):
    """Abstract base class for all storage backends.

    This interface defines the contract that all storage implementations
    must follow, ensuring consistent behavior across different storage types.
    """

    @abc.abstractmethod
    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame using dictionary-like syntax.

        Args:
            key: Table name to store the DataFrame as
            other: DataFrame to store

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve stored DataFrame using dictionary-like syntax.

        Args:
            key: Table name to retrieve

        Returns:
            The stored DataFrame

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete stored DataFrame using dictionary-like syntax.

        Args:
            key: Table name to delete

        """
        raise NotImplementedError

    @abc.abstractmethod
    def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame.

        Args:
            df: DataFrame to store
            table_name: Name to store the DataFrame as
            schema: Optional schema name (for SQL backends)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve stored DataFrame.

        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name (for SQL backends)

        Returns:
            The stored DataFrame

        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete stored DataFrame.

        Args:
            table_name: Name of the table to delete
            schema: Optional schema name (for SQL backends)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def table_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored table names.

        Args:
            schema: Optional schema name (for SQL backends)

        Returns:
            List of table names

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over table names.

        Returns:
            Iterator of table names

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get number of stored tables.

        Returns:
            Number of tables

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        """Check if table exists.

        Args:
            key: Table name to check

        Returns:
            True if table exists, False otherwise

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __enter__(self) -> IStorage:
        """Enter context manager.

        Returns:
            Self for use in with statements

        """
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        """
        raise NotImplementedError


class IFileStorage(IStorage):
    """Interface for file-based storage backends."""

    @abc.abstractmethod
    def __init__(self, path: str | Path) -> None:
        """Initialize file storage.

        Args:
            path: Path to storage location

        """
        raise NotImplementedError


class ISqlStorage(IStorage):
    """Interface for SQL-based storage backends."""

    @abc.abstractmethod
    def load_metadata(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Load metadata for a table.

        Args:
            table_name: Name of the table
            schema: Optional schema name

        Returns:
            DataFrame containing metadata

        """
        raise NotImplementedError

    @abc.abstractmethod
    def metadata_names(self, schema: str | None = None) -> list[str]:
        """Get list of metadata table names.

        Args:
            schema: Optional schema name

        Returns:
            List of metadata table names

        """
        raise NotImplementedError


class IAsyncStorage(abc.ABC):
    """Abstract base class for async storage backends.

    This interface defines the contract for asynchronous storage implementations,
    ensuring consistent async/await patterns across different storage types.
    """

    @abc.abstractmethod
    async def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame asynchronously.

        Args:
            df: DataFrame to store
            table_name: Name to store the DataFrame as
            schema: Optional schema name (for SQL backends)

        """
        raise NotImplementedError

    @abc.abstractmethod
    async def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve stored DataFrame asynchronously.

        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name (for SQL backends)

        Returns:
            The stored DataFrame

        """
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete stored DataFrame asynchronously.

        Args:
            table_name: Name of the table to delete
            schema: Optional schema name (for SQL backends)

        """
        raise NotImplementedError

    @abc.abstractmethod
    async def table_names(self, schema: str | None = None) -> list[str]:
        """Get list of stored table names asynchronously.

        Args:
            schema: Optional schema name (for SQL backends)

        Returns:
            List of table names

        """
        raise NotImplementedError

    async def __aenter__(self) -> IAsyncStorage:
        """Enter async context manager.

        Returns:
            Self for use in async with statements

        """
        return self

    @abc.abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        """
        pass

    async def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames asynchronously.

        Default implementation stores sequentially. Subclasses can override
        for parallel/optimized bulk operations.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name

        """
        for table_name, df in dataframes.items():
            await self.store(df, table_name, schema)

    async def load_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames asynchronously.

        Default implementation loads sequentially. Subclasses can override
        for parallel/optimized bulk operations.

        Args:
            table_names: List of table names to load
            schema: Optional schema name

        Returns:
            Dictionary mapping table names to DataFrames

        """
        result = {}
        for table_name in table_names:
            result[table_name] = await self.load(table_name, schema)
        return result

    async def delete_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> None:
        """Delete multiple tables asynchronously.

        Default implementation deletes sequentially. Subclasses can override
        for parallel/optimized bulk operations.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name

        """
        for table_name in table_names:
            await self.delete(table_name, schema)


class IAsyncFileStorage(IAsyncStorage):
    """Interface for async file-based storage backends."""

    pass
