"""Async SQL storage operations for TrashPandas.

This module provides async versions of SQL storage operations using SQLAlchemy's
async engine and connection capabilities.
"""

from __future__ import annotations

import asyncio

from pandas import DataFrame, read_sql_table
from sqlalchemy import Connection, MetaData, Table, inspect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.schema import DropTable

from trashpandas.interfaces import IAsyncStorage
from trashpandas.utils import (
    cast_type,
    convert_meta_to_dict,
    df_metadata,
    name_no_names,
    unname_no_names,
)


class AsyncSqlStorage(IAsyncStorage):
    """Async SQL storage backend for DataFrames.

    This class provides async versions of all SQL storage operations,
    allowing for concurrent database operations and better performance
    in async applications.
    """

    def __init__(self, engine: AsyncEngine | str) -> None:
        """Initialize async SQL storage.

        Args:
            engine: SQLAlchemy async engine or database connection string

        """
        if isinstance(engine, str):
            # Convert sync URL to async if needed
            if engine.startswith("sqlite://"):
                engine = engine.replace("sqlite://", "sqlite+aiosqlite://")
            elif engine.startswith("postgresql://"):
                engine = engine.replace("postgresql://", "postgresql+asyncpg://")
            elif engine.startswith("mysql://"):
                engine = engine.replace("mysql://", "mysql+aiomysql://")

            engine = create_async_engine(engine)
        self.engine = engine

    def __repr__(self) -> str:
        return f"AsyncSqlStorage('{self.engine.url}')"

    async def __aenter__(self) -> AsyncSqlStorage:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager."""
        await self.engine.dispose()

    async def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame and metadata in database asynchronously.

        Args:
            df: DataFrame to store
            table_name: Name of the table
            schema: Optional schema name

        """
        await store_df_sql_async(df, table_name, self.engine, schema=schema)

    async def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame from database asynchronously.

        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name

        Returns:
            The stored DataFrame

        """
        return await load_df_sql_async(table_name, self.engine, schema=schema)

    async def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete DataFrame and metadata from database asynchronously.

        Args:
            table_name: Name of the table to delete
            schema: Optional schema name

        """
        await delete_table_sql_async(table_name, self.engine, schema)

    async def load_metadata(
        self, table_name: str, schema: str | None = None,
    ) -> DataFrame:
        """Retrieve DataFrame metadata from database asynchronously.

        Args:
            table_name: Name of the table
            schema: Optional schema name

        Returns:
            DataFrame containing metadata

        """
        return await load_metadata_sql_async(table_name, self.engine, schema=schema)

    async def table_names(self, schema: str | None = None) -> list[str]:
        """Query database for list of non-metadata table names asynchronously.

        Args:
            schema: Optional schema name

        Returns:
            List of table names

        """
        return await table_names_sql_async(self.engine, schema=schema)

    async def metadata_names(self, schema: str | None = None) -> list[str]:
        """Query database for list of metadata table names asynchronously.

        Args:
            schema: Optional schema name

        Returns:
            List of metadata table names

        """
        return await metadata_names_sql_async(self.engine, schema=schema)

    async def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames in a single transaction asynchronously.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name

        """
        async with self.engine.begin():
            for table_name, df in dataframes.items():
                await store_df_sql_async(df, table_name, self.engine, schema=schema)

    async def load_many(
        self,
        table_names: list[str],
        schema: str | None = None,
        max_concurrent: int = 10,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames asynchronously with concurrency control.

        Args:
            table_names: List of table names to load
            schema: Optional schema name
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
        """Delete multiple tables in a single transaction asynchronously.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name

        """
        async with self.engine.begin():
            for table_name in table_names:
                await delete_table_sql_async(table_name, self.engine, schema)


async def store_df_sql_async(
    df: DataFrame, table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> None:
    """Store DataFrame and metadata in database asynchronously.

    Args:
        df: DataFrame to store
        table_name: Name of the table
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    """
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)

    async with engine.begin() as conn:
        # Store main DataFrame
        await conn.run_sync(
            lambda sync_conn: df.to_sql(
                table_name, sync_conn, if_exists="replace", schema=schema,
            ),
        )

        # Store metadata
        await conn.run_sync(
            lambda sync_conn: metadata.to_sql(
                f"_{table_name}_metadata",
                sync_conn,
                if_exists="replace",
                index=False,
                schema=schema,
            ),
        )


async def load_df_sql_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame from database asynchronously.

    Args:
        table_name: Name of the table to retrieve
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        The stored DataFrame

    """
    meta_name = f"_{table_name}_metadata"

    # Check if metadata exists
    metadata_names = await metadata_names_sql_async(engine, schema=schema)
    if meta_name not in metadata_names:
        return await _first_load_df_sql_async(table_name, engine, schema=schema)

    # Load metadata and DataFrame
    metadata = await _read_cast_metadata_sql_async(meta_name, engine, schema)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata["column"][metadata["index"]])

    async with engine.begin() as conn:
        df = await conn.run_sync(
            lambda sync_conn: read_sql_table(table_name, sync_conn, schema=schema)
            .astype(types)
            .set_index(indexes),
        )

    unname_no_names(df)
    return df


async def delete_table_sql_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> None:
    """Delete DataFrame and metadata from database asynchronously.

    Args:
        table_name: Name of the table to delete
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    """
    table = await _get_table_async(table_name, engine, schema)
    metadata = await _get_table_async(f"_{table_name}_metadata", engine, schema)

    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: sync_conn.execute(DropTable(table)))
        await conn.run_sync(lambda sync_conn: sync_conn.execute(DropTable(metadata)))


async def load_metadata_sql_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame metadata from database asynchronously.

    Args:
        table_name: Name of the table
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        DataFrame containing metadata

    """
    meta_name = f"_{table_name}_metadata"
    return await _read_cast_metadata_sql_async(meta_name, engine, schema=schema)


async def table_names_sql_async(
    engine: AsyncEngine, schema: str | None = None,
) -> list[str]:
    """Query database for list of non-metadata table names asynchronously.

    Args:
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        List of table names

    """
    async with engine.begin() as conn:
        inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
        table_names = await conn.run_sync(
            lambda sync_conn: inspector.get_table_names(schema=schema),
        )
        return [
            name for name in table_names if "_metadata" not in name and name[0] != "_"
        ]


async def metadata_names_sql_async(
    engine: AsyncEngine, schema: str | None = None,
) -> list[str]:
    """Query database for list of metadata table names asynchronously.

    Args:
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        List of metadata table names

    """
    async with engine.begin() as conn:
        inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
        table_names = await conn.run_sync(
            lambda sync_conn: inspector.get_table_names(schema=schema),
        )
        return [name for name in table_names if "_metadata" in name and name[0] == "_"]


async def _read_cast_metadata_sql_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> DataFrame:
    """Load metadata table and cast column datatypes asynchronously.

    Args:
        table_name: Name of the metadata table
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        DataFrame with cast metadata

    """
    async with engine.begin() as conn:
        meta = await conn.run_sync(
            lambda sync_conn: read_sql_table(table_name, sync_conn, schema=schema),
        )
        meta["datatype"] = cast_type(meta["datatype"])
        return meta


async def _first_load_df_sql_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> DataFrame:
    """Load a SQL table without metadata, create and store metadata.

    Performs operations asynchronously.

    Args:
        table_name: Name of the table
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        DataFrame with metadata stored

    """
    async with engine.begin() as conn:
        df = await conn.run_sync(
            lambda sync_conn: read_sql_table(table_name, sync_conn, schema=schema),
        )

    await store_df_sql_async(df, table_name, engine, schema)
    return df


async def _get_table_async(
    table_name: str, engine: AsyncEngine, schema: str | None = None,
) -> Table:
    """Get SQLAlchemy Table mapped to database table asynchronously.

    Uses modern SQLAlchemy 2.x API without deprecated parameters.

    Args:
        table_name: Name of the table
        engine: Async SQLAlchemy engine
        schema: Optional schema name

    Returns:
        SQLAlchemy Table object

    """
    async with engine.begin() as conn:

        def _get_table(sync_conn: Connection) -> Table:
            # Use modern SQLAlchemy 2.x API - avoid deprecated bind parameter
            metadata = MetaData(schema=schema)
            return Table(
                table_name,
                metadata,
                autoload_with=sync_conn,  # Use autoload_with instead of bind + autoload
                schema=schema,
            )

        return await conn.run_sync(_get_table)
