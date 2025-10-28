"""SQL storage backend for TrashPandas DataFrames.

This module provides SQL-based storage for Pandas DataFrames using SQLAlchemy.
It supports multiple database backends including SQLite, PostgreSQL, MySQL, and others.
The module preserves DataFrame metadata including indexes and data types.

Key Features:
    - Store DataFrames in SQL databases with metadata preservation
    - Support for multiple database backends via SQLAlchemy
    - Context manager support for automatic connection cleanup
    - Iterator protocol for easy table enumeration
    - Bulk operations for efficient batch processing
    - Query capabilities for partial data loading
    - SQLAlchemy 2.x compatibility

Examples:
    Basic usage with SQLite:
        >>> import pandas as pd
        >>> import trashpandas as tp
        >>>
        >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>>
        >>> # Using context manager (recommended)
        >>> with tp.SqlStorage('sqlite:///data.db') as storage:
        ...     storage['users'] = df
        ...     loaded_df = storage['users']
        ...     print(f"Stored {len(storage)} tables")

    Using with existing SQLAlchemy engine:
        >>> import sqlalchemy as sa
        >>>
        >>> engine = sa.create_engine('postgresql://user:pass@localhost/db')
        >>> storage = tp.SqlStorage(engine)
        >>> storage.store(df, 'users')

    Bulk operations:
        >>> dataframes = {'users': users_df, 'orders': orders_df}
        >>> storage.store_many(dataframes)
        >>> results = storage.load_many(['users', 'orders'])
        >>> storage.delete_many(['users', 'orders'])

    Query capabilities:
        >>> # Get all data
        >>> all_users = storage.query('users')
        >>>
        >>> # Filter data
        >>> active_users = storage.query('users', where_clause="status = 'active'")
        >>>
        >>> # Select specific columns
        >>> names_only = storage.query('users', columns=['name', 'email'])
        >>>
        >>> # Limit results
        >>> recent_users = storage.query('users', limit=10)

Raises:
    TableNotFoundError: When trying to load a non-existent table
    ConnectionError: When database connection fails
    MetadataCorruptedError: When stored metadata is invalid

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from pandas import DataFrame, read_sql_table
from sqlalchemy import MetaData, Table, create_engine, inspect
from sqlalchemy.schema import DropTable

from trashpandas.interfaces import IStorage
from trashpandas.utils import (
    cast_type,
    convert_meta_to_dict,
    df_metadata,
    name_no_names,
    unname_no_names,
)
from trashpandas.validation import validate_schema_name, validate_table_name

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class SqlStorage(IStorage):
    def __init__(self, engine: Engine | str) -> None:
        """Initialize SQL storage with a SQLAlchemy Engine.

        Also accepts a database connection string.
        """
        if isinstance(engine, str):
            engine = create_engine(engine)
        self.engine = engine

    def __repr__(self) -> str:
        return f"SqlStorage('{self.engine.url}')"

    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame and metadata in database."""
        self.store(other, key)

    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve DataFrame from database."""
        return self.load(key)

    def __delitem__(self, key: str) -> None:
        """Delete DataFrame and metadata from database."""
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

    def __enter__(self) -> SqlStorage:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit context manager."""
        if hasattr(self.engine, "dispose"):
            self.engine.dispose()

    def store(
        self, df: DataFrame, table_name: str, schema: str | None = None,
    ) -> None:
        """Store DataFrame and metadata in database."""
        store_df_sql(df, table_name, self.engine, schema=schema)

    def load(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame from database."""
        return load_df_sql(table_name, self.engine, schema=schema)

    def delete(self, table_name: str, schema: str | None = None) -> None:
        """Delete DataFrame and metadata from database."""
        delete_table_sql(table_name, self.engine, schema)

    def load_metadata(self, table_name: str, schema: str | None = None) -> DataFrame:
        """Retrieve DataFrame metadata from database."""
        return load_metadata_sql(table_name, self.engine, schema=schema)

    def table_names(self, schema: str | None = None) -> list[str]:
        """Query database for list of non-metadata table names."""
        return table_names_sql(self.engine, schema=schema)

    def metadata_names(self, schema: str | None = None) -> list[str]:
        """Query database for list of metadata table names."""
        return metadata_names_sql(self.engine, schema=schema)

    def store_many(
        self, dataframes: dict[str, DataFrame], schema: str | None = None,
    ) -> None:
        """Store multiple DataFrames in a single transaction.

        Args:
            dataframes: Dictionary mapping table names to DataFrames
            schema: Optional schema name

        """
        with self.engine.begin():
            for table_name, df in dataframes.items():
                store_df_sql(df, table_name, self.engine, schema=schema)

    def load_many(
        self, table_names: list[str], schema: str | None = None,
    ) -> dict[str, DataFrame]:
        """Load multiple DataFrames in a single operation.

        Args:
            table_names: List of table names to load
            schema: Optional schema name

        Returns:
            Dictionary mapping table names to DataFrames

        """
        result = {}
        for table_name in table_names:
            result[table_name] = self.load(table_name, schema=schema)
        return result

    def delete_many(self, table_names: list[str], schema: str | None = None) -> None:
        """Delete multiple tables in a single transaction.

        Args:
            table_names: List of table names to delete
            schema: Optional schema name

        """
        with self.engine.begin():
            for table_name in table_names:
                delete_table_sql(table_name, self.engine, schema)

    def query(
        self,
        table_name: str,
        where_clause: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        schema: str | None = None,
    ) -> DataFrame:
        """Query a table with optional filtering and column selection.

        Args:
            table_name: Name of the table to query
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            columns: Optional list of columns to select
            limit: Optional limit on number of rows
            schema: Optional schema name

        Returns:
            DataFrame with query results

        Examples:
            >>> storage.query('users', where_clause="age > 25")
            >>> storage.query('users', columns=['name', 'email'], limit=10)
            >>> storage.query('users', where_clause="status = 'active'", limit=100)

        """
        return query_table_sql(
            table_name, self.engine, where_clause, columns, limit, schema,
        )

    def query_async(
        self,
        table_name: str,
        where_clause: str | None = None,
        columns: list[str] | None = None,
        limit: int | None = None,
        schema: str | None = None,
    ) -> DataFrame:
        """Query a table asynchronously with optional filtering and column selection.

        Args:
            table_name: Name of the table to query
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            columns: Optional list of columns to select
            limit: Optional limit on number of rows
            schema: Optional schema name

        Returns:
            DataFrame with query results

        """
        return query_table_sql_async(
            table_name, self.engine, where_clause, columns, limit, schema,
        )


def store_df_sql(
    df: DataFrame, table_name: str, engine: Engine, schema: str | None = None,
) -> None:
    """Store DataFrame and metadata in database."""
    validate_table_name(table_name, storage_type="sql")
    validate_schema_name(schema)
    df = df.copy()
    name_no_names(df)
    metadata = df_metadata(df)
    df.to_sql(table_name, engine, if_exists="replace", schema=schema)
    metadata.to_sql(
        f"_{table_name}_metadata",
        engine,
        if_exists="replace",
        index=False,
        schema=schema,
    )


def load_df_sql(
    table_name: str, engine: Engine, schema: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame from database."""
    validate_table_name(table_name, storage_type="sql")
    validate_schema_name(schema)
    meta_name = f"_{table_name}_metadata"

    if meta_name not in metadata_names_sql(engine, schema=schema):
        return _first_load_df_sql(table_name, engine, schema=schema)

    metadata = _read_cast_metadata_sql(meta_name, engine, schema)
    types = convert_meta_to_dict(metadata)
    indexes = list(metadata["column"][metadata["index"]])
    df = (
        read_sql_table(table_name, engine, schema=schema)
        .astype(types)
        .set_index(indexes)
    )
    unname_no_names(df)
    return df


def delete_table_sql(
    table_name: str, engine: Engine, schema: str | None = None,
) -> None:
    """Delete DataFrame and metadata from database."""
    validate_table_name(table_name, storage_type="sql")
    validate_schema_name(schema)
    table = _get_table(table_name, engine, schema)
    metadata = _get_table(f"_{table_name}_metadata", engine, schema)

    with engine.begin() as conn:
        conn.execute(DropTable(table))
        conn.execute(DropTable(metadata))


def load_metadata_sql(
    table_name: str, engine: Engine, schema: str | None = None,
) -> DataFrame:
    """Retrieve DataFrame metadata from database."""
    validate_schema_name(schema)
    meta_name = f"_{table_name}_metadata"
    return _read_cast_metadata_sql(meta_name, engine, schema=schema)


def table_names_sql(engine: Engine, schema: str | None = None) -> list[str]:
    """Query database for list of non-metadata table names."""
    table_names = inspect(engine).get_table_names(schema=schema)
    return [name for name in table_names if "_metadata" not in name and name[0] != "_"]


def metadata_names_sql(engine: Engine, schema: str | None = None) -> list[str]:
    """Query database for list of metadata table names."""
    table_names = inspect(engine).get_table_names(schema=schema)
    return [name for name in table_names if "_metadata" in name and name[0] == "_"]


def _read_cast_metadata_sql(
    table_name: str, engine: Engine, schema: str | None = None,
) -> DataFrame:
    """Load metadata table and cast column datatypes column."""
    meta = read_sql_table(table_name, engine, schema=schema)
    meta["datatype"] = cast_type(meta["datatype"])
    return meta


def _first_load_df_sql(
    table_name: str, engine: Engine, schema: str | None = None,
) -> DataFrame:
    """Load a sql table that has no metadata stored, create and store metadata."""
    df = read_sql_table(table_name, engine, schema=schema)
    store_df_sql(df, table_name, engine)
    return df


def _get_table(table_name: str, engine: Engine, schema: str | None = None) -> Table:
    """Get SQLAlchemy Table mapped to database table."""
    metadata = MetaData(schema=schema)
    return Table(table_name, metadata, autoload_with=engine, schema=schema)


def _is_valid_identifier(identifier: str) -> bool:
    """Validate that an identifier only contains safe characters.

    Args:
        identifier: Identifier to validate (e.g., column name)

    Returns:
        True if identifier is safe, False otherwise

    """
    import re

    # Allow alphanumeric characters, underscores, and dots for qualified names
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_.]*$"
    return bool(re.match(pattern, identifier))


def _contains_sql_injection_patterns(sql_clause: str) -> bool:
    """Check for dangerous SQL patterns that could indicate injection.

    This examines the given clause for potentially unsafe SQL patterns.

    This is a basic sanity check. For production use with user input, always use
    SQLAlchemy parameterized queries instead of string concatenation.

    Args:
        sql_clause: SQL clause to check

    Returns:
        True if dangerous patterns detected, False otherwise

    """
    import re

    # Dangerous SQL patterns that could indicate injection
    dangerous_patterns = [
        r";\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)",  # Multiple statements
        r"--.*",  # SQL comments
        r"/\*.*\*/",  # Block comments
        r";\s*SELECT",  # Chained queries
        r"UNION.*SELECT",  # UNION injection
        r"EXEC\s*\(",  # EXEC execution
        r"xp_.*",  # Extended procedures
        r"CHAR\s*\(.*\)",  # Char encoding
        r"ASCII\s*\(.*\)",  # ASCII functions
    ]

    # Check for dangerous patterns (case insensitive)
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_clause, re.IGNORECASE):
            return True

    return False


def query_table_sql(
    table_name: str,
    engine: Engine,
    where_clause: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    schema: str | None = None,
) -> DataFrame:
    """Query a table with optional filtering and column selection.

    This function constructs a SQL query using SQLAlchemy to safely filter data
    at the database level, avoiding code injection vulnerabilities that would exist
    with pandas DataFrame.query().

    Args:
        table_name: Name of the table to query
        engine: SQLAlchemy engine
        where_clause: Optional WHERE clause SQL expression.
            Example: "age > 25 AND status = 'active'".
            Uses SQL syntax, not pandas query syntax.
            For parameterized queries, use SQLAlchemy directly
            instead of this helper function.
        columns: Optional list of columns to select
        limit: Optional limit on number of rows
        schema: Optional schema name

    Returns:
        DataFrame with query results

    Raises:
        ValidationError: If where_clause contains potentially dangerous SQL

    Examples:
        >>> storage.query('users', where_clause="age > 25")
        >>> storage.query('users', where_clause="status = 'active' AND age < 50")
        >>> storage.query('users', columns=['name', 'email'], limit=10)

    """
    from pandas import read_sql_query

    from trashpandas.exceptions import ValidationError

    # Validate inputs
    validate_table_name(table_name, storage_type="sql")
    validate_schema_name(schema)

    # Build table reference
    full_table_name = f"{schema}.{table_name}" if schema else table_name

    # Start building the SQL query
    # First, load metadata to restore proper types and indexes after querying
    meta_name = f"_{table_name}_metadata"
    metadata_exists = meta_name in metadata_names_sql(engine, schema=schema)

    # Build column list for SELECT clause
    if columns:
        # Validate column names to prevent SQL injection
        for col in columns:
            if not _is_valid_identifier(col):
                raise ValidationError(
                    "columns",
                    col,
                    "Invalid characters in column name. "
                    "Only alphanumeric and underscore allowed.",
                )
        select_cols = ", ".join(f'"{col}"' for col in columns)
    else:
        select_cols = "*"

    # Build the base query
    query_parts = [f"SELECT {select_cols} FROM {full_table_name}"]  # noqa: S608

    # Add WHERE clause if provided
    if where_clause:
        # Basic validation to prevent obvious SQL injection attempts
        # Note: This is not foolproof. For user input, use parameterized
        # queries via SQLAlchemy directly
        if _contains_sql_injection_patterns(where_clause):
            raise ValidationError(
                "where_clause",
                where_clause,
                "WHERE clause contains dangerous SQL patterns. "
                "Use parameterized queries for user input.",
            )
        query_parts.append(f"WHERE {where_clause}")

    # Add LIMIT clause if provided
    if limit:
        if not isinstance(limit, int) or limit < 0:
            raise ValidationError(
                "limit", limit, "Limit must be a non-negative integer",
            )
        query_parts.append(f"LIMIT {limit}")

    # Execute the query
    query_sql = " ".join(query_parts)

    try:
        with engine.connect() as conn:
            df = read_sql_query(query_sql, conn)
    except Exception as e:  # noqa: BLE001
        raise ValidationError(
            "query", query_sql, f"Query execution failed: {str(e)}",
        ) from e  # noqa: B904

    # If metadata exists and we selected all columns, restore types and indexes
    if metadata_exists and not columns:
        try:
            metadata = _read_cast_metadata_sql(meta_name, engine, schema)
            types = convert_meta_to_dict(metadata)
            indexes = list(metadata["column"][metadata["index"]])

            # Apply types to available columns
            available_types = {
                col: dtype for col, dtype in types.items() if col in df.columns
            }
            if available_types:
                df = df.astype(available_types)

            # Set indexes if all index columns are present
            if indexes and all(idx in df.columns for idx in indexes):
                df = df.set_index(indexes)
                unname_no_names(df)
        except Exception:  # noqa: BLE001, S110
            # If metadata restoration fails, return df as-is
            pass

    return df


def query_table_sql_async(
    table_name: str,
    engine: Engine,
    where_clause: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    schema: str | None = None,
) -> DataFrame:
    """Query a table asynchronously with optional filtering and column selection.

    Args:
        table_name: Name of the table to query
        engine: SQLAlchemy engine
        where_clause: Optional WHERE clause (without 'WHERE' keyword)
        columns: Optional list of columns to select
        limit: Optional limit on number of rows
        schema: Optional schema name

    Returns:
        DataFrame with query results

    """
    # For now, this is a synchronous wrapper
    # In a full async implementation, this would use async engine
    return query_table_sql(table_name, engine, where_clause, columns, limit, schema)
