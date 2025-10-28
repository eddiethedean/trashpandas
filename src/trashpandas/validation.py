"""Validation utilities for TrashPandas.

This module provides validation functions to ensure data safety and compatibility
across different storage backends.
"""

from __future__ import annotations

import re
from typing import Literal

from trashpandas.exceptions import ValidationError

# SQL reserved keywords (common subset across databases)
SQL_RESERVED_KEYWORDS = {
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TABLE",
    "INDEX",
    "VIEW",
    "FROM",
    "WHERE",
    "JOIN",
    "UNION",
    "INTO",
    "ORDER",
    "GROUP",
    "HAVING",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "IS",
    "AS",
    "BY",
    "ON",
    "IN",
    "BETWEEN",
    "LIKE",
    "EXISTS",
    "ALL",
    "ANY",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "WITH",
    "DATABASE",
    "SCHEMA",
}

# Filesystem reserved names (Windows)
FS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
    "CLOCK$",  # Additional Windows reserved name
}


def validate_table_name(
    table_name: str,
    storage_type: str = "storage",
    name_format: Literal["strict", "qualified"] = "strict",
    max_length: int = 255,
) -> None:
    """Validate table name for safety and compatibility.

    This function validates table names to prevent:
    - SQL injection attacks
    - Path traversal vulnerabilities
    - Filesystem compatibility issues
    - Reserved keyword conflicts

    Args:
        table_name: Name to validate
        storage_type: Type of storage ('sql', 'csv', 'pickle', 'hdf5', 'storage')
        name_format: Naming format to use.
            'strict': Standard identifiers (alphanumeric and underscore only).
            'qualified': Allows dots for SQL qualified names like schema.table.
        max_length: Maximum allowed length for table name

    Raises:
        ValidationError: If table name is invalid

    Examples:
        >>> validate_table_name('users')  # Valid
        >>> validate_table_name('user_data_2024')  # Valid
        >>> validate_table_name('DROP')  # Raises ValidationError (reserved keyword)
        >>> validate_table_name('../etc/passwd')  # Raises ValidationError
        >>> # (path traversal)
        >>> validate_table_name('table; DROP TABLE users')
        >>> # Raises ValidationError (SQL injection)

    """
    # Check for None
    if table_name is None:
        raise ValidationError("table_name", table_name, "Table name cannot be None")

    # Check for empty/whitespace only
    if not table_name or not table_name.strip():
        raise ValidationError(
            "table_name", table_name, "Table name cannot be empty or whitespace-only",
        )

    # Check length
    if len(table_name) > max_length:
        raise ValidationError(
            "table_name",
            table_name,
            f"Table name too long (max {max_length} characters, got {len(table_name)})",
        )

    # Check for path traversal sequences
    if ".." in table_name:
        raise ValidationError(
            "table_name",
            table_name,
            "Table name contains path traversal sequence (..)",
        )

    # Check for absolute path indicators
    if table_name.startswith("/") or table_name.startswith("\\"):
        raise ValidationError(
            "table_name",
            table_name,
            "Table name cannot start with path separator",
        )

    # Check for Windows absolute path (C:, D:, etc.)
    if len(table_name) > 1 and table_name[1] == ":":
        raise ValidationError(
            "table_name",
            table_name,
            "Table name appears to be an absolute path",
        )

    # Check character validity
    if name_format == "qualified":
        # For SQL qualified names like "schema.table"
        pattern = r"^[a-zA-Z][a-zA-Z0-9_.]*$"
        char_description = "alphanumeric characters, underscores, and dots"
    else:
        # Standard identifier pattern
        pattern = r"^[a-zA-Z][a-zA-Z0-9_]*$"
        char_description = "alphanumeric characters and underscores"

    if not re.match(pattern, table_name):
        raise ValidationError(
            "table_name",
            table_name,
            f"Table name must start with a letter and contain only {char_description}",
        )

    # Check for SQL injection patterns
    dangerous_sql_chars = [";", "--", "/*", "*/", "'", '"', "="]
    for char in dangerous_sql_chars:
        if char in table_name:
            raise ValidationError(
                "table_name",
                table_name,
                f"Table name contains potentially dangerous character: {char}",
            )

    # Check for SQL reserved keywords (case insensitive)
    if storage_type in ("sql", "storage"):
        # Check the base name (without schema prefix if present)
        base_name = table_name.split(".")[-1] if "." in table_name else table_name
        if base_name.upper() in SQL_RESERVED_KEYWORDS:
            raise ValidationError(
                "table_name",
                table_name,
                f'Table name "{base_name}" is a SQL reserved keyword',
            )

    # Check for filesystem reserved names
    if storage_type in ("csv", "pickle", "hdf5", "storage"):
        # Check both the full name and base name (without extension)
        name_upper = table_name.upper()
        base_name_upper = (
            table_name.split(".")[0].upper() if "." in table_name else name_upper
        )

        if name_upper in FS_RESERVED_NAMES or base_name_upper in FS_RESERVED_NAMES:
            raise ValidationError(
                "table_name",
                table_name,
                f'Table name "{table_name}" is a filesystem reserved name',
            )

    # Check for dangerous filesystem characters
    if storage_type in ("csv", "pickle", "hdf5", "storage"):
        # Characters that are problematic on Windows or Unix filesystems
        dangerous_chars = '<>:"|?*\0\t\n\r'
        for char in dangerous_chars:
            if char in table_name:
                # Make special chars visible in error message
                char_repr = repr(char) if char in "\0\t\n\r" else char
                raise ValidationError(
                    "table_name",
                    table_name,
                    f"Table name contains invalid filesystem character: {char_repr}",
                )


def validate_schema_name(schema_name: str | None) -> None:
    """Validate schema name for SQL storage.

    Args:
        schema_name: Schema name to validate (can be None)

    Raises:
        ValidationError: If schema name is invalid

    """
    if schema_name is None:
        return

    # Use same validation as table name but allow qualified format
    validate_table_name(
        schema_name, storage_type="sql", name_format="qualified", max_length=64,
    )
