"""Custom exception hierarchy for TrashPandas."""

from __future__ import annotations

from typing import Any


class TrashPandasError(Exception):
    """Base exception for all TrashPandas errors."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Additional context or data about the error

        """
        super().__init__(message)
        self.message = message
        self.details = details


class StorageError(TrashPandasError):
    """Base exception for storage-related errors."""

    pass


class TableNotFoundError(StorageError):
    """Raised when a requested table does not exist."""

    def __init__(self, table_name: str, storage_type: str) -> None:
        """Initialize the exception.

        Args:
            table_name: Name of the table that was not found
            storage_type: Type of storage backend (sql, csv, hdf5, pickle)

        """
        message = f"Table '{table_name}' not found in {storage_type} storage"
        super().__init__(message)
        self.table_name = table_name
        self.storage_type = storage_type


class TableAlreadyExistsError(StorageError):
    """Raised when trying to create a table that already exists."""

    def __init__(self, table_name: str, storage_type: str) -> None:
        """Initialize the exception.

        Args:
            table_name: Name of the table that already exists
            storage_type: Type of storage backend (sql, csv, hdf5, pickle)

        """
        message = f"Table '{table_name}' already exists in {storage_type} storage"
        super().__init__(message)
        self.table_name = table_name
        self.storage_type = storage_type


class MetadataError(TrashPandasError):
    """Base exception for metadata-related errors."""

    pass


class MetadataNotFoundError(MetadataError):
    """Raised when metadata for a table is not found."""

    def __init__(self, table_name: str) -> None:
        """Initialize the exception.

        Args:
            table_name: Name of the table with missing metadata

        """
        message = f"Metadata for table '{table_name}' not found"
        super().__init__(message)
        self.table_name = table_name


class MetadataCorruptedError(MetadataError):
    """Raised when metadata is corrupted or invalid."""

    def __init__(self, table_name: str, details: str | None = None) -> None:
        """Initialize the exception.

        Args:
            table_name: Name of the table with corrupted metadata
            details: Additional details about the corruption

        """
        message = f"Metadata for table '{table_name}' is corrupted"
        if details:
            message += f": {details}"
        super().__init__(message)
        self.table_name = table_name


class StorageConnectionError(StorageError):
    """Raised when there are issues with storage connections."""

    def __init__(self, storage_type: str, details: str | None = None) -> None:
        """Initialize the exception.

        Args:
            storage_type: Type of storage backend
            details: Additional details about the connection issue

        """
        message = f"Connection error with {storage_type} storage"
        if details:
            message += f": {details}"
        super().__init__(message)
        self.storage_type = storage_type


# Note: ConnectionError was removed to avoid shadowing Python builtin.
# Use StorageConnectionError instead.


class ValidationError(TrashPandasError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        """Initialize the exception.

        Args:
            field: Name of the field that failed validation
            value: The invalid value
            reason: Reason for validation failure

        """
        message = f"Validation error for '{field}': {reason} (got: {value!r})"
        super().__init__(message)
        self.field = field
        self.value = value
        self.reason = reason


class ConversionError(TrashPandasError):
    """Raised when data conversion between storage formats fails."""

    def __init__(
        self, source_format: str, target_format: str, details: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            source_format: Source storage format
            target_format: Target storage format
            details: Additional details about the conversion failure

        """
        message = f"Failed to convert from {source_format} to {target_format}"
        if details:
            message += f": {details}"
        super().__init__(message)
        self.source_format = source_format
        self.target_format = target_format


class CompressionError(TrashPandasError):
    """Raised when compression/decompression operations fail."""

    def __init__(self, operation: str, details: str | None = None) -> None:
        """Initialize the exception.

        Args:
            operation: The compression operation that failed
            details: Additional details about the failure

        """
        message = f"Compression error during {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)
        self.operation = operation
