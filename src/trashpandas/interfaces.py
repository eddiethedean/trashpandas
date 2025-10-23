"""Storage interface definitions for TrashPandas."""

from __future__ import annotations

import abc
from typing import Optional, List, Iterator, Union
from pathlib import Path

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
    def store(self, df: DataFrame, table_name: str, schema: Optional[str] = None) -> None:
        """Store DataFrame.
        
        Args:
            df: DataFrame to store
            table_name: Name to store the DataFrame as
            schema: Optional schema name (for SQL backends)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve stored DataFrame.
        
        Args:
            table_name: Name of the table to retrieve
            schema: Optional schema name (for SQL backends)
            
        Returns:
            The stored DataFrame
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, table_name: str, schema: Optional[str] = None) -> None:
        """Delete stored DataFrame.
        
        Args:
            table_name: Name of the table to delete
            schema: Optional schema name (for SQL backends)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def table_names(self, schema: Optional[str] = None) -> List[str]:
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
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
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
    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize file storage.
        
        Args:
            path: Path to storage location
        """
        raise NotImplementedError


class ISqlStorage(IStorage):
    """Interface for SQL-based storage backends."""
    
    @abc.abstractmethod
    def load_metadata(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Load metadata for a table.
        
        Args:
            table_name: Name of the table
            schema: Optional schema name
            
        Returns:
            DataFrame containing metadata
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def metadata_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of metadata table names.
        
        Args:
            schema: Optional schema name
            
        Returns:
            List of metadata table names
        """
        raise NotImplementedError
