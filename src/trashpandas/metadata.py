"""Metadata handling for TrashPandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import pandas_dtype

from trashpandas.exceptions import MetadataCorruptedError


@dataclass
class TableMetadata:
    """Metadata for a stored DataFrame table.
    
    This class encapsulates all the metadata needed to properly restore
    a DataFrame from storage, including column types, index information,
    and other structural details.
    """
    
    table_name: str
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, Any] = field(default_factory=dict)
    index_columns: List[str] = field(default_factory=list)
    index_types: Dict[str, Any] = field(default_factory=dict)
    storage_format: str = "unknown"
    created_at: Optional[str] = None
    version: str = "1.0.0"
    
    @classmethod
    def from_dataframe(cls, df: DataFrame, table_name: str, storage_format: str = "unknown") -> TableMetadata:
        """Create metadata from a DataFrame.
        
        Args:
            df: The DataFrame to extract metadata from
            table_name: Name of the table
            storage_format: Format used for storage (sql, csv, hdf5, pickle)
            
        Returns:
            TableMetadata object with extracted information
        """
        # Extract column information
        columns = list(df.columns)
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Extract index information
        index_columns = []
        index_types = {}
        
        if len(df.index.names) == 1:
            if df.index.name is not None:
                index_columns = [df.index.name]
                index_types = {df.index.name: str(df.index.dtype)}
        else:
            for i, name in enumerate(df.index.names):
                if name is not None:
                    index_columns.append(name)
                    index_types[name] = str(df.index.dtypes[i])
        
        return cls(
            table_name=table_name,
            columns=columns,
            column_types=column_types,
            index_columns=index_columns,
            index_types=index_types,
            storage_format=storage_format
        )
    
    def to_dataframe(self) -> DataFrame:
        """Convert metadata to a DataFrame for storage.
        
        Returns:
            DataFrame with metadata information
        """
        data = []
        
        # Add index information
        for col in self.index_columns:
            data.append({
                'column': col,
                'index': True,
                'datatype': self.index_types.get(col, 'object')
            })
        
        # Add column information
        for col in self.columns:
            data.append({
                'column': col,
                'index': False,
                'datatype': self.column_types.get(col, 'object')
            })
        
        return DataFrame(data)
    
    @classmethod
    def from_dataframe_metadata(cls, metadata_df: DataFrame, table_name: str, storage_format: str = "unknown") -> TableMetadata:
        """Create metadata from a stored metadata DataFrame.
        
        Args:
            metadata_df: DataFrame containing metadata information
            table_name: Name of the table
            storage_format: Format used for storage
            
        Returns:
            TableMetadata object
            
        Raises:
            MetadataCorruptedError: If metadata is invalid or corrupted
        """
        try:
            # Validate required columns
            required_columns = {'column', 'index', 'datatype'}
            if not required_columns.issubset(metadata_df.columns):
                raise MetadataCorruptedError(
                    table_name,
                    f"Missing required columns. Expected: {required_columns}, got: {set(metadata_df.columns)}"
                )
            
            # Extract column and index information
            columns = []
            column_types = {}
            index_columns = []
            index_types = {}
            
            for _, row in metadata_df.iterrows():
                col_name = row['column']
                is_index = bool(row['index'])
                dtype = row['datatype']
                
                if is_index:
                    index_columns.append(col_name)
                    index_types[col_name] = dtype
                else:
                    columns.append(col_name)
                    column_types[col_name] = dtype
            
            return cls(
                table_name=table_name,
                columns=columns,
                column_types=column_types,
                index_columns=index_columns,
                index_types=index_types,
                storage_format=storage_format
            )
            
        except Exception as e:
            raise MetadataCorruptedError(table_name, f"Failed to parse metadata: {e}") from e
    
    def get_column_types(self) -> Dict[str, Any]:
        """Get column types as pandas dtypes.
        
        Returns:
            Dictionary mapping column names to pandas dtypes
        """
        return {col: pandas_dtype(dtype) for col, dtype in self.column_types.items()}
    
    def get_index_types(self) -> Dict[str, Any]:
        """Get index types as pandas dtypes.
        
        Returns:
            Dictionary mapping index names to pandas dtypes
        """
        return {col: pandas_dtype(dtype) for col, dtype in self.index_types.items()}
    
    def validate(self) -> None:
        """Validate the metadata for consistency.
        
        Raises:
            MetadataCorruptedError: If metadata is invalid
        """
        # Check for duplicate column names
        all_names = self.columns + self.index_columns
        if len(all_names) != len(set(all_names)):
            duplicates = [name for name in set(all_names) if all_names.count(name) > 1]
            raise MetadataCorruptedError(
                self.table_name,
                f"Duplicate column/index names found: {duplicates}"
            )
        
        # Check that all columns have types
        for col in self.columns:
            if col not in self.column_types:
                raise MetadataCorruptedError(
                    self.table_name,
                    f"Missing type information for column: {col}"
                )
        
        # Check that all index columns have types
        for col in self.index_columns:
            if col not in self.index_types:
                raise MetadataCorruptedError(
                    self.table_name,
                    f"Missing type information for index column: {col}"
                )
    
    def __str__(self) -> str:
        """String representation of metadata."""
        return f"TableMetadata(table='{self.table_name}', columns={len(self.columns)}, indices={len(self.index_columns)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of metadata."""
        return (f"TableMetadata(table_name='{self.table_name}', "
                f"columns={self.columns}, "
                f"index_columns={self.index_columns}, "
                f"storage_format='{self.storage_format}')")
