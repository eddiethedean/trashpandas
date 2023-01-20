import abc
from typing import Optional, List

from pandas import DataFrame


class IStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __setitem__(self, key: str, other: DataFrame) -> None:
        """Store DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key: str) -> DataFrame:
        """Retrieve stored DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def __delitem__(self, key) -> None:
        """Delete stored DataFrame."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def store(self, df: DataFrame, table_name: str, schema: Optional[str] = None) -> None:
        """Store DataFrame."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, table_name: str, schema: Optional[str] = None) -> DataFrame:
        """Retrieve stored DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, table_name: str, schema: Optional[str] = None) -> None:
        """Delete stored DataFrame."""
        raise NotImplementedError

    @abc.abstractmethod
    def table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of stored table names."""
        raise NotImplementedError


    #__setitem__

    #__getitem__

    #__delitem__    

    #store

    #load

    #delete

    #load_metadata -Optional

    #table_names

    #metadata_names -Optional


