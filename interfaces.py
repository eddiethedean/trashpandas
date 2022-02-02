import abc

from pandas import DataFrame


class IStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, key: str) -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, key: str, other: DataFrame):
        raise NotImplementedError

    @abc.abstractmethod
    def store(self, df: DataFrame, name: str) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, name: str) -> DataFrame:
        raise NotImplementedError

    