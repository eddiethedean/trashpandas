"""
Convert tables between storage types. 

CsvStorage <-> SqlStorage
CsvStorage <-> HdfStorage
CsvStorage <-> PickleStorage
SqlStorage <-> HdfStorage
SqlStorage <-> PickleStorage
HdfStorage <-> PickleStorage

"""

from sqlalchemy.engine import Engine

from trashpandas.interfaces import IStorage
from trashpandas.csv import CsvStorage
from trashpandas.pickle import PickleStorage
from trashpandas.sql import SqlStorage
from trashpandas.hdf5 import HdfStorage


def convert_table_storage(table_name: str, starting_storage: IStorage, ending_storage: IStorage) -> None:
    df = starting_storage.load(table_name)
    ending_storage.store(df, table_name)


def convert_all_tables_storage(starting_storage: IStorage, ending_storage: IStorage) -> None:
    for table_name in starting_storage.table_names():
        convert_table_storage(table_name, starting_storage, ending_storage)


def csv_to_sql(table_name: str, csv_folder_path: str, engine: Engine) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    sql_storage = SqlStorage(engine)
    convert_table_storage(table_name, csv_storage, sql_storage)


def csv_to_sql_all(csv_folder_path: str, engine: Engine) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    sql_storage = SqlStorage(engine)
    convert_all_tables_storage(csv_storage, sql_storage)


def csv_to_hdf(table_name: str, csv_folder_path: str, hdf_file_path: str) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_table_storage(table_name, csv_storage, hdf_storage)


def csv_to_hdf_all(csv_folder_path: str, hdf_file_path: str) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_all_tables_storage(csv_storage, hdf_storage)


def csv_to_pickle(table_name: str, csv_folder_path: str, pickle_folder_path: str) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_table_storage(table_name, csv_storage, pickle_storage)


def csv_to_pickle_all(csv_folder_path: str, pickle_folder_path: str) -> None:
    csv_storage = CsvStorage(csv_folder_path)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_all_tables_storage(csv_storage, pickle_storage)


def sql_to_csv(table_name: str, engine: Engine, csv_folder_path: str) -> None:
    sql_storage = SqlStorage(engine)
    csv_storage = CsvStorage(csv_folder_path)
    convert_table_storage(table_name, sql_storage, csv_storage)


def sql_to_csv_all(engine: Engine, csv_folder_path: str) -> None:
    sql_storage = SqlStorage(engine)
    csv_storage = CsvStorage(csv_folder_path)
    convert_all_tables_storage(sql_storage, csv_storage)


def sql_to_hdf(table_name: str, engine: Engine, hdf_file_path: str) -> None:
    sql_storage = SqlStorage(engine)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_table_storage(table_name, sql_storage, hdf_storage)


def sql_to_hdf_all(engine: Engine, hdf_file_path: str) -> None:
    sql_storage = SqlStorage(engine)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_all_tables_storage(sql_storage, hdf_storage)


def sql_to_pickle(table_name: str, engine: Engine, pickle_folder_path: str) -> None:
    sql_storage = SqlStorage(engine)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_table_storage(table_name, sql_storage, pickle_storage)


def sql_to_pickle_all(engine: Engine, pickle_folder_path: str) -> None:
    sql_storage = SqlStorage(engine)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_all_tables_storage(sql_storage, pickle_storage)


def hdf_to_csv(table_name: str, hdf_file_path: str, csv_folder_path: str) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    csv_storage = CsvStorage(csv_folder_path)
    convert_table_storage(table_name, hdf_storage, csv_storage)


def hdf_to_csv_all(hdf_file_path: str, csv_folder_path: str) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    csv_storage = CsvStorage(csv_folder_path)
    convert_all_tables_storage(hdf_storage, csv_storage)


def hdf_to_sql(table_name: str, hdf_file_path: str, engine: Engine) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    sql_storage = SqlStorage(engine)
    convert_table_storage(table_name, hdf_storage, sql_storage)


def hdf_to_sql_all(hdf_file_path: str, engine: Engine) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    sql_storage = SqlStorage(engine)
    convert_all_tables_storage(hdf_storage, sql_storage)


def hdf_to_pickle(table_name: str, hdf_file_path: str, pickle_folder_path: str) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_table_storage(table_name, hdf_storage, pickle_storage)


def hdf_to_pickle_all(hdf_file_path: str, pickle_folder_path: str) -> None:
    hdf_storage = HdfStorage(hdf_file_path)
    pickle_storage = PickleStorage(pickle_folder_path)
    convert_all_tables_storage(hdf_storage, pickle_storage)


def pickle_to_csv(table_name: str, pickle_folder_path: str, csv_folder_path: str) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    csv_storage = CsvStorage(csv_folder_path)
    convert_table_storage(table_name, pickle_storage, csv_storage)


def pickle_to_csv_all(pickle_folder_path: str, csv_folder_path: str) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    csv_storage = CsvStorage(csv_folder_path)
    convert_all_tables_storage(pickle_storage, csv_storage)


def pickle_to_sql(table_name: str, pickle_folder_path: str, engine: Engine) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    sql_storage = SqlStorage(engine)
    convert_table_storage(table_name, pickle_storage, sql_storage)


def pickle_to_sql_all(pickle_folder_path: str, engine: Engine) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    sql_storage = SqlStorage(engine)
    convert_all_tables_storage(pickle_storage, sql_storage)


def pickle_to_hdf(table_name: str, pickle_folder_path: str, hdf_file_path: str) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_table_storage(table_name, pickle_storage, hdf_storage)


def pickle_to_hdf_all(pickle_folder_path: str, hdf_file_path: str) -> None:
    pickle_storage = PickleStorage(pickle_folder_path)
    hdf_storage = HdfStorage(hdf_file_path)
    convert_all_tables_storage(pickle_storage, hdf_storage)