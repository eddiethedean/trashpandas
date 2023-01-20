__version__ = '0.0.9'

import importlib.util

from trashpandas.sql import SqlStorage
from trashpandas.csv import CsvStorage
from trashpandas.pickle import PickleStorage

from trashpandas.sql import store_df_sql, load_df_sql, delete_table_sql, table_names_sql
from trashpandas.csv import store_df_csv, load_df_csv, delete_table_csv, table_names_csv
from trashpandas.pickle import store_df_pickle, load_df_pickle, delete_table_pickle, table_names_pickle

if importlib.util.find_spec('h5py'):
    from trashpandas.hdf5 import HdfStorage
    from trashpandas.hdf5 import store_df_hdf5, load_df_hdf5, delete_table_hdf5, table_names_hdf5