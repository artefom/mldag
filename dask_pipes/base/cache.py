import yaml
import pandas as pd
import dask.dataframe as dd
import os
from uuid import uuid4
from datetime import datetime

__all__ = ['DataFrameRepresenter', 'CacheBase', 'dump']


class DataFrameRepresenter(yaml.SafeDumper):

    def ignore_aliases(self, data):
        return True

    def __init__(self, directory, *args, **kwargs):
        self.directory = directory
        self.df_dump_cache = dict()
        super().__init__(*args, **kwargs)


def represent_dd_dataframe(dumper: DataFrameRepresenter, data: dd.DataFrame):
    if data in dumper.df_dump_cache:
        dump_data = dumper.df_dump_cache[data]
    else:
        fname = '{}.parquet'.format(uuid4())
        fname_full = os.path.join(dumper.directory, fname)
        data.to_parquet(fname_full)
        dump_data = {'file': fname, 'ts': os.path.getatime(fname_full)}
        dumper.df_dump_cache[data] = dump_data
    return dumper.represent_mapping('tag:yaml.org,2002:dask.DataFrame', dump_data)


def represent_pd_dataframe(dumper: DataFrameRepresenter, data: pd.DataFrame):
    if data in dumper.df_dump_cache:
        dump_data = dumper.df_dump_cache[data]
    else:
        fname = '{}.hdf'.format(uuid4())
        fname_full = os.path.join(dumper.directory, fname)
        data.to_hdf(fname_full, 'df')
        dump_data = {'file': fname, 'ts': os.path.getatime(fname_full)}
        dumper.df_dump_cache[data] = dump_data
    return dumper.represent_mapping('tag:yaml.org,2002:pandas.DataFrame', dump_data)


DataFrameRepresenter.add_representer(dd.DataFrame, represent_dd_dataframe)
DataFrameRepresenter.add_representer(pd.DataFrame, represent_pd_dataframe)


def dump(data, data_dir, stream=None):
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    dumper = DataFrameRepresenter(data_dir, stream)
    try:
        dumper.open()
        represented = dumper.represent_data(data)
        dumper.close()
    finally:
        dumper.dispose()

    return represented


class CacheBase:
    pass
