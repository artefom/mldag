import yaml
import pandas as pd
import dask.dataframe as dd
import os
from uuid import uuid4
import io
from datetime import datetime
from urllib.parse import urlparse
import pathlib
import socket


__all__ = ['DataFrameRepresenter', 'CacheBase', 'dump', 'path_from_uri', 'path_to_uri']


def path_to_uri(path):
    return pathlib.Path(path).as_uri()


def path_from_uri(uri):
    p = urlparse(uri)
    return os.path.abspath(os.path.join(p.netloc, p.path))


class DataFrameRepresenter(yaml.SafeDumper):

    def ignore_aliases(self, data):
        return True

    def get_fname(self):
        ts = datetime.now().strftime('%Y%m%d%H%M')
        rv = str(uuid4())
        if self.node_name is not None:
            rv = rv[:5]
        if self.parameter_name is not None:
            rv = '{}-{}'.format(self.parameter_name, rv)
        if self.node_name is not None:
            rv = '{}-{}'.format(self.node_name, rv)
        if self.run_name is not None:
            rv = '{}-{}'.format(self.run_name, rv)
        return '{}-{}'.format(rv, ts)

    def __init__(self, directory, *args, node_name=None, run_name=None, parameter_name=None, **kwargs):
        self.directory = directory
        self.df_dump_cache = dict()
        self.node_name = node_name
        self.run_name = run_name
        self.parameter_name = parameter_name
        super().__init__(*args, **kwargs)


def represent_dd_dataframe(dumper: DataFrameRepresenter, data: dd.DataFrame):
    if data in dumper.df_dump_cache:
        dump_data = dumper.df_dump_cache[data]
    else:
        fname = '{}.parquet'.format(dumper.get_fname())
        fname_full = os.path.join(dumper.directory, fname)
        data.to_parquet(fname_full)
        dump_data = {
            'uri': path_to_uri(fname_full),
            'host': socket.gethostname(),
            'ts': os.path.getatime(fname_full)}
        dumper.df_dump_cache[data] = dump_data
    return dumper.represent_mapping('tag:yaml.org,2002:dask.DataFrame', dump_data)


def represent_pd_dataframe(dumper: DataFrameRepresenter, data: pd.DataFrame):
    if data in dumper.df_dump_cache:
        dump_data = dumper.df_dump_cache[data]
    else:
        fname = '{}.hdf'.format(uuid4())
        fname_full = os.path.join(dumper.directory, fname)
        data.to_hdf(fname_full, 'df')
        dump_data = {
            'uri': path_to_uri(fname_full),
            'host': socket.gethostname(),
            'ts': os.path.getatime(fname_full)}
        dumper.df_dump_cache[data] = dump_data
    return dumper.represent_mapping('tag:yaml.org,2002:pandas.DataFrame', dump_data)


DataFrameRepresenter.add_representer(dd.DataFrame, represent_dd_dataframe)
DataFrameRepresenter.add_representer(pd.DataFrame, represent_pd_dataframe)


def dump(data, data_dir, stream=None, node_name=None, run_name=None, parameter_name=None):
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if stream is None:
        stream = io.StringIO()
        dumper = DataFrameRepresenter(data_dir, stream,
                                      node_name=node_name,
                                      run_name=run_name,
                                      parameter_name=parameter_name)
        try:
            dumper.open()
            dumper.represent(data)
            dumper.close()
            return stream.getvalue()
        finally:
            dumper.dispose()
            stream.close()
    else:
        dumper = DataFrameRepresenter(data_dir, stream)
        try:
            dumper.open()
            dumper.represent(data)
            dumper.close()
        finally:
            dumper.dispose()


class CacheBase:
    pass
