import yaml
import pandas as pd
import dask.dataframe as dd
import os
from uuid import uuid4
from datetime import datetime
from urllib.parse import urlparse, unquote
import pathlib
import socket
from yaml.constructor import ConstructorError
import base64

__all__ = ['path_from_uri', 'path_to_uri',
           'dataframe_dumper_factory',
           'dataframe_loader_factory']


def label(a):
    val = int(abs(a % 1099511627775))
    return base64.b64encode(val.to_bytes(5, byteorder='big')).decode('ascii')


def path_to_uri(path):
    return pathlib.Path(path).as_uri()


def path_from_uri(uri):
    p = urlparse(uri)
    return os.path.abspath(os.path.join(p.netloc, unquote(p.path)))


class DataFrameLoader(yaml.SafeLoader):

    def __init__(self,
                 *args,
                 df_dump_cache=None,
                 df_load_cache=None,
                 recover_categories=True,
                 **kwargs):
        self.recover_categories = recover_categories
        if df_load_cache is None:
            self.df_load_cache = dict()
        else:
            self.df_load_cache = df_load_cache

        if df_dump_cache is None:
            self.df_dump_cache = dict()
        else:
            self.df_dump_cache = df_dump_cache
        super().__init__(*args, **kwargs)

    def construct_dask_dataframe(self, node):
        state = self.construct_mapping(node)
        if tuple(state.items()) in self.df_load_cache:
            rv = self.df_load_cache[tuple(state.items())]
            return rv
        path = path_from_uri(state['uri'])
        if not os.path.exists(path):
            raise ConstructorError("while constructing a dask.DataFrame", node.start_mark,
                                   "file %s not found" % path,
                                   node.start_mark)
        mtime_orig = state['mtime']
        mtime = os.path.getmtime(path)
        if mtime_orig != mtime:
            raise ConstructorError("while constructing a dask.DataFrame", node.start_mark,
                                   "file modified. original ts: %s, but found: %s" % (mtime_orig, mtime),
                                   node.start_mark)
        rv = dd.read_parquet(path)
        if self.recover_categories:
            for col in rv.columns:
                try:
                    rv[col] = rv[col].cat.set_categories(rv[col].head(1).dtype.categories)
                except AttributeError:
                    pass

        self.df_load_cache[tuple(state.items())] = rv
        self.df_dump_cache[rv] = state
        return rv

    def construct_pandas_dataframe(self, node):
        state = self.construct_mapping(node)
        if tuple(state.items()) in self.df_load_cache:
            rv = self.df_load_cache[tuple(state.items())]
            return rv
        path = path_from_uri(state['uri'])
        if not os.path.exists(path):
            raise ConstructorError("while constructing a pandas.DataFrame", node.start_mark,
                                   "file %s not found" % path,
                                   node.start_mark)
        mtime_orig = state['mtime']
        mtime = os.path.getmtime(path)
        if mtime_orig != mtime:
            raise ConstructorError("while constructing a pandas.DataFrame", node.start_mark,
                                   "file modified. original ts: %s, but found: %s" % (mtime_orig, mtime),
                                   node.start_mark)
        rv = dd.read_hdf(path, 'df')
        self.df_dump_cache[rv] = state
        self.df_load_cache[tuple(state.items())] = rv
        return rv.copy()


DataFrameLoader.add_constructor('tag:yaml.org,2002:dask.DataFrame',
                                DataFrameLoader.construct_dask_dataframe)
DataFrameLoader.add_constructor('tag:yaml.org,2002:pandas.DataFrame',
                                DataFrameLoader.construct_pandas_dataframe)


class DataFrameDumper(yaml.SafeDumper):

    def __init__(self,
                 *args,
                 directory=None,
                 node_name=None,
                 run_name=None,
                 df_dump_cache=None,
                 df_load_cache=None,
                 **kwargs):
        if directory is None:
            raise ValueError("Directory of %s cannot be null. "
                             "Use dataframe_dumper_factory to provide additional arguments to dumper"
                             % self.__class__.__name__)
        self.directory = directory

        if df_load_cache is None:
            self.df_load_cache = dict()
        else:
            self.df_load_cache = df_load_cache

        if df_dump_cache is None:
            self.df_dump_cache = dict()
        else:
            self.df_dump_cache = df_dump_cache

        self.node_name = node_name
        self.run_name = run_name
        super().__init__(*args, **kwargs)

    def ignore_aliases(self, data):
        return True

    def get_fname(self):
        ts = datetime.now().strftime('%Y%m%d%H%M')
        rv = str(uuid4())
        if self.node_name is not None:
            rv = rv[:5]
        if self.node_name is not None:
            rv = '{}-{}'.format(self.node_name, rv)
        if self.run_name is not None:
            rv = '{}-{}'.format(self.run_name, rv)
        return '{}-{}'.format(rv, ts)

    def represent_dd_dataframe(self, data: dd.DataFrame):
        if data in self.df_dump_cache:
            dump_data = self.df_dump_cache[data]
        else:
            fname = '{}.parquet'.format(self.get_fname())
            fname_full = os.path.join(self.directory, fname)
            data.to_parquet(fname_full)
            dump_data = {
                'uri': path_to_uri(fname_full),
                'host': socket.gethostname(),
                'mtime': os.path.getmtime(fname_full)}
            self.df_dump_cache[data] = dump_data
        return self.represent_mapping('tag:yaml.org,2002:dask.DataFrame', dump_data)

    def represent_pd_dataframe(self, data: pd.DataFrame):
        if data in self.df_dump_cache:
            dump_data = self.df_dump_cache[data]
        else:
            fname = '{}.hdf'.format(uuid4())
            fname_full = os.path.join(self.directory, fname)
            data.to_hdf(fname_full, 'df')
            dump_data = {
                'uri': path_to_uri(fname_full),
                'host': socket.gethostname(),
                'ts': os.path.getatime(fname_full)}
            self.df_dump_cache[data] = dump_data
            self.df_load_cache[tuple(dump_data.items())] = data
        return self.represent_mapping('tag:yaml.org,2002:pandas.DataFrame', dump_data)


DataFrameDumper.add_representer(dd.DataFrame, DataFrameDumper.represent_dd_dataframe)
DataFrameDumper.add_representer(pd.DataFrame, DataFrameDumper.represent_pd_dataframe)


def dataframe_loader_factory(df_dump_cache=None, df_load_cache=None, recover_categories=True):
    def get_loader(*args, **kwargs):
        return DataFrameLoader(*args,
                               df_dump_cache=df_dump_cache,
                               df_load_cache=df_load_cache,
                               recover_categories=recover_categories,
                               **kwargs)

    return get_loader


def dataframe_dumper_factory(directory, node_name=None, run_name=None, df_dump_cache=None, df_load_cache=None):
    directory = os.path.abspath(directory)

    def get_dumper(*args, **kwargs):
        if not os.path.exists(directory):
            os.mkdir(directory)
        return DataFrameDumper(*args,
                               directory=directory,
                               node_name=node_name,
                               run_name=run_name,
                               df_dump_cache=df_dump_cache,
                               df_load_cache=df_load_cache,
                               **kwargs)

    return get_dumper
