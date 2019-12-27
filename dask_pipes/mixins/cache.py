from typing import Any, Dict, Tuple, List
from collections import namedtuple
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
from dask_pipes.base import PipelineMixin, NodeCallable, NodeBase
from ..exceptions import DaskPipesException

__all__ = ['path_from_uri', 'path_to_uri',
           'dataframe_dumper_factory',
           'dataframe_loader_factory',
           'CacheMixin']


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


NodeCache = namedtuple('NodeCache', ['id', 'run_id', 'node_name', 'time', 'node_input', 'node_output'])
NodeCacheLoad = namedtuple('NodeCacheLoad', ['id', 'run_id', 'node_name', 'time', 'cache_id'])


class CacheMixin(PipelineMixin):

    def __init__(self, cache_dir, recover_categories=True):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.recover_categories = recover_categories

        self._cache_id_counter = 0
        self._cache: List[NodeCache] = list()

        self._cache_loads_id_counter = 0
        self._cache_loads: List[NodeCacheLoad] = list()

        # Used for fast access of latest cache
        self._latest_cache_by_input = dict()

        # Used in yaml loader, dumper to prevent multiple dumps/loads of same dataframe
        self._df_dump_cache = dict()
        self._df_load_cache = dict()

    def __str__(self):
        return "<CacheMixin {}>".format('./' + os.path.relpath(self.cache_dir, '.'))

    def __repr__(self):
        return str(self)

    @property
    def cache(self):
        return pd.DataFrame(self._cache).set_index('id')

    @property
    def cache_loads(self):
        return pd.DataFrame(self._cache_loads).set_index('id')

    def add_record(self, node_name, node_input, node_output, time=None):
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node_name,
            df_dump_cache=self._df_dump_cache,
            df_load_cache=self._df_load_cache,
        )
        node_input = yaml.dump(node_input, Dumper=dumper)
        node_output = yaml.dump(node_output, Dumper=dumper)

        rec = NodeCache(
            id=self._cache_id_counter,
            run_id=self.run_id,
            node_name=node_name,
            time=time or datetime.now(),
            node_input=node_input,
            node_output=node_output
        )
        self._cache_id_counter += 1

        # Make cache records
        self._cache.append(rec)
        self._latest_cache_by_input[(node_name, node_input)] = rec

    # def load(self, data):
    #     loader = dataframe_loader_factory(
    #         df_dump_cache=self.dump_cache,
    #         df_load_cache=self.load_cache,
    #     )
    #     return yaml.load(data, Loader=loader)
    #
    # def dump(self, data):
    #     dumper = dataframe_dumper_factory(
    #         self.cache_dir,
    #         node_name='manual',
    #         df_dump_cache=self.dump_cache,
    #         df_load_cache=self.load_cache,
    #     )
    #     yaml.dump(data, Dumper=dumper)

    def get_latest_output(self, node_name, node_input=None, time=None):
        if node_input is None:
            matches = sorted(filter(lambda x: x.node_name == node_name, self._cache), key=lambda x: x.time)
            if len(matches) == 0:
                raise DaskPipesException("Cache for node {} does not exist".format(node_name))
            loader = dataframe_loader_factory(
                df_dump_cache=self._df_dump_cache,
                df_load_cache=self._df_load_cache,
            )
            return yaml.load(matches[-1].node_output, Loader=loader)
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node_name,
            df_dump_cache=self._df_dump_cache,
            df_load_cache=self._df_load_cache,
        )
        orig_input = yaml.dump(node_input, Dumper=dumper)
        if (node_name, orig_input) in self._latest_cache_by_input:
            node_cache: NodeCache = self._latest_cache_by_input[(node_name, orig_input)]
            loader = dataframe_loader_factory(
                df_dump_cache=self._df_dump_cache,
                df_load_cache=self._df_load_cache,
            )
            match = yaml.load(node_cache.node_output, Loader=loader)

            # Append cache load info
            rec = NodeCacheLoad(
                id=self._cache_loads_id_counter,
                run_id=self.run_id,
                node_name=node_name,
                time=time or datetime.now(),
                cache_id=node_cache.id,
            )
            self._cache_loads_id_counter += 1
            self._cache_loads.append(rec)

            return match
        else:
            raise DaskPipesException("Cache for node %s and specific input does not exist" % node_name)

    def _fit(self,
             func: NodeCallable,
             node: NodeBase,
             node_input: Tuple[Tuple[Any], Dict[str, Any]],
             has_downstream=True):
        try:
            node_output = self.get_latest_output(node.name, node_input)
        except DaskPipesException:
            node_output = func(node, node_input, has_downstream=has_downstream)
            if has_downstream:
                self.add_record(node.name, node_input, node_output)
                node_output = self.get_latest_output(node.name, node_input)
        return node_output

    def _transform(self,
                   func: NodeCallable,
                   node: NodeBase,
                   node_input: Tuple[Tuple[Any], Dict[str, Any]],
                   has_downstream=True):
        try:
            node_output = self.get_latest_output(node.name, node_input)
        except DaskPipesException:
            node_output = func(node, node_input, has_downstream=has_downstream)
            self.add_record(node.name, node_input, node_output)
            node_output = self.get_latest_output(node.name, node_input)
        return node_output

    def reset(self):
        pass
