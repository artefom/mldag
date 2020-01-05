from typing import Any, Dict, Tuple, List, Union
from collections import namedtuple
import yaml
import pandas as pd
import dask.dataframe as dd
import os
from uuid import uuid4
from datetime import datetime
from urllib.parse import urlparse, unquote
import pathlib
from yaml.constructor import ConstructorError
from dask_pipes.base import PipelineMixin, NodeCallable, NodeBase
from functools import partial
from ..exceptions import DaskPipesException
import weakref
import inspect
import hashlib
import logging
import base64

logger = logging.getLogger(__name__)

__all__ = ['path_from_uri', 'path_to_uri',
           'dataframe_dumper_factory',
           'dataframe_loader_factory',
           'CacheMixin']


def path_to_uri(path):
    return pathlib.Path(path).as_uri()


def path_from_uri(uri):
    p = urlparse(uri)
    return os.path.abspath(os.path.join(p.netloc, unquote(p.path)))


def hash_class_code(cls, depth=2, cur_depth=0, class_hash=None):
    class_hash = class_hash or hashlib.sha256()

    try:
        mtime = os.path.getmtime(inspect.getfile(cls))
        class_hash.update(str(mtime).encode('utf-8'))
        return class_hash.hexdigest()
    except TypeError:
        pass

    def hash_dict(d):
        for k, v in d.items():
            try:
                class_hash.update(v.__code__.co_code)
                for v in v.__code__.co_consts:
                    class_hash.update(str(v).encode('utf-8'))
                continue
            except AttributeError:
                pass
            if k[:2] != '__':
                class_hash.update(str(v).encode('utf-8'))

    if isinstance(cls, type):
        hash_dict(cls.__dict__)
    else:
        hash_dict(cls.__class__.__dict__)
        # Find classes used by instance of object and hash them too
        if cur_depth < depth - 1:
            sub_classes = set()
            for k, v in cls.__dict__.items():
                if hasattr(v, '__dict__'):
                    sub_classes.add(v.__class__)
            sub_class_hashes = set()
            for c in sub_classes:
                sub_class_hashes.add(hash_class_code(c, depth=depth, cur_depth=cur_depth + 1))
            # Update in sorted order to make current hash invariant to order
            # (elements in set may have different order from run to run,
            # so without this resulting hash will be different from run to run too)
            for c in sorted(sub_class_hashes):
                class_hash.update(c.encode('ascii'))
    return base64.b64encode(class_hash.digest()).decode('ascii')


FIT_RUN_ID_CACHE_ENTRY = 'fit_run_id'


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

    @staticmethod
    def _validate_cache(state, node):
        path = path_from_uri(state['uri'])

        if not os.path.exists(path):
            raise ConstructorError("while constructing a DataFrame", node.start_mark,
                                   "file %s not found" % path,
                                   node.start_mark)

        mtime_orig = state['mtime']
        mtime = os.path.getmtime(path)

        if mtime_orig != mtime:
            raise ConstructorError("while constructing a DataFrame", node.start_mark,
                                   "file modified. original ts: %s, but found: %s" % (mtime_orig, mtime),
                                   node.start_mark)
        return path

    def _construct_df(self, node, as_dask=False):
        state = self.construct_mapping(node)
        path = self._validate_cache(state, node)

        if tuple(state.items()) in self.df_load_cache:
            rv = self.df_load_cache[tuple(state.items())]
            return rv

        ext = os.path.splitext(path)[1].lower()

        if as_dask:
            module = dd
        else:
            module = pd

        read_any = {
            '.parquet': module.read_parquet,
            '.h5': partial(module.read_hdf, key='df'),
            '.hdf': partial(module.read_hdf, key='df'),
        }.get(ext)

        if read_any is None:
            raise DaskPipesException("Unknown file format: {}".format(ext))

        rv = read_any(path)

        if ext == '.parquet' and self.recover_categories and as_dask:
            for col in rv.columns:
                try:
                    rv[col] = rv[col].cat.set_categories(rv[col].head(1).dtype.categories)
                except AttributeError:
                    pass

        self.df_load_cache[tuple(state.items())] = rv
        self.df_dump_cache[id(rv)] = (state, weakref.ref(rv))
        return rv

    def construct_dask_dataframe(self, node):
        return self._construct_df(node, as_dask=True)

    def construct_pandas_dataframe(self, node):
        return self._construct_df(node, as_dask=False)


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

    def _represent_dataframe(self, data: Union[dd.DataFrame, pd.DataFrame], ext='.parquet'):
        dump_data, ref = self.df_dump_cache.get(id(data), (None, None))
        if ref is None or ref() is None:
            fname = '{}{}'.format(self.get_fname(), ext)
            fname_full = os.path.join(self.directory, fname)
            if ext == '.parquet':
                data.to_parquet(fname_full, engine='fastparquet', compression='gzip')
            elif ext == '.h5' or ext == '.hdf':
                data.to_hdf(fname_full, 'df', mode='w', format='table', append=False)
            dump_data = {
                'uri': path_to_uri(fname_full),
                'mtime': os.path.getmtime(fname_full)}
            self.df_dump_cache[id(data)] = (dump_data, weakref.ref(data))
        return dump_data

    def represent_dd_dataframe(self, data: dd.DataFrame):
        dump_data = self._represent_dataframe(data, ext='.parquet')
        return self.represent_mapping('tag:yaml.org,2002:dask.DataFrame', dump_data)

    def represent_pd_dataframe(self, data: pd.DataFrame):
        dump_data = self._represent_dataframe(data, ext='.hdf')
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


NodeFitCache = namedtuple('NodeFitCache', ['id', 'run_id', 'node_name',
                                           'time', 'node_input', 'code_hash'])
NodeCache = namedtuple('NodeCache', ['id', 'run_id', 'fit_run_id', 'node_name',
                                     'time', 'node_input', 'node_output', 'code_hash'])
CacheLoadHistory = namedtuple('CacheLoadHistory', ['id', 'run_id', 'node_name', 'time', 'cache_id'])


class CacheMixin(PipelineMixin):

    def __init__(self, cache_dir, recover_categories=True, verbose=True):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.recover_categories = recover_categories

        # Map
        self._fit_cache_id_counter = 0
        # node_name, -> NodeFitCache
        self._fit_cache: Dict[str, NodeFitCache] = dict()

        self._cache_id_counter = 0
        self._cache: List[NodeCache] = list()

        self._cache_loads_id_counter = 0
        self._cache_loads: List[CacheLoadHistory] = list()

        # Used for fast access of latest cache
        self._latest_cache_by_input = dict()

        # Used in yaml loader, dumper to prevent multiple dumps/loads of same dataframe
        self._df_dump_cache = dict()
        self._df_load_cache = dict()

        self.verbose = verbose

    def __str__(self):
        return "<CacheMixin {}>".format('./' + os.path.relpath(self.cache_dir, '.'))

    def __repr__(self):
        return str(self)

    @property
    def fit_cache(self):
        return pd.DataFrame(self._fit_cache.values()).set_index('id')

    @property
    def cache(self):
        return pd.DataFrame(self._cache).set_index('id')

    @property
    def cache_loads(self):
        return pd.DataFrame(self._cache_loads).set_index('id')

    def dump(self, data, node_name=None):
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node_name,
            df_dump_cache=self._df_dump_cache,
            df_load_cache=self._df_load_cache,
        )
        rv = yaml.dump(data, Dumper=dumper)
        return rv

    def load(self, data):
        loader = dataframe_loader_factory(
            df_dump_cache=self._df_dump_cache,
            df_load_cache=self._df_load_cache,
        )
        return yaml.load(data, Loader=loader)

    def get_fit_args(self, node):
        rv = self._fit_cache[node.name]
        if node._meta['cache'].get(FIT_RUN_ID_CACHE_ENTRY) != rv.run_id:
            raise DaskPipesException("Fit cache is invalid, %s changed" % FIT_RUN_ID_CACHE_ENTRY)
        return rv.node_input

    def add_fit_record(self, run_id: str, node: NodeBase, node_input: Any, time=None):
        node_input = self.dump(node_input, node_name=node.name)
        rec = NodeFitCache(
            id=self._fit_cache_id_counter,
            run_id=run_id,
            node_name=node.name,
            time=time or datetime.now(),
            node_input=node_input,
            code_hash=hash_class_code(node)
        )
        self._fit_cache_id_counter += 1
        # Make cache records
        self._fit_cache[node.name] = rec

    def add_record(self, run_id: str, fit_run_id: str, node: NodeBase, node_input: Any, node_output: Any, time=None):
        """
        Dump node_input and node_output to yaml and save
        :param run_id: For loggins purposes
        :param fit_run_id: Run id when node yielding this specific output was fitted
        :param node_name: Name of node
        :param node_input: Any picklable object
        :param node_output: Any picklable object
        :param time:
        :return:
        """
        node_input = self.dump(node_input, node_name=node.name)
        node_output = self.dump(node_output, node_name=node.name)

        rec = NodeCache(
            id=self._cache_id_counter,
            run_id=run_id,
            fit_run_id=fit_run_id,
            node_name=node.name,
            time=time or datetime.now(),
            node_input=node_input,
            node_output=node_output,
            code_hash=hash_class_code(node),
        )
        self._cache_id_counter += 1

        # Make cache records
        self._cache.append(rec)
        self._latest_cache_by_input[(node.name, node_input)] = rec

    def get_latest_output(self,
                          node_name: str,
                          fit_run_id: str = None,
                          node_input: Any = None,
                          time: datetime = None,
                          run_id: str = None):
        if node_input is None:
            matches = sorted(filter(lambda x: x.node_name == node_name, self._cache), key=lambda x: x.time)
            if len(matches) == 0:
                raise DaskPipesException("Cache for node {} does not exist".format(node_name))
            return self.load(matches[-1].node_output)
        orig_input = self.dump(node_input, node_name=node_name)
        if (node_name, orig_input) not in self._latest_cache_by_input:
            raise DaskPipesException("Cache for node %s and specific input does not exist" % node_name)

        node_cache: NodeCache = self._latest_cache_by_input[(node_name, orig_input)]
        if node_cache.fit_run_id != fit_run_id:
            raise DaskPipesException("Cached data fit_run_id does not match current fit_run_id")

        match = self.load(node_cache.node_output)

        # Append cache load info
        rec = CacheLoadHistory(
            id=self._cache_loads_id_counter,
            run_id=run_id,
            node_name=node_name,
            time=time or datetime.now(),
            cache_id=node_cache.id,
        )
        self._cache_loads_id_counter += 1
        self._cache_loads.append(rec)
        return match

    def _fit(self,
             run,
             func: NodeCallable,
             node: NodeBase,
             node_input: Tuple[Tuple[Any], Dict[str, Any]]):
        cur_input = self.dump(node_input, node_name=node.name)
        try:
            expected_input = self.get_fit_args(node)
            if cur_input == expected_input:
                if self.verbose:
                    logger.info("Skipping fit for {}".format(node.name))
                return
        except (KeyError, DaskPipesException):
            pass
        node._meta['cache'][FIT_RUN_ID_CACHE_ENTRY] = run.run_id
        func(run, node, node_input)
        self.add_fit_record(run.run_id, node, node_input)

    def _transform(self,
                   run,
                   func: NodeCallable,
                   node: NodeBase,
                   node_input: Tuple[Tuple[Any], Dict[str, Any]]):
        try:
            node_output = self.get_latest_output(
                node.name,
                node._meta['cache'].get(FIT_RUN_ID_CACHE_ENTRY),
                node_input,
                run_id=run.run_id,
            )
            logger.info("Loaded cache for {}".format(node.name))
            return node_output
        except (DaskPipesException, ConstructorError) as ex:
            if self.verbose:
                logger.info("{} invalid cache, reason: {}".format(node.name, str(ex)))
            node_output = func(run, node, node_input)
            self.add_record(
                run.run_id,
                node._meta['cache'].get(FIT_RUN_ID_CACHE_ENTRY),
                node,
                node_input,
                node_output,
            )
            node_output = self.get_latest_output(
                node.name,
                node._meta['cache'].get(FIT_RUN_ID_CACHE_ENTRY),
                node_input,
                run_id=run.run_id,
            )
            return node_output

    def reset(self):
        pass
