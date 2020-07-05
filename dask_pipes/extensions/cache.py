import base64
import hashlib
import inspect
import logging
import os
import pathlib
import types
import urllib
import urllib.parse
import weakref
from collections import namedtuple
from functools import partial
from typing import Any, Dict, Tuple, Optional
from uuid import uuid4

import dask.dataframe as dd
import numpy
import pandas as pd
import sqlalchemy as db
import sqlalchemy.sql.functions as func
import yaml
from yaml.constructor import ConstructorError

from dask_pipes.core import PipelineMixin, NodeCallable, NodeBase
from dask_pipes.exceptions import DaskPipesException

logger = logging.getLogger(__name__)

__all__ = ['path_from_uri', 'path_to_uri',
           'dataframe_dumper_factory',
           'dataframe_loader_factory',
           'CacheMixin']

FIT_RUN_ID_CACHE_ENTRY = 'fit_run_id'
FIT_RUN_ID_TRUE_CACHE_ENTRY = 'fit_run_id_true'
DB_SQLITE_FILENAME = 'cache.db'


def path_to_uri(path):
    return pathlib.Path(path).as_uri()


def path_from_uri(file_uri):
    """
    Rwturns a str object for the supplied file URI.

    Parameters
    ----------
    file_uri : str
        File URI string

    Returns
    -------
    path : str
        absolute path

    """
    path_class = pathlib.PurePath
    windows_path = isinstance(path_class(), pathlib.PureWindowsPath)
    file_uri_parsed = urllib.parse.urlparse(file_uri)
    file_uri_path_unquoted = urllib.parse.unquote(file_uri_parsed.path)
    if windows_path and file_uri_path_unquoted.startswith("/"):
        result = path_class(file_uri_path_unquoted[1:])
    else:
        result = path_class(file_uri_path_unquoted)
    if not result.is_absolute():
        raise ValueError("Invalid file uri {} : resulting path {} not absolute".format(
            file_uri, result))
    return str(result)


_builtins = [int, float, complex, str, tuple, list, bytes,
             bytearray, property, memoryview, set, frozenset, type,
             type(None)]
_builtins += [getattr(types, i) for i in types.__all__ if isinstance(getattr(types, i), type)]
_builtins = set(_builtins)


class CacheStorageBase:
    def record_fit(self, node_name, node_input, code_hash, run_id) -> None:
        raise NotImplementedError()

    def record_transform(self, fit_run_id, node_name, node_input, node_output, run_id) -> None:
        raise NotImplementedError()

    def get_fit_run_id(self, node_name, node_input, code_hash) -> Optional[str]:
        """
        Get fit run id by node_input and code hash

        Parameters
        ----------
        node_name
        node_input
        code_hash

        Returns
        -------
        run_id
            Fit run id

        """
        raise NotImplementedError()

    def get_fit_run_input(self, node_name, run_id) -> Optional[str]:
        """
        Get node fit run input by specific run id

        Parameters
        ----------
        node_name
        run_id

        Returns
        -------

        """
        raise NotImplementedError()

    def get_node_output(self, fit_run_id, node_name, node_input) -> Optional[str]:
        """
        Read node_output by fit_run_id, node_name, node_input

        Parameters
        ----------
        fit_run_id
        node_name
        node_input

        Returns
        -------

        """
        raise NotImplementedError()

    def all_transforms(self) -> pd.DataFrame:
        """
        Get all transform data

        Returns
        -------

        """
        pass

    def all_fits(self) -> pd.DataFrame:
        """
        Get all fit data

        Returns
        -------

        """
        pass


class MemoryStorage(CacheStorageBase):
    """
    Class for storing cache data in-memory
    """

    def __init__(self):
        self.fits = dict()
        self.transforms = dict()

    def record_fit(self, node_name, node_input, code_hash, run_id) -> None:
        self.fits[(node_name, node_input, code_hash)] = run_id

    def record_transform(self, fit_run_id, node_name, node_input, node_output, run_id) -> None:
        self.transforms[(fit_run_id, node_name, node_input)] = node_output

    def get_fit_run_id(self, node_name, node_input, code_hash) -> Optional[str]:
        try:
            return self.fits[(node_name, node_input, code_hash)]
        except KeyError:
            return None

    def get_fit_run_input(self, node_name, run_id) -> Optional[str]:
        for (node_name, node_input), run_id in self.fits.items():
            if node_name == node_name and run_id == run_id:
                return node_input

    def get_node_output(self, fit_run_id, node_name, node_input) -> Optional[str]:
        try:
            return self.transforms[(fit_run_id, node_name, node_input)]
        except KeyError:
            return None


class SQLStorage(CacheStorageBase):
    """
    Class fot storing cache data in database for cross-process availability
    """

    def __init__(self,
                 conn_str,
                 run_id_max_len=255,
                 node_name_max_len=255,
                 create_tables=True):
        self.meta = db.MetaData()
        self.encoding = 'utf-8'
        self.run_id_max_len = run_id_max_len
        self.node_name_max_len = node_name_max_len
        # Define tables
        self.DataTable = db.Table(
            'transform_cache',
            self.meta,
            db.Column('id', db.Integer, primary_key=True, index=True),

            # Cache key
            db.Column('fit_run_id', db.String(self.run_id_max_len), nullable=False),
            db.Column('node_name', db.String(self.node_name_max_len), nullable=False),
            db.Column('node_input', db.Text, nullable=False),

            # Caching data
            db.Column('node_output', db.Text, nullable=False),

            # Tracking & History
            db.Column('created_at', db.TIMESTAMP, server_default=func.now()),
            db.Column('updated_at', db.TIMESTAMP, server_default=func.now(), onupdate=func.current_timestamp()),
            db.Column('run_id', db.String(self.run_id_max_len), nullable=False),

            # Constraints
            db.UniqueConstraint('fit_run_id', 'node_name', 'node_input', name='uq_transform_cache_key'),
            db.UniqueConstraint('run_id', 'node_name', 'node_input', name='uq_fit_cache_run_id'),
            db.Index('ix_transform_cache_key', 'fit_run_id', 'node_name', 'node_input'),
        )

        self.FitRunIds = db.Table(
            # Keys
            'fit_cache',
            self.meta,
            db.Column('id', db.Integer, primary_key=True, index=True),

            # Key - last record with specific node input
            db.Column('node_name', db.String(self.node_name_max_len), nullable=False),
            db.Column('node_input', db.Text, nullable=False),
            db.Column('code_hash', db.String(64), nullable=False),

            # Value
            db.Column('run_id', db.String(self.run_id_max_len), nullable=False),

            # Tracking & History
            db.Column('created_at', db.TIMESTAMP, server_default=func.now()),
            db.Column('updated_at', db.TIMESTAMP, server_default=func.now(), onupdate=func.current_timestamp()),

            # Constraints
            db.UniqueConstraint('node_name', 'node_input', 'code_hash', name='uq_fit_cache_key'),
            db.UniqueConstraint('node_name', 'run_id', name='uq_fit_cache_run_id'),
            db.Index('ix_fit_cache_key', 'node_name', 'node_input', 'code_hash'),

        )

        # Initialize tables
        self.engine = db.create_engine(conn_str)

        if create_tables:
            self.meta.create_all(self.engine)

    def get_fit_run_id(self, node_name, node_input, code_hash) -> Optional[str]:
        assert node_name is not None
        assert node_input is not None
        assert code_hash is not None
        read_query = db.select([self.FitRunIds.c.run_id]).where(
            db.and_(
                self.FitRunIds.c.node_name == node_name,
                self.FitRunIds.c.node_input == node_input,
                self.FitRunIds.c.code_hash == code_hash,
            )
        )
        with self.engine.begin() as conn:
            with conn.begin():
                res = conn.execute(read_query)
                rv = res.fetchone()
                return rv[0] if rv else None

    def get_fit_run_input(self, node_name, run_id) -> Optional[str]:
        assert node_name is not None
        assert run_id is not None
        read_query = db.select([self.FitRunIds.c.node_input, self.FitRunIds.c.updated_at]).where(
            db.and_(
                self.FitRunIds.c.node_name == node_name,
                self.FitRunIds.c.run_id == run_id,
            )
        )
        with self.engine.begin() as conn:
            with conn.begin():
                res = conn.execute(read_query)
                rv = res.fetchone()
                return rv[0] if rv else None

    def record_fit(self, node_name, node_input, code_hash, run_id) -> None:
        assert node_name is not None
        assert node_input is not None
        assert code_hash is not None
        assert run_id is not None
        count_query = db.select([func.count(), ]).select_from(self.FitRunIds).where(
            db.and_(
                self.FitRunIds.c.node_name == node_name,
                self.FitRunIds.c.node_input == node_input,
                self.FitRunIds.c.code_hash == code_hash,
            )
        )

        upd_query = db.update(self.FitRunIds).where(
            db.and_(
                self.FitRunIds.c.node_name == node_name,
                self.FitRunIds.c.node_input == node_input,
                self.FitRunIds.c.code_hash == code_hash,
            )
        ).values(
            run_id=run_id,
        )

        create_query = db.insert(self.FitRunIds).values(
            node_name=node_name,
            node_input=node_input,
            code_hash=code_hash,
            run_id=run_id,
        )
        with self.engine.begin() as conn:
            with conn.begin():
                res = conn.execute(count_query)
                cur_count = res.fetchone()[0]
                if cur_count > 0:
                    conn.execute(upd_query)
                else:
                    conn.execute(create_query)

    def get_node_output(self, fit_run_id, node_name, node_input) -> Optional[str]:
        if fit_run_id is None:
            return None

        assert node_name is not None
        assert node_input is not None
        read_query = db.select([self.DataTable.c.node_output]).where(
            db.and_(
                self.DataTable.c.fit_run_id == fit_run_id,
                self.DataTable.c.node_name == node_name,
                self.DataTable.c.node_input == node_input,
            )
        )
        with self.engine.begin() as conn:
            with conn.begin():
                res = conn.execute(read_query)
                rv = res.fetchone()
                return rv[0] if rv else None

    def record_transform(self, fit_run_id, node_name, node_input, node_output, run_id):
        assert fit_run_id is not None
        assert node_name is not None
        assert node_input is not None
        assert node_output is not None
        count_query = db.select([func.count(), ]).select_from(self.DataTable).where(
            db.and_(
                self.DataTable.c.fit_run_id == fit_run_id,
                self.DataTable.c.node_name == node_name,
                self.DataTable.c.node_input == node_input,
            )
        )

        upd_query = db.update(self.DataTable).where(
            db.and_(
                self.DataTable.c.fit_run_id == fit_run_id,
                self.DataTable.c.node_name == node_name,
                self.DataTable.c.node_input == node_input,
            )
        ).values(
            node_output=node_output,
            run_id=run_id,
        )

        create_query = db.insert(self.DataTable).values(
            fit_run_id=fit_run_id,
            node_name=node_name,
            node_input=node_input,
            node_output=node_output,
            run_id=run_id,
        )
        with self.engine.begin() as conn:
            with conn.begin():
                res = conn.execute(count_query)
                cur_count = res.fetchone()[0]
                if cur_count > 0:
                    conn.execute(upd_query)
                else:
                    conn.execute(create_query)

    def all_transforms(self):
        query = db.select([self.DataTable])
        with self.engine.begin() as conn:
            res = conn.execute(query)
            try:
                rv = pd.DataFrame(res, columns=res.keys())
                return rv.set_index('id')
            finally:
                res.close()

    def all_fits(self):
        query = db.select([self.FitRunIds])
        with self.engine.begin() as conn:
            res = conn.execute(query)
            try:
                rv = pd.DataFrame(res, columns=res.keys())
                return rv.set_index('id')
            finally:
                res.close()


def hash_class_code(cls, depth=2, _cur_depth=0, _class_hash=None, verbose=False):  # noqa: C901
    """
    Hash code for specific class or object
    If object is passed, also considers sub-classes used by instance

    If class is defined in file, uses file class name, file name and modified timestamp for hash
    If class defined at runtime (including jupyter notebooks) uses .__code__ to hash all methods of class

    Parameters
    ----------
    cls : class or class instance
        object to get code hash from
    depth : int
        Recursion depth. if instance is passed, also hash classes used by this instance recursively.
    _cur_depth : int
        Current recursion depth
    _class_hash : hash object, optional
        If specified, _class_hash is updated with new data
    verbose : bool
        indicates if printing verbose info desired

    Returns
    -------
    hash : str
        base64 encoded digested hash

    """
    _class_hash = _class_hash or hashlib.sha256()

    # First, try hashing by class name, file name and file modified timestamp
    try:
        if isinstance(cls, type) or callable(cls):
            path = inspect.getfile(cls)
            mtime = os.path.getmtime(path)
            class_name = cls.__name__
        else:
            path = inspect.getfile(cls.__class__)
            mtime = os.path.getmtime(path)
            class_name = cls.__class__.__name__
        verbose and logger.debug("Cache1: {}".format(mtime))
        _class_hash.update(str(mtime).encode('utf-8'))
        verbose and logger.debug("Cache2: {}".format(path))
        _class_hash.update(path.encode('utf-8'))
        verbose and logger.debug("Cache3: {}".format(class_name))
        _class_hash.update(class_name)
        return base64.b64encode(_class_hash.digest()).decode('ascii')
    except (TypeError, FileNotFoundError):
        pass

    def hash_code(f):
        try:
            code_bytes = f.__code__.co_code
            verbose and logger.debug("Cache4: {}".format(base64.b64encode(code_bytes)))
            _class_hash.update(code_bytes)
            for v in f.__code__.co_consts:
                val_str = str(v)
                if v is not None and type(v) in _builtins and '0x' not in val_str:  # Hash primitives but not pointers
                    verbose and logger.debug("Cache5: {}".format(val_str))
                    _class_hash.update(val_str.encode('utf-8'))
            return True
        except AttributeError:
            return False

    def hash_dict(d):
        for k, v in d.items():
            if not hash_code(v) and k[:2] != '__':
                val_str = str(v)
                if v is not None and type(v) in _builtins and '0x' not in val_str:  # Hash primitives but not pointers
                    verbose and logger.debug("Cache6: {}".format(val_str))
                    _class_hash.update(val_str.encode('utf-8'))

    if hasattr(cls, '__code__'):
        if cls != types.FunctionType:  # noqa: E721
            hash_code(cls)
    elif isinstance(cls, type):
        # Getting hash of class
        if cls not in _builtins:
            hash_dict(cls.__dict__)
    elif isinstance(cls, property):
        hash_code(cls.getter)
        hash_code(cls.setter)
        hash_code(cls.deleter)
    elif cls.__class__ not in _builtins:
        # Step1 - hash class of instance
        hash_dict(cls.__class__.__dict__)
        # Find classes used by instance of object and hash them too
        if _cur_depth < depth - 1:
            sub_classes = set()
            for k, v in cls.__dict__.items():
                if callable(v):
                    sub_classes.add(v)
                elif hasattr(v, '__dict__') and len(v.__dict__) > 0:
                    sub_cls = v.__class__
                    if sub_cls not in _builtins:
                        sub_classes.add(sub_cls)
            sub_class_hashes = set()
            for c in sub_classes:
                sub_class_hashes.add(hash_class_code(c, depth=depth, _cur_depth=_cur_depth + 1))
            # Update in sorted order to make current hash invariant to order
            # (elements in set may have different order from run to run,
            # so without this resulting hash will be different from run to run too)
            for c in sorted(sub_class_hashes):
                verbose and logger.debug("Cache6: {}".format(c.encode('ascii')))
                _class_hash.update(c.encode('ascii'))
    return base64.b64encode(_class_hash.digest()).decode('ascii')


class DataFrameLoader(yaml.SafeLoader):
    """
    Overrides yaml SafeLoader and dumps all dataframes to specific directory, serializing just links to them in yaml
    """

    def __init__(self,
                 *args,
                 df_dump_cache=None,
                 df_load_cache=None,
                 recover_categories=True,
                 **kwargs):
        """

        Parameters
        ----------
        args
        df_dump_cache : dict
            Cache of dumped dataframes
            When dataframe is loaded from disk the dataframe object associates with the file it was read from
            by storing its id in this dictionary
            Later, if dictionary representation of object is needed path of the file it was loaded from is returned
        df_load_cache
        recover_categories : bool
            Parquet files do not support information about column categories
            We could just store category info somewhere else and recover it later
        kwargs
        """
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
        if tuple(state.items()) in self.df_load_cache:
            rv = self.df_load_cache[tuple(sorted(state.items()))]
            return rv

        path = self._validate_cache(state, node)

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

        # TODO: Replace load cache with weak reference
        self.df_load_cache[tuple(sorted(state.items()))] = rv
        self.df_dump_cache[id(rv)] = (state, weakref.ref(rv))
        return rv

    def _construct_numpy_array(self, node):
        state = self.construct_mapping(node)
        if tuple(state.items()) in self.df_load_cache:
            rv = self.df_load_cache[tuple(sorted(state.items()))]
            return rv

        path = self._validate_cache(state, node)
        with open(path, 'rb') as f:
            rv = numpy.load(f)

        # Record value for future use
        # TODO: Replace load cache with weak reference
        self.df_load_cache[tuple(sorted(state.items()))] = rv
        self.df_dump_cache[id(rv)] = (state, weakref.ref(rv))

        return rv

    def construct_dask_dataframe(self, node):
        return self._construct_df(node, as_dask=True)

    def construct_pandas_dataframe(self, node):
        return self._construct_df(node, as_dask=False)

    def construct_numpy_array(self, node):
        return self._construct_numpy_array(node)

    def construct_numpy_float(self, node):
        return numpy.float(self.construct_yaml_float(node))

    def construct_numpy_float128(self, node):
        value = self.construct_scalar(node)
        value = value.replace('_', '').lower()
        sign = +1
        if value[0] == '-':
            sign = -1
        if value[0] in '+-':
            value = value[1:]
        if value == '.inf':
            return sign * self.inf_value
        elif value == '.nan':
            return self.nan_value
        else:
            return sign * numpy.float128(value)


DataFrameLoader.add_constructor('tag:yaml.org,2002:dask.DataFrame',
                                DataFrameLoader.construct_dask_dataframe)
DataFrameLoader.add_constructor('tag:yaml.org,2002:pandas.DataFrame',
                                DataFrameLoader.construct_pandas_dataframe)
DataFrameLoader.add_constructor('tag:yaml.org,2002:numpy.ndarray',
                                DataFrameLoader.construct_numpy_array)
DataFrameLoader.add_constructor('tag:yaml.org,2002:numpy.floating',
                                DataFrameLoader.construct_numpy_float)
DataFrameLoader.add_constructor('tag:yaml.org,2002:numpy.float128',
                                DataFrameLoader.construct_numpy_float128)


class DataFrameDumper(yaml.SafeDumper):
    """
    Overrides yaml.SafeDumper to use links to dataframes created by DataFrameLoader to load dataframes
    """

    def __init__(self,
                 *args,
                 directory=None,
                 node_name=None,
                 run_name=None,
                 tag=None,
                 df_dump_cache=None,
                 **kwargs):
        """

        Parameters
        ----------
        args
        directory
        node_name : str, optional
            Name of node used in naming stored data
            (multiple dumped datasets from one node will contain same part)
        run_name : str, optional
            Name of run used in naming stored data
            (all datasets dumped within one run will have same this prefix)
        tag : str, optional
            Some tag to append to file names so they can be visually identified on the disk
        df_dump_cache : dict
            Cache of dumped objects
            If object was already dumped, cached value is re-used
            Can be shared between multiple instances of DataFrameDumpers or loaders
        kwargs
        """
        if directory is None:
            raise ValueError("Directory of %s cannot be null. "
                             "Use dataframe_dumper_factory to provide additional arguments to dumper"
                             % self.__class__.__name__)
        self.directory = directory

        if df_dump_cache is None:
            self.df_dump_cache = dict()
        else:
            self.df_dump_cache = df_dump_cache

        self.node_name = node_name
        self.run_name = run_name
        self.tag = tag
        super().__init__(*args, **kwargs)

    def ignore_aliases(self, data):
        return True

    def get_path(self, directory, ext):
        counter = 0
        if self.node_name is None:
            return os.path.join(directory, '{}{}'.format(str(uuid4()), ext))

        while True:
            rv = self.node_name
            if self.tag is not None:
                rv = '{}-{}'.format(rv, self.tag)
            if self.run_name is not None:
                rv = '{}-{}'.format(self.run_name[:5], rv)
            if counter == 0:
                rv = os.path.join(directory, '{}{}{}'.format(rv, '_{}'.format(counter) if counter else '', ext))
            if not os.path.exists(rv):
                break
            counter += 1

        return rv

    @staticmethod
    def _represent_path(path):
        """
        Get representation of dataframe on disc as dictionary
        Returns its location and modification time

        Parameters
        ----------
        path : str
            Path to dataframe to represent

        Returns
        -------
        file_info : dict
            Dictionary containing location of represented dataframe and its modification time
            {'uri': ..., 'mtime': ...}
        """
        return {
            'uri': path_to_uri(path),
            'mtime': os.path.getmtime(path),
        }

    # TODO: Write doc, mention caching mechanism and why should we associate files
    @staticmethod
    def associate(df_dump_cache, path, obj):
        """
        Associates object with file so when dictionary representation of object is needed
        we don't need to return whole binary data but 'path' containing all necessary data instead

        Parameters
        ----------
        df_dump_cache : dict
            Dictionary of weak references
        path : str
            Path to file
        obj : object
            Any object to associate with given file

        Returns
        -------
        file_info : dict
            Dictionary containing location of represented dataframe and its modification time
            {'uri': ..., 'mtime': ...}
        """
        representation_dict = DataFrameDumper._represent_path(path)
        df_dump_cache[id(obj)] = (representation_dict, weakref.ref(obj))
        return representation_dict

    def _dataframe_as_dict(self, df, ext='.parquet'):
        """
        Get dictionary representation of dataframe
        Writes dataframe to disk if necessary

        If dataframe was already dumped to disc (or associated with file)
        write is skipped

        Parameters
        ----------
        df : dask.dataframe.DataFrame or pandas.DataFrame
            Dataframe to get representation for
        ext : str
            File extension for writing dataframe
            (.h5, .hdf, .parquet are supported)
        Returns
        -------
        file_info : dict
            Dictionary containing location of represented dataframe and its modification time
            {'uri': ..., 'mtime': ...}

        """
        # If dataframe was already represented (dumped to disk), do not do it again and just reuse
        if id(df) in self.df_dump_cache:
            return self.df_dump_cache[id(df)][0]

        path_full = self.get_path(self.directory, ext)
        if ext == '.parquet':
            df.to_parquet(path_full, engine='fastparquet', compression='gzip')
        elif ext == '.h5' or ext == '.hdf':
            df.to_hdf(path_full, 'df', mode='w', format='table', append=False)
        return self.associate(self.df_dump_cache, path_full, df)

    def _ndarray_as_dict(self, array):
        """
        Get dictionary representation of numpy array
        Parameters
        ----------
        array : numpy.ndarray
            Array to represent

        Returns
        -------
        file_info : dict
            Dictionary containing information about file with data of array on disc
            {'uri': ..., 'mtime': ...}
        """
        if id(array) in self.df_dump_cache:
            return self.df_dump_cache[id(array)]

        path_full = self.get_path(self.directory, '.npy')
        with open(path_full, 'wb') as f:
            numpy.save(f, array)
        return self.associate(self.df_dump_cache, path_full, array)

    def represent_dd_dataframe(self, data: dd.DataFrame):
        """
        Method used by yaml to dump objects of specific type
        Is registered later by .add_representer
        Parameters
        ----------
        data : dask.dataframe.DataFrame
            Dataframe to represent

        Returns
        -------
        node : MappingNode
            internal yaml object
        """
        # TODO: Add threshold on number of rows for dataframe to be dumped to disk
        representation_dict = self._dataframe_as_dict(data, ext='.parquet')
        # tag:yaml.org,2002: seems to be generic prefix
        return self.represent_mapping('tag:yaml.org,2002:dask.DataFrame', representation_dict)

    def represent_pd_dataframe(self, data: pd.DataFrame):
        """
        Method used by yaml to dump objects of specific type
        Is registered later by .add_representer
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe to represent

        Returns
        -------
        node : MappingNode
            internal yaml object
        """
        # TODO: Add threshold on number of rows for dataframe to be dumped to disk
        representation_dict = self._dataframe_as_dict(data, ext='.hdf')
        # tag:yaml.org,2002: seems to be generic prefix
        return self.represent_mapping('tag:yaml.org,2002:pandas.DataFrame', representation_dict)

    def represent_numpy_array(self, data: numpy.ndarray):
        """
        Method used by yaml to dump objects of specific type
        Is registered later by .add_representer
        Parameters
        ----------
        data : numpy.ndarray
            N-dimensional array to represent

        Returns
        -------
        node : MappingNode
            internal yaml object
        """
        representation_dict = self._ndarray_as_dict(data)
        return self.represent_mapping('tag:yaml.org,2002:numpy.ndarray', representation_dict)

    def represent_numpy_float(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = '.nan'
        elif data == self.inf_value:
            value = '.inf'
        elif data == -self.inf_value:
            value = '-.inf'
        else:
            value = repr(data).lower()
            # Note that in some cases `repr(data)` represents a float number
            # without the decimal parts.  For instance:
            #   >>> repr(1e17)
            #   '1e17'
            # Unfortunately, this is not a valid float representation according
            # to the definition of the `!!float` tag.  We fix this by adding
            # '.0' before the 'e' symbol.
            if '.' not in value and 'e' in value:
                value = value.replace('e', '.0e', 1)
        return self.represent_scalar('tag:yaml.org,2002:numpy.floating', value)

    def represent_numpy_float128(self, data):
        if data != data or (data == 0.0 and data == 1.0):
            value = '.nan'
        elif data == self.inf_value:
            value = '.inf'
        elif data == -self.inf_value:
            value = '-.inf'
        else:
            value = repr(data).lower()
            # Note that in some cases `repr(data)` represents a float number
            # without the decimal parts.  For instance:
            #   >>> repr(1e17)
            #   '1e17'
            # Unfortunately, this is not a valid float representation according
            # to the definition of the `!!float` tag.  We fix this by adding
            # '.0' before the 'e' symbol.
            if '.' not in value and 'e' in value:
                value = value.replace('e', '.0e', 1)
        return self.represent_scalar('tag:yaml.org,2002:numpy.float128', value)


DataFrameDumper.add_representer(dd.DataFrame, DataFrameDumper.represent_dd_dataframe)
DataFrameDumper.add_representer(pd.DataFrame, DataFrameDumper.represent_pd_dataframe)
DataFrameDumper.add_representer(numpy.ndarray, DataFrameDumper.represent_numpy_array)
DataFrameDumper.add_representer(numpy.float16, DataFrameDumper.represent_numpy_float)
DataFrameDumper.add_representer(numpy.float32, DataFrameDumper.represent_numpy_float)
DataFrameDumper.add_representer(numpy.float64, DataFrameDumper.represent_numpy_float)
DataFrameDumper.add_representer(numpy.float128, DataFrameDumper.represent_numpy_float128)
DataFrameDumper.add_representer(numpy.int8, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.int16, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.int32, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.int64, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.uint8, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.uint16, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.uint32, DataFrameDumper.represent_int)
DataFrameDumper.add_representer(numpy.uint64, DataFrameDumper.represent_int)


def dataframe_loader_factory(df_dump_cache=None, df_load_cache=None, recover_categories=True):
    def get_loader(*args, **kwargs):
        return DataFrameLoader(*args,
                               df_dump_cache=df_dump_cache,
                               df_load_cache=df_load_cache,
                               recover_categories=recover_categories,
                               **kwargs)

    return get_loader


def dataframe_dumper_factory(directory,
                             node_name=None,
                             run_name=None,
                             tag=None,
                             df_dump_cache=None):
    directory = os.path.abspath(directory)

    def get_dumper(*args, **kwargs):
        if not os.path.exists(directory):
            os.mkdir(directory)
        return DataFrameDumper(*args,
                               directory=directory,
                               node_name=node_name,
                               run_name=run_name,
                               tag=tag,
                               df_dump_cache=df_dump_cache,
                               **kwargs)

    return get_dumper


NodeFitCache = namedtuple('NodeFitCache', ['id', 'run_id', 'node_name',
                                           'time', 'node_input', 'code_hash'])
NodeCache = namedtuple('NodeCache', ['id', 'run_id', 'fit_run_id', 'node_name',
                                     'time', 'node_input', 'node_output'])
CacheLoadHistory = namedtuple('CacheLoadHistory', ['id', 'run_id', 'node_name', 'time', 'cache_id'])


class CacheMixin(PipelineMixin):
    """
    Skips fit and transform operations when possible,
    while ensuring data integrity and consistency across the pipeline.

    Uses external db for node output caching (sqlite), allowing for multi-processing usage

    Writes dumped with yaml data into database as text.
    All pd.Dataframes and dd.Dataframes are not serialized to the yaml text, but dumped to disk as
    file paths + timestamp with DataFrame tag

    DataFrameLoader - loads dataframes by file path inside yaml file
    DataFrameDumper - dumps python object to yaml, writing all DataFrames to files

    Node output is keyed by
    node's fit run id - run id at which fit operation was executed. (fit run id is also characterised by code hash)
    node name - name of node
    node input - yaml serialised data of node input

    Fit run ids are keyed by (used to get fit run id by specific code hash for specific input and node)
    Having this fit run id, we can recover which transform outputs were created by parameters fitted in that fit run
    node name,
    node input,
    code hash
    """

    def __init__(self,
                 cache_dir,
                 recover_categories=True,
                 verbose=True,
                 record_code_hash=True,
                 ):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.recover_categories = recover_categories

        conn_str = 'sqlite:///{}'.format(os.path.join(self.cache_dir, DB_SQLITE_FILENAME))
        self.storage = SQLStorage(conn_str)

        self.verbose = verbose

        self.record_code_hash = record_code_hash

        self._df_dump_cache = dict()
        self._df_load_cache = dict()

    def __str__(self):
        return "<CacheMixin {}>".format('./' + os.path.relpath(self.cache_dir, '.'))

    def __repr__(self):
        return str(self)

    @property
    def fit_cache(self):
        return self.storage.all_fits()

    @property
    def cache(self):
        return self.storage.all_transforms()

    # Todo: write doc, mention why  we should  associate files
    def associate(self, path, df):
        path = os.path.abspath(path)
        return DataFrameDumper.associate(self._df_dump_cache, path, df)

    def dump(self, data, node_name=None, run_name=None, tag=None):
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node_name,
            run_name=run_name,
            tag=tag,
            df_dump_cache=self._df_dump_cache,
        )
        rv = yaml.dump(data, Dumper=dumper)
        return rv

    def load(self, data):
        loader = dataframe_loader_factory(
            df_dump_cache=self._df_dump_cache,
            df_load_cache=self._df_load_cache,
        )
        return yaml.load(data, Loader=loader)

    @staticmethod
    def _get_fit_run_id(node, true_run_id=False) -> str:
        if true_run_id:
            return node._meta['cache'].get(FIT_RUN_ID_TRUE_CACHE_ENTRY)
        else:
            return node._meta['cache'].get(FIT_RUN_ID_CACHE_ENTRY)

    @staticmethod
    def _set_fit_run_id(node, run_id, true_run_id=False):
        if true_run_id:
            node._meta['cache'][FIT_RUN_ID_TRUE_CACHE_ENTRY] = run_id
        node._meta['cache'][FIT_RUN_ID_CACHE_ENTRY] = run_id

    def _fit(self,
             run,
             func: NodeCallable,
             node: NodeBase,
             node_input: Tuple[Tuple[Any], Dict[str, Any]]):
        """
        Run fit for node or skip if cached transform data available

        Parameters
        ----------
        run
        func
            Function to execute (node's fit with some wrappers)
        node
            Node of pipeline
        node_input
            Input
        """
        node_name = node.name
        node_input_dump = self.dump(node_input, node_name=node_name, tag='input', run_name=run.run_id)
        code_hash = hash_class_code(node) if self.record_code_hash else 'unknown'
        fit_run_id = self.storage.get_fit_run_id(node_name, node_input_dump, code_hash)

        if fit_run_id is not None:
            # Set fit run id, so transform can search cache by this id
            self._set_fit_run_id(node, fit_run_id)
            self.verbose and logger.warning("Skipping fit for {}".format(node_name))
            return

        func(run, node, node_input)
        # Set 'true' fit run id - true fit run id is only set when node is actually fitted
        # By comparing 'true' fit run id with current fit run id we can check if
        # Node was actually fitted and non-cached transform can be applied
        self._set_fit_run_id(node, run.run_id, true_run_id=True)
        self.storage.record_fit(
            node_name=node_name,
            node_input=node_input_dump,
            code_hash=code_hash,
            run_id=run.run_id,
        )

    def _transform(self,
                   run,
                   func: NodeCallable,
                   node: NodeBase,
                   node_input: Tuple[Tuple[Any], Dict[str, Any]]):

        node_name = node.name
        node_input_dump = self.dump(
            node_input,
            node_name=node_name,
            tag='input',
            run_name=run.run_id,
        )

        # Get fit run id for searching transforms of that fit
        fit_run_id = self._get_fit_run_id(node)

        if fit_run_id is None:
            raise DaskPipesException("Looks like node was not fitted with this cache mixin. Re-run fit")

        node_output_dump = self.storage.get_node_output(
            fit_run_id=fit_run_id,
            node_name=node_name,
            node_input=node_input_dump,
        )

        if node_output_dump is not None:
            self.verbose and logger.warning("Using transform cache for {}".format(node_name))
        else:
            true_fit_run_id = self._get_fit_run_id(node, true_run_id=True)

            if true_fit_run_id is None or true_fit_run_id != fit_run_id:
                # TODO: maybe we can load old node by its fit run id?
                raise NotImplementedError("Node was not fitted for current parameters (fit skipped due to caching). "
                                          "Cannot call .transform directly. "
                                          "Transform cache does not exist. "
                                          "True fit run id: {}, current: {} "
                                          "Delete cache and try again.".format(true_fit_run_id, fit_run_id))

            node_output_dump = self.dump(
                func(run, node, node_input),
                node_name=node_name,
                tag='output',
                run_name=run.run_id,
            )

            # Use 'true' run id here since we're not using cache.
            # Rather, calling true transform on current fit parameters
            self.storage.record_transform(
                fit_run_id=true_fit_run_id,
                node_name=node_name,
                node_input=node_input_dump,
                node_output=node_output_dump,
                run_id=run.run_id,
            )

        return self.load(node_output_dump)
