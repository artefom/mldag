from typing import Union, Dict, Any, Callable, List, Type
from collections import namedtuple
import yaml

import os
import shutil
import numpy as np
import inspect

import dask.dataframe as dd
import importlib
from .exceptions import DaskPipesException

__all__ = ['read_file', 'dump_yaml', 'load_yaml', 'cleanup_empty_dirs', 'try_create_dir', 'import_class', 'is_int',
           'assert_subclass', 'get_arguments_description', 'get_return_description', 'ArgumentDescription',
           'ReturnDescription', 'is_categorical']

VALIDATE_SUBCLASSES = False

ArgumentDescription = namedtuple("ArgumentDescription", ['name', 'type', 'description'])
ReturnDescription = namedtuple("ArgumentDescription", ['name', 'type', 'description'])

RETURN_UNNAMED = '<unnamed>'


def is_categorical(type: Type) -> bool:
    try:
        if (np.issubdtype(type, np.number) or
                np.issubdtype(type, np.bool_) or
                np.issubdtype(type, np.datetime64) or
                isinstance(type, int) or
                isinstance(type, bool) or
                isinstance(type, float)):
            return False
    except TypeError:
        pass
    return True


def get_arguments_description(func: Callable, skip_first=1) -> List[ArgumentDescription]:
    fullargspec = inspect.getfullargspec(func)
    argspec = fullargspec.args[skip_first:]
    annotations = func.__annotations__
    rv = [ArgumentDescription(name=k, type=annotations.get(k, object), description=None) for k in argspec]
    return rv


def get_return_description(func: Callable) -> List[ReturnDescription]:
    func_name = func.__qualname__
    return_type = func.__annotations__.get('return', Any)
    if isinstance(return_type, tuple) or isinstance(return_type, list):
        if len(return_type) == 0:
            raise DaskPipesException("return type '{}' of {} not understood".format(repr(return_type), func_name))
        rv = list()
        for v in return_type:
            if isinstance(v, tuple) or isinstance(v, list):
                if len(v) != 2:
                    raise NotImplementedError()
                var_name = v[0]
                var_type = v[1]
            elif isinstance(v, str):
                var_name = v[0]
                var_type = object
            else:
                raise NotImplementedError()
            if var_name in set((i[0] for i in rv)):
                raise DaskPipesException("duplicate return name: {} of {}".format(var_name, func_name))
            rv.append(ArgumentDescription(name=var_name, type=var_type, description=None))
        return rv
    elif isinstance(return_type, dict):
        return [ReturnDescription(name=k, type=v, description=None) for k, v in return_type.items()]
    elif isinstance(return_type, str):
        return [ReturnDescription(name=return_type, type=object, description=None)]
    else:
        return [ReturnDescription(name=RETURN_UNNAMED, type=return_type, description=None)]


def assert_subclass(obj, cls):
    if VALIDATE_SUBCLASSES:
        if not issubclass(obj.__class__, cls):
            raise DaskPipesException(
                "Expected subclass of {}; got {}".format(cls.__name__,
                                                         obj.__class__.__name__))


def is_int(x):
    try:
        if isinstance(x, float) or np.issubdtype(x, np.floating):
            return x == int(x)
        int(x)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    cls = getattr(module, class_name)
    return cls


def try_create_dir(folder):
    if folder is not None and not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except FileNotFoundError:
            raise FileNotFoundError("Could not create {} folder because {} does not exist".format(
                folder,
                os.path.split(folder)[0])) from None


# Cleanup empty meta directories
def cleanup_empty_dirs(folder):
    for d in os.listdir(folder):
        d = os.path.join(folder, d)
        if os.path.isdir(d) and len(os.listdir(d)) == 0:
            shutil.rmtree(d)


def convert_to_list(x):
    if isinstance(x, set) or isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        return [convert_to_list(i) for i in x]
    if isinstance(x, dict):
        return {k: convert_to_list(v) for k, v in x.items()}
    return x


def dump_yaml(fname, meta: Union[list, dict]):
    meta = convert_to_list(meta)
    with open(fname, 'w') as f:
        yaml.dump(meta, f)


def load_yaml(fname) -> Dict[Any, Any]:
    with open(fname, 'r') as f:
        rv = yaml.load(f, Loader=yaml.FullLoader)
        if rv is None:
            return dict()
        return rv


def read_file(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.csv':
        return None, dd.read_csv(file)
    if ext == '.hdf':
        rv = dd.read_hdf(file, 'df')
        return rv.index.name, rv
    if ext == '.parquet':
        rv = dd.read_parquet(file)
        return rv.index.name, rv
    raise ValueError("Extension %s not recognized" % ext)
