import importlib
import inspect
import os
import shutil
from collections import namedtuple
from typing import Union, Dict, Any, Callable, List, Type

import dask.dataframe as dd
import numpy as np
import yaml

from dask_pipes.exceptions import DaskPipesException

__all__ = ['read_file', 'dump_yaml', 'load_yaml', 'cleanup_empty_dirs', 'try_create_dir', 'import_class', 'is_int',
           'assert_subclass', 'get_arguments_description', 'get_return_description',
           'ReturnDescription', 'is_categorical', 'replace_signature', 'to_snake_case',
           'INSPECT_EMPTY_PARAMETER']

# noinspection PyProtectedMember
INSPECT_EMPTY_PARAMETER = inspect._empty

VALIDATE_SUBCLASSES = False

ReturnDescription = namedtuple("ReturnDescription", ['name', 'type', 'description'])

RETURN_UNNAMED = 'result'


def to_snake_case(text):
    return ''.join(('_' + i.lower() if i.isupper() else i for i in text if i.isalnum())).lstrip('_')


def replace_signature(func, sign, doc=None):
    """
    Return wrapped function 'func' with specific signature 'sign' and doc string 'doc'

    Parameters
    ----------
    func
        Function to wrap
    sign
        Signature to use
    doc
        Doc string to use

    Returns
    -------
    func_w
        function wrapper
    """
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped.__func__ = func
    wrapped.__signature__ = sign
    wrapped.__doc__ = doc or func.__doc__
    wrapped.__name__ = func.__name__
    return wrapped


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


def get_arguments_description(func: Callable) -> List[inspect.Parameter]:
    return list(inspect.signature(func).parameters.values())


def get_return_description(func: Callable) -> List[ReturnDescription]:
    func_name = func.__qualname__

    return_type = inspect.signature(func).return_annotation

    if isinstance(return_type, tuple) or isinstance(return_type, list):
        if len(return_type) == 0:
            raise DaskPipesException("return type '{}' of {} not understood".format(repr(return_type), func_name))
        rv = list()
        for v in return_type:
            if isinstance(v, tuple) or isinstance(v, list):
                if len(v) != 2:
                    raise DaskPipesException(
                        "Return type annotations must be "
                        "((var_name, var_type1), (var_name2, var_type2))")
                var_name = v[0]
                var_type = v[1]
            elif isinstance(v, str):
                var_name = v
                var_type = INSPECT_EMPTY_PARAMETER
            else:
                raise NotImplementedError()
            if var_name in set((i[0] for i in rv)):
                raise DaskPipesException("duplicate return name: {} of {}".format(var_name, func_name))
            rv.append(ReturnDescription(name=var_name, type=var_type, description=None))
        return rv
    elif isinstance(return_type, dict):
        return [ReturnDescription(name=k, type=v, description=None) for k, v in return_type.items()]
    elif isinstance(return_type, str):
        return [ReturnDescription(name=return_type, type=INSPECT_EMPTY_PARAMETER, description=None)]
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


def dump_yaml(path, meta: Union[list, dict]):
    meta = convert_to_list(meta)
    with open(path, 'w') as f:
        yaml.dump(meta, f)


def load_yaml(path) -> Dict[Any, Any]:
    with open(path, 'r') as f:
        rv = yaml.load(f, Loader=yaml.FullLoader)
        if rv is None:
            return dict()
        return rv


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return None, dd.read_csv(path)
    if ext == '.hdf':
        rv = dd.read_hdf(path, 'df')
        return rv.index.name, rv
    if ext == '.parquet':
        rv = dd.read_parquet(path)
        return rv.index.name, rv
    raise ValueError("Extension %s not recognized" % ext)
