from typing import Union, Dict, Any

import yaml

import os
import shutil
import numpy as np

import dask.dataframe as dd
import importlib

__all__ = ['read_file', 'dump_yaml', 'load_yaml', 'cleanup_empty_dirs', 'try_create_folder', 'import_class', 'is_int']


def is_int(x):
    try:
        if isinstance(x, float) or np.issubdtype(x, np.floating):
            return x == int(x)
        int(x)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def import_class(class_meta):
    module_name = class_meta['module']
    class_name = class_meta['class']
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    cls = getattr(module, class_name)
    return cls


def try_create_folder(folder):
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


def dump_yaml(fname, meta: Union[list, dict]):
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
