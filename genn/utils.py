import os
import dask.dataframe as dd
import yaml
from typing import Union, Dict, Any
import os

__all__ = ['read_file', 'dump_yaml', 'load_yaml']


def dump_yaml(fname, meta: Union[list, dict]):
    with open(fname, 'w') as f:
        yaml.dump(meta, f)


def load_yaml(fname) -> Dict[Any, Any]:
    with open(fname, 'r') as f:
        return yaml.load(f)


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
    raise ValueError(f"Extension {ext} not recognized")
