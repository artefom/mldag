from enum import Enum, auto
import numpy as np
import pandas as pd
import os
import dask.dataframe as dd

__all__ = ['is_numeric',
           'read_file', ]


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


def is_numeric(col_name, dtype, cat_cols=None):
    if cat_cols is not None and col_name in cat_cols:
        return False
    return np.issubdtype(dtype, np.number)
