import os
import dask.dataframe as dd

__all__ = ['read_file', ]


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
