import numpy as np
import torch.utils.data
import dask.dataframe as dd

__all__ = ['DaskDataset']


def dask_generator(ds, x_cols, y_col):
    for part in ds.partitions:
        part = part.compute()
        X = part[x_cols].astype(np.float32)
        y = part[y_col]
        for (loc, row), (y_loc, y) in zip(X.iterrows(), y.iteritems()):
            yield row.values, y


class DaskDataset(torch.utils.data.IterableDataset):
    def __init__(self, fname, target_col):
        super().__init__()
        self.ds = dd.read_parquet(fname)
        if target_col not in self.ds.columns:
            raise ValueError("Column {} not found!".format(target_col))
        self.target_col = target_col
        self.columns = [i for i in self.ds.columns if i != self.target_col]

    def __len__(self):
        return self.ds.shape[0].compute()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return dask_generator(self.ds, self.columns, self.target_col)
        else:  # in a worker process
            raise NotImplementedError()
