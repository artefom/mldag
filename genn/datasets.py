import numpy as np
import torch.utils.data
import dask.dataframe as dd

__all__ = ['DaskDataset']


def dask_generator(ds, x_cols, y_col, dtype=None):
    if dtype is None:
        dtype = np.float32
    for part in ds.partitions:
        part = part.compute()
        X = part[x_cols].astype(dtype)
        y = part[y_col].astype(dtype)
        for (loc, row), (y_loc, y) in zip(X.iterrows(), y.iteritems()):
            yield row.values, y


class DaskDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 ds: dd.DataFrame,
                 target_col,
                 feature_columns=None,
                 dtype=None):
        super().__init__()
        self.ds = ds
        if target_col not in self.ds.columns:
            raise ValueError("Column {} not found!".format(target_col))
        self.target_col = target_col
        self.dtype = dtype
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [i for i in self.ds.columns if i != self.target_col]

        for col in self.feature_columns:
            if col not in ds.columns:
                raise ValueError("Column {} not found!".format(col))
        if self.target_col in self.feature_columns:
            raise ValueError("Target column {} is also in feature columns!".format(self.target_col))

    def __len__(self):
        return self.ds.shape[0].compute()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return dask_generator(self.ds, self.feature_columns, self.target_col, dtype=self.dtype)
        else:  # in a worker process
            raise NotImplementedError()


class DaskDataLoaderIter:
    def __init__(self, dl):
        self.dl = dl
        self.next_batch = 0

    def __next__(self):
        if self.next_batch >= len(self.dl.batches):
            raise StopIteration()
        rv = self.dl.read_batch(self.next_batch)
        self.next_batch += 1
        return rv


class DaskDataLoader:

    def __init__(self, ds, columns):
        # Get partition sizes
        self.ds = ds
        self.batches = None

        self.column_ids = []
        self.all_columns = []
        for i in columns:
            if isinstance(i, list):
                col_batch = []
                for j in i:
                    col_batch.append(len(self.all_columns))
                    self.all_columns.append(j)
                self.column_ids.append(col_batch)
            else:
                self.column_ids.append(len(self.all_columns))
                self.all_columns.append(i)
        self.cur_part_i = None
        self.cur_part = None
        self.get_partition(0)

    def infer_batch_split(self, batch_size):

        self.batches = list()

        part_lens = list()
        for part_i, part in enumerate(self.ds.partitions):
            print("\rComputing lengths {} of {} ({:.2f}%)".format(part_i + 1, self.ds.npartitions,
                                                                  (part_i + 1) / self.ds.npartitions * 100), end='')
            part_lens.append(part.shape[0].compute())
        print()

        cur_part = 0
        cur_ind = 0
        while cur_part < len(part_lens):
            batch_part = cur_part
            batch_beg = cur_ind
            batch_end = min(part_lens[cur_part], batch_beg + batch_size)
            batch_size_fact = batch_end - batch_beg
            if batch_size_fact < batch_size:
                cur_part += 1
                cur_ind = 0
            else:
                cur_ind = batch_end
                self.batches.append((batch_part, batch_beg, batch_end))

    def set_batch_split(self, batches):
        self.batches = batches

    def get_partition(self, part_i):
        if self.cur_part_i is not None and self.cur_part_i == part_i:
            return self.cur_part
        else:
            self.cur_part_i = part_i
            self.cur_part = self.ds.get_partition(self.cur_part_i)[self.all_columns].compute().values
            return self.cur_part

    def read_batch(self, batch_i):
        part_i, batch_beg, batch_end = self.batches[batch_i]
        part = self.get_partition(part_i)[batch_beg:batch_end]
        rv = tuple((torch.tensor(part[:, i]) for i in self.column_ids))
        return rv

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return DaskDataLoaderIter(self)
