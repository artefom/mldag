from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch.utils.data

__all__ = ['DaskDataLoader', 'DaskDataLoaderIter']


def _assert_sorted(arr):
    it = iter(arr)
    v_prev = next(it)
    for v in it:
        assert v_prev < v
        v_prev = v


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
        self.batches: Optional[List[Tuple[int, int, int]]] = None

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

    def infer_batch_split(self, min_batch_size, max_batch_size):

        # Patch format
        self.batches = list()

        for part_i, part in enumerate(self.ds.partitions):

            index = pd.Series(part.index.compute())
            _assert_sorted(index)

            # Dedup index - we want all entries with same index in same batch
            batch_index = np.array(index.drop_duplicates(keep='last').index)

            ind_max = batch_index[-1]

            batch_end = 0
            while batch_end < ind_max:
                batch_beg = batch_end
                batch_end = batch_index[batch_index < batch_beg + max_batch_size][-1]
                batch_size = batch_end - batch_beg

                if batch_size >= min_batch_size:
                    self.batches.append((part_i, batch_beg, batch_end))

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
