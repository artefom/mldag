from typing import List

import dask.dataframe as dd
import numpy as np
import pandas as pd

__all__ = ['Numeric', 'Nullable', 'Categorical', 'Binary']


def nullable(dataset: dd.DataFrame) -> List[str]:
    """
    Return list of nullable columns

    Parameters
    ----------
    dataset

    Returns
    -------

    """
    na_cols = (dataset.isna().sum() > 0)
    if isinstance(na_cols, dd.core.Scalar):
        na_cols = na_cols.compute()
    return list(na_cols[na_cols].index)


def numeric(dataset: dd.DataFrame) -> List[str]:
    """
    Get all numeric columns

    Parameters
    ----------
    dataset

    Returns
    -------

    """

    def is_numeric(col):
        try:
            return np.issubdtype(col, np.number)
        except TypeError:
            return False

    return [col_name for col_name in dataset.columns if is_numeric(dataset[col_name])]


def categorical(dataset: dd.DataFrame) -> List[str]:
    """
    Get all categorical columns

    Parameters
    ----------
    dataset

    Returns
    -------

    """

    def is_categorical(col):
        return pd.api.types.is_categorical(col)

    return [col_name for col_name in dataset.columns if is_categorical(dataset[col_name])]


def binary(dataset: dd.DataFrame) -> List[str]:
    """
    Return list containing 2 or less values

    Parameters
    ----------
    dataset

    Returns
    -------

    """
    counts = {col_name: dataset[col_name].unique().shape[0]
              for col_name in dataset.columns}
    counts = {k: v.compute() if isinstance(v, dd.core.Scalar) else v for k, v in counts.items()}
    return [i for i in dataset.columns if counts[i] <= 2]


class LogicMeta(type):

    def __and__(self, other):
        return And(self() if isinstance(self, type) else self,
                   other() if isinstance(other, type) else other)

    def __or__(self, other):
        return Or(self() if isinstance(self, type) else self,
                  other() if isinstance(other, type) else other)

    def __invert__(self):
        return Not(self() if isinstance(self, type) else self)

    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and (isinstance(args[0], dd.DataFrame) or isinstance(args[0], pd.DataFrame)):
            return super(LogicMeta, cls).__call__()(args[0])
        return super(LogicMeta, cls).__call__(*args, **kwargs)


class Logic:
    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


class Not(Logic, metaclass=LogicMeta):
    def __init__(self, q):
        self.q = q

    def __call__(self, dataset):
        cols = self.q(dataset)
        return [i for i in dataset.columns if i not in cols]


class Or(Logic, metaclass=LogicMeta):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, dataset):
        cols1 = self.q1(dataset)
        cols2 = self.q2(dataset)
        return [i for i in dataset.columns if i in cols1 or i in cols2]


class And(Logic, metaclass=LogicMeta):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def __call__(self, dataset):
        cols1 = self.q1(dataset)
        cols2 = self.q2(dataset)
        return [i for i in dataset.columns if i in cols1 and i in cols2]


class Numeric(Logic, metaclass=LogicMeta):
    def __call__(self, dataset):
        return numeric(dataset)


class Categorical(Logic, metaclass=LogicMeta):
    def __call__(self, dataset):
        return categorical(dataset)


class Binary(Logic, metaclass=LogicMeta):
    def __call__(self, dataset):
        return binary(dataset)


class Nullable(Logic, metaclass=LogicMeta):
    def __call__(self, dataset):
        return nullable(dataset)
