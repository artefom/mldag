import numpy as np
from typing import TYPE_CHECKING, List
import pandas as pd

if TYPE_CHECKING:
    import dask.dataframe as dd

__all__ = ['numeric', 'categorical', 'numeric_binary', 'nullable', 'numeric_nullable']


def numeric_nullable(dataset):
    """
    Get columns containing Na values
    :param dataset:
    :type dataset: dd.DataFrame
    :return:
    """
    numeric_cols = numeric(dataset)
    non_nullable_candidates = numeric_cols
    part = dataset.get_partition(0)
    non_nullable_candidates = {col_name for col_name in non_nullable_candidates
                               if (part[col_name].isna().sum() == 0).compute()}
    non_nullable_candidates = {col_name for col_name in non_nullable_candidates
                               if (dataset[col_name].isna().sum() == 0).compute()}
    rv = [col_name for col_name in dataset.columns if
          col_name not in non_nullable_candidates and col_name in numeric_cols]
    return rv


def nullable(dataset):
    """
    Get columns containing Na values
    :param dataset:
    :type dataset: dd.DataFrame
    :return:
    """
    non_nullable_candidates = set(dataset.columns)
    part = dataset.get_partition(0)
    non_nullable_candidates = {col_name for col_name in non_nullable_candidates
                               if (part[col_name].isna().sum() == 0).compute()}
    non_nullable_candidates = {col_name for col_name in non_nullable_candidates
                               if (dataset[col_name].isna().sum() == 0).compute()}
    return [col_name for col_name in dataset.columns if col_name not in non_nullable_candidates]


def numeric(dataset):
    """
    Get all numeric columns
    :param dataset:
    :type dataset: dd.DataFrame
    :return: List[str]
    """

    def is_numeric(col):
        try:
            return np.issubdtype(col, np.number)
        except TypeError:
            return False

    return [col_name for col_name in dataset.columns if is_numeric(dataset[col_name])]


def categorical(dataset):
    """
    Get all categorical columns
    :param dataset: 
    :type dataset: dd.DataFrame
    :return: List[str]
    """

    def is_categorical(col):
        return pd.api.types.is_categorical(col)

    return [col_name for col_name in dataset.columns if is_categorical(dataset[col_name])]


def numeric_binary(dataset):
    """
    Get numeric columns with only 2 values and no null
    :param dataset:
    :type dataset: dd.DataFrame
    :return: List[str]
    """
    binary_candidates = set(numeric(dataset))
    part = dataset.get_partition(0).compute()
    head = part.head()
    binary_candidates = {col_name for col_name in binary_candidates
                         if len(head[col_name].value_counts()) <= 2 and
                         head[col_name].isna().sum() == 0}
    binary_candidates = {col_name for col_name in binary_candidates
                         if len(part[col_name].value_counts()) <= 2 and
                         part[col_name].isna().sum() == 0}
    return [i for i in dataset.columns if i in binary_candidates]
