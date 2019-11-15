from typing import Dict, Any

import math

import dask_pipes as dp

import numpy as np
import pandas as pd
from dask import dataframe as dd

__all__ = ['StandardScaler', 'FillNa']


class StandardScaler(dp.DaskColumnMapper):
    """
    Used as pre-processing step to make column normally-distributed
    This usually makes training neural networks much easier
    Computes mean, std values on fit and computes (x-mean)/std on train
    Applied only to numeric columns
    """

    def get_stats(self, column: dd.Series, force_categorical=False) -> Dict[str, Any]:
        if not np.issubdtype(column.dtype, np.number) or force_categorical:
            raise dp.ProcessingException("Cannot apply StandardScaler to "
                                         "column {}, dtype {}".format(column.name, column.dtype))
        return {
            'mean': column.mean().compute(),
            'std': column.std().compute()
        }

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        mean = params.get('mean', None)
        std = params.get('std', None)
        if not pd.isna(mean) and not pd.isna(std):
            return (column - mean) / std
        else:
            raise dp.ProcessingException("Could not normalize column %s: no mean, std values" % column.name)


class FillNa(dp.DaskColumnMapper):
    """
    Fills Na values
    For numeric columns, computes median value
    For string columns, uses <Unknown> as fill value
    """

    def __init__(self,
                 str_fillna='<Unknown>',
                 numeric_method='median',
                 ordered_method='-1 or min-1'):
        self.str_fillna = str_fillna
        self.numeric_method = numeric_method
        self.is_numeric = False
        self.ordered_method = ordered_method

    def get_stats(self, column: dd.Series, force_categorical=False) -> Dict[str, Any]:

        if column.isna().sum().compute() == 0:
            raise dp.ProcessingException("Column %s does not contain nan" % column.name)

        if np.issubdtype(column.dtype, np.number) and not force_categorical:
            if self.numeric_method == 'median':
                fillna = column.quantile(0.5).compute()
            else:
                raise NotImplementedError(self.numeric_method)
        elif np.issubdtype(column.dtype, np.number):
            if self.ordered_method == '-1 or min-1':
                min_val = column.min().compute()
                if not np.isfinite(min_val):
                    raise NotImplementedError()
                if min_val > -1:
                    fillna = -1
                else:
                    fillna = math.floor(min_val) - 1
            else:
                raise NotImplementedError(self.ordered_method)
        else:
            fillna = self.str_fillna

        return {
            'fillna': fillna
        }

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> dd.DataFrame:
        fillna = params.get('fillna', None)
        if pd.isna(fillna):
            raise dp.ProcessingException("Could not fillna column %s: no fillna value" % column.name)

        try:
            fillna = float(fillna)
        except ValueError:
            pass

        if isinstance(fillna, float):
            if int(fillna) == fillna:
                fillna = int(fillna)

        # Convert to string if necessary
        if isinstance(fillna, str) and column.dtype != np.dtype('O'):
            raise NotImplementedError()
        else:
            if column.dtype == np.dtype('O') and not isinstance(fillna, str):
                fillna = str(fillna)
            rv = column.fillna(fillna)

        # Try convert to int
        if np.issubdtype(rv.dtype, np.floating):
            # all round
            all_round = rv.apply(lambda x: int(x) == x, meta=(None, 'bool')).min().compute()
            if all_round:
                rv = rv.astype(int)
        return rv
