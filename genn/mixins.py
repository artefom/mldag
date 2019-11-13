from typing import Optional, Dict, Any

import pandas as pd

from .base import DaskColumnProcessorMixin
from genn.exceptions import ProcessingException

import numpy as np
from dask import dataframe as dd


class StandardScaler(DaskColumnProcessorMixin):

    def get_stats(self, column: dd.Series, categorical=False) -> Dict[str, Any]:
        if not np.issubdtype(column.dtype, np.number) or categorical:
            raise ProcessingException("Cannot apply StandardScaler to "
                                      "column {}, dtype {}".format(column.name, column.dtype))
        return {
            'mean': column.mean().compute(),
            'std': column.std().compute()
        }

    @staticmethod
    def transform(column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        mean = params.get('mean', None)
        std = params.get('std', None)
        if not pd.isna(mean) and not pd.isna(std):
            return (column - mean) / std
        else:
            raise ProcessingException(f"Could not normalize column {column.name}: no mean, std values")


class FillNa(DaskColumnProcessorMixin):

    def __init__(self,
                 default_cat_nan='<Unknown>',
                 numeric_method='median'):
        self.default_cat_nan = default_cat_nan
        self.numeric_method = numeric_method
        self.is_numeric = False

    def get_stats(self, column: dd.Series, categorical=False) -> Dict[str, Any]:
        fillna = None

        if column.isna().sum().compute() == 0:
            raise ProcessingException(f"Column {column.name} does not contain nan")

        if np.issubdtype(column.dtype, np.number) and not categorical:
            if self.numeric_method == 'median':
                fillna = column.quantile(0.5).compute()
        elif column.dtype == np.dtype('O'):
            fillna = self.default_cat_nan

        return {
            'fillna': fillna
        }

    @staticmethod
    def transform(column: dd.Series, params: Dict[str, Any]) -> dd.DataFrame:
        fillna = params.get('fillna', None)
        if not pd.isna(fillna):
            return column.fillna(fillna)
        else:
            raise ProcessingException(f"Could not fillna column {column.name}: no fillna value")
