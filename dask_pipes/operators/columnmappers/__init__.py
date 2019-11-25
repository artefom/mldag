from typing import Dict, Any, List

import math

from dask_pipes.base import ColumnMapperBase
from dask_pipes.exceptions import DaskPipesException
from dask_pipes.utils import is_categorical

import numpy as np
import pandas as pd
from dask import dataframe as dd

__all__ = ['StandardScaler', 'FillNa', 'Coalescence', 'OtherLabelEncoder']


# TODO: Replace standard scaler with dask-ml standard scaler
class StandardScaler(ColumnMapperBase):
    """
    Used as pre-processing step to make column normally-distributed
    This usually makes training neural networks much easier
    Computes mean, std values on fit and computes (x-mean)/std on train
    Applied only to numeric columns
    """

    def get_stats(self, dataset, column: dd.Series) -> Dict[str, Any]:
        if is_categorical(column.dtype):
            raise DaskPipesException("Cannot apply StandardScaler to "
                                     "column {}, dtype {}".format(column.name, column.dtype))
        return {
            'mean': column.mean().compute(),
            'std': column.std().compute()
        }

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        mean = params.get('mean', None)
        std = params.get('std', None)
        if std == 0:
            std = 1
        if mean == 0:
            mean = 0
        if not pd.isna(mean) and not pd.isna(std):
            return [(column - mean) / std, ]
        else:
            raise DaskPipesException("Could not normalize column %s: no mean, std values" % column.name)


class Coalescence(ColumnMapperBase):

    def get_stats(self, dataset, column: dd.Series, force_categorical=False) -> Dict[str, Any]:
        return dict()

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        return [column, ]


class OneHotEncoder(ColumnMapperBase):

    def get_stats(self, dataset: dd.DataFrame, column: dd.Series, force_categorical=False) -> Dict[str, Any]:
        pass

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        pass


class OtherLabelEncoder(ColumnMapperBase):
    def __init__(self,
                 max_labels=20,
                 min_coverage=0.7,
                 other_label='<Other>',
                 skip_cols=None):
        super().__init__(skip_cols=skip_cols)
        self.max_labels = max_labels
        self.min_coverage = min_coverage
        self.other_label = other_label
        self.counts_prefix = '_encodings'

    def get_stats(self, dataset: dd.DataFrame, column: dd.Series) -> Dict[str, Any]:

        if not is_categorical(column.dtype):
            raise DaskPipesException("Cannot apply LabelEncoder to "
                                     "column {}, dtype {}".format(column.name, column.dtype))

        val_counts = column.value_counts().compute()
        val_counts.index.name = 'value'
        val_counts.name = 'count'
        val_counts = val_counts.reset_index()

        top_labels = val_counts[:self.max_labels]
        total_len = val_counts['count'].sum()

        top_coverage = top_labels['count'].sum() / total_len
        if top_coverage < self.min_coverage:
            raise DaskPipesException("Insufficient coverage")

        other_repl = None
        if top_coverage < 1:
            other_repl = self.other_label

        if other_repl is not None:
            top_labels = pd.concat([pd.DataFrame([{'value': 'a', 'count': 1}]), top_labels]) \
                .sort_values(['count', 'value'], ascending=[False, True]).reset_index(drop=True)

        top_labels.index.name = 'encoding'
        encodings_name = '{}_enc'.format(column.name)

        rv = {
            'encodings_name': encodings_name,
            encodings_name: top_labels
        }
        if other_repl is not None:
            rv['other_repl'] = other_repl

        return rv

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        encodings = params.get(params.get('encodings_name'))
        if encodings is None:
            return [column, ]
        encodings_dict = {row.value: row.Index for row in encodings.itertuples()}
        other_repl = params.get('other_repl')

        orig_name = column.name
        if other_repl is not None:
            column = column.apply(lambda x: encodings_dict.get(x, other_repl), meta=(column.name, int))
            column = column.map_partitions(
                lambda d: pd.Series(pd.Categorical.from_codes(d, categories=encodings['value'].values)),
                meta='category')
            column.name = orig_name
        else:
            column = column.apply(lambda x: encodings_dict[x], meta=(column.name, int))
            column = column.map_partitions(
                lambda d: pd.Series(pd.Categorical.from_codes(d, categories=encodings['value'].values)),
                meta='category')
            column.name = orig_name

        return [column, ]


class FillNa(ColumnMapperBase):
    """
    Fills Na values
    For numeric columns, computes median value
    For string columns, uses <Unknown> as fill value
    """

    def __init__(self,
                 str_fillna='<Unknown>',
                 numeric_method='median',
                 ordered_method='out_of_bounds',
                 na_indicator=True,
                 skip_cols=None):
        super().__init__(skip_cols)
        self.str_fillna = str_fillna
        self.numeric_method = numeric_method
        self.is_numeric = False
        self.ordered_method = ordered_method
        self.na_indicator = na_indicator

    def get_stats(self, dataset, column: dd.Series) -> Dict[str, Any]:

        if column.isna().sum().compute() == 0:
            raise DaskPipesException("Column %s does not contain nan" % column.name)

        # Convert boolean to integer
        if np.issubdtype(column.dtype, np.bool_):
            column = column.astype(int)

        if is_categorical(column.dtype):
            fillna = self.str_fillna
        elif np.issubdtype(column.dtype, np.floating):
            if self.numeric_method == 'median':
                fillna = column.quantile(0.5).compute()
            else:
                raise NotImplementedError(self.numeric_method)
        elif np.issubdtype(column.dtype, np.number):
            if self.ordered_method == 'out_of_bounds':
                min_val = column.min().compute()
                if not np.isfinite(min_val):
                    raise NotImplementedError()
                if min_val >= 0:
                    fillna = -1
                else:
                    fillna = math.floor(min_val) - column.std().compute() * 0.05
            else:
                raise NotImplementedError(self.ordered_method)
        else:
            raise ValueError("Unrecognized type: {}".format(column.dtype))

        na_indicator_col_name = None if not self.na_indicator else '{}_na'.format(column.name)

        if na_indicator_col_name in dataset.columns:
            # Rename col if it already contains in dataset
            addition = 2
            na_indicator_col_name_new = None
            while na_indicator_col_name_new is None or na_indicator_col_name_new in dataset.columns:
                na_indicator_col_name_new = '{}_{}'.format(na_indicator_col_name, addition)
                addition += 1
            na_indicator_col_name = na_indicator_col_name_new

        if pd.isna(fillna):
            raise ValueError("Invalid fillna value")

        return {
            'fillna': fillna,
            'na_indicator_col': na_indicator_col_name,
        }

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        fillna = params.get('fillna', None)
        na_indicator_col = params.get('na_indicator_col')

        if pd.isna(fillna):
            raise DaskPipesException("Could not fillna column %s: no fillna value" % column.name)

        try:
            fillna = float(fillna)
        except ValueError:
            pass

        if isinstance(fillna, float):
            if int(fillna) == fillna:
                fillna = int(fillna)

        na_col = None

        # Convert to string if necessary
        if isinstance(fillna, str) and column.dtype != np.dtype('O'):
            column = column.astype(str)

        if column.dtype == np.dtype('O') and not isinstance(fillna, str):
            fillna = str(fillna)

        if na_indicator_col:
            na_col = column.isna()
            na_col.name = na_indicator_col
        rv = column.fillna(fillna)

        # Try convert to int
        if np.issubdtype(rv.dtype, np.floating):
            # all round
            all_round = rv.apply(lambda x: int(x) == x, meta=(None, 'bool')).min().compute()
            if all_round:
                rv = rv.astype(int)

        if na_col is not None:
            return [rv, na_col]
        return [rv, ]
