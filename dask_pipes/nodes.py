from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.api.types

from dask_pipes.base import NodeBase
from dask_pipes.exceptions import DaskPipesException
from .base.pipeline import as_node

__all__ = [
    'RobustCategoriser',
    'DateProcessor',
    'AddNaCategory',
    'AddNaIndicator',
    'as_node',
]


class RobustCategoriser(NodeBase):
    """
    Remove infrequent values
    By default transforms only columns of dtype object
    """

    def __init__(self,
                 name=None,
                 columns=None,
                 include_columns=None,
                 exclude_columns=None,
                 max_categories=100,
                 min_coverage=0.5,
                 replacement='<Other>',
                 drop=True):
        super().__init__(name)
        self.columns = columns
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns
        self.max_categories = max_categories
        self.min_coverage = min_coverage
        self.replacement = replacement
        self.drop = drop

        # Fittable variables
        self.columns_ = None
        self.drop_columns_ = None
        self.frequent_vals_ = None
        self.coverage_ = None

    def get_col_stats(self, col: Union[pd.Series, dd.Series]):
        # Compute value counts
        val_counts = col.value_counts()
        if isinstance(col, dd.Series):
            val_counts = val_counts.compute()
        total_count = val_counts.sum()
        categories = list()
        coverage = 0

        for count in val_counts:
            new_categories = list(val_counts[(val_counts >= count)].index)
            new_coverage = val_counts[new_categories].sum() / total_count
            if new_coverage < 1:
                new_categories = [self.replacement, ] + new_categories
            if len(new_categories) > self.max_categories:
                break
            if new_coverage >= self.min_coverage:
                coverage = new_coverage
                categories = new_categories
                break

        return categories, coverage

    def fit(self, X, y=None):
        self.columns_ = self.columns
        self.frequent_vals_ = dict()
        self.drop_columns_ = list()
        self.coverage_ = dict()

        if self.columns_ is None:
            self.columns_ = X.select_dtypes(object).columns

        include_columns = self.include_columns or list()
        exclude_columns = self.exclude_columns or list()
        self.columns_ = [i for i in X.columns if
                         (i in self.columns_ or i in include_columns)
                         and i not in exclude_columns]

        for col_name in self.columns_:
            categories, coverage = self.get_col_stats(X[col_name])
            if coverage >= self.min_coverage:
                self.frequent_vals_[col_name] = categories
                self.coverage_[col_name] = coverage
            else:
                self.drop_columns_.append(col_name)
        return self

    def transform(self, X, y=None):
        def get_col_transformer(categories, repl_val):
            frequent_val_set = set(categories)

            def check_frequent(x, **kwargs):
                return x if x in frequent_val_set or pd.isna(x) else repl_val

            return check_frequent

        for col_name in X.columns:
            if col_name in self.frequent_vals_:
                col_categories = self.frequent_vals_[col_name]
                if self.coverage_[col_name] < 1:
                    X[col_name] = X[col_name].apply(
                        get_col_transformer(col_categories, self.replacement),
                        meta=(col_name, object))
                cat_type = pd.api.types.CategoricalDtype(sorted(col_categories), ordered=False)
                X[col_name] = X[col_name].astype(cat_type)
            elif col_name in self.drop_columns_:
                if self.drop:
                    X.drop(col_name, axis=1)
        return X


class AddNaCategory(NodeBase):

    def __init__(self, name=None, unknown_cat='<Unknown>'):
        super().__init__(name=name)
        self.unknown_cat = unknown_cat

        # Fittable
        self.categories_ = None

    def fit(self, X, y=None):
        self.categories_ = dict()
        for col_name in X.columns:
            col = X[col_name]
            if pd.api.types.is_categorical(col):
                # Checking if column is nullable
                if col.isna().sum().compute() == 0:
                    pass
                try:
                    if not col.cat.known:
                        raise ValueError("Can only add null category to known categoricals")
                except AttributeError:
                    pass

                old_cats = list(col.dtype.categories)
                new_cats = [self.unknown_cat] + old_cats
                self.categories_[col_name] = pd.CategoricalDtype(sorted(new_cats), ordered=False)
        return self

    def transform(self, X, y=None):
        for col_name, dtype in self.categories_.items():
            X[col_name] = X[col_name].astype(dtype)
        return X


class DateProcessor(NodeBase):

    def __init__(self,
                 name=None,
                 retro_date_mapping=None):
        super().__init__(name=name)

        self.retro_date_mapping = retro_date_mapping
        self.timedeltas_ = None
        self.datetimes_ = None
        self.retro_dates_ = None
        self.datetime_retro_date_mapping_ = None

    @staticmethod
    def is_timedelta(dtype):
        try:
            return np.issubdtype(dtype, np.timedelta64)
        except TypeError:
            return False

    @staticmethod
    def is_datetime(dtype):
        try:
            return np.issubdtype(dtype, np.datetime64)
        except TypeError:
            return False

    def fit(self, X, y=None):
        retro_date_mapping = self.retro_date_mapping or dict()
        self.timedeltas_ = list()
        self.datetimes_ = list()
        self.retro_dates_ = list(set(retro_date_mapping.values()))
        self.datetime_retro_date_mapping_ = dict()

        if len(set(retro_date_mapping.keys()).intersection(self.retro_dates_)) > 0:
            raise DaskPipesException(
                "Columns {} cannot be date and retro_date at same time".format(
                    set(retro_date_mapping.keys()).intersection(self.retro_dates_)))

        for col_name in X.columns:
            if DateProcessor.is_timedelta(X[col_name]):
                self.timedeltas_.append(col_name)
            elif DateProcessor.is_datetime(X[col_name]):
                if col_name in self.retro_dates_:
                    continue
                if col_name not in retro_date_mapping:
                    raise DaskPipesException("Column {} has no assigned retro-date".format(col_name))
                self.datetimes_.append(col_name)
                self.datetime_retro_date_mapping_[col_name] = retro_date_mapping[col_name]

        return self

    def transform(self, X, y=None):
        def to_seconds(x, **kwargs):
            return x.total_seconds() if not pd.isna(x) else None

        for col_name in self.timedeltas_:
            X[col_name] = X[col_name].apply(to_seconds,
                                            meta=(col_name, float))
        for col_name in self.datetimes_:
            retro_date = self.datetime_retro_date_mapping_[col_name]
            X[col_name] = (X[col_name] - X[retro_date]).apply(to_seconds,
                                                              meta=(col_name, float))
        for col in self.retro_dates_:
            X = X.drop(col, axis=1)
        return X


class AddNaIndicator(NodeBase):

    def __init__(self, name=None):
        super().__init__(name=name)

        self.indicator_cols_ = None

    def fit(self, X, y=None):
        self.indicator_cols_ = dict()
        new_cols = set()
        for col_name in X.columns:
            na_col_name = '{}_na'.format(col_name)
            counter = 0
            while na_col_name in X.columns or na_col_name in new_cols:
                na_col_name = '{}_na{}'.format(na_col_name, counter)
                counter += 1
            self.indicator_cols_[col_name] = na_col_name
            new_cols.add(na_col_name)
        return self

    def transform(self, X, y=None):
        for col_name, na_col_name in self.indicator_cols_.items():
            X[na_col_name] = X[col_name].isna().astype(int)
        return X
