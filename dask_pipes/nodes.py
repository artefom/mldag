from dask_pipes.base import NodeBase
from dask_pipes.exceptions import DaskPipesException
from dask_pipes.utils import replace_signature
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import pandas.api.types
import numpy as np
from types import MethodType
from typing import Optional
import inspect
from copy import copy
import dask_ml.preprocessing

__all__ = ['as_node', 'NodeWrapper', 'SelectDtypes', 'MergeColumns', 'RemoveInfrequent', 'NNPrepare']


class NodeWrapper(NodeBase):
    """
    Wraps BaseEstimator for use in pipeline
    """

    def __init__(self, name=None, estimator=None):
        super().__init__(name)
        self._estimator = None
        self.estimator = estimator

    @property
    def estimator(self):
        if not self._estimator:
            raise DaskPipesException("{} does not have assigned estimator".format(self))
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if estimator is not None:
            if not hasattr(estimator, 'fit'):
                raise DaskPipesException("{} must implement fit".format(estimator))
            if not hasattr(estimator, 'transform'):
                raise DaskPipesException("{} must implement transform".format(estimator))
            fit_sign = inspect.signature(estimator.fit.__func__)
            fit_sign = inspect.Signature(
                parameters=list(fit_sign.parameters.values()),
                return_annotation=NodeWrapper
            )
            self._set_fit_signature(fit_sign, doc=estimator.fit.__doc__)
            self._set_transform_signature(inspect.signature(estimator.transform.__func__),
                                          doc=estimator.transform.__doc__)
            self.__doc__ = estimator.__doc__
        else:
            self.__doc__ = self.__class__.__doc__
            self._reset_transform_signature()
            self._reset_fit_signature()
        self._estimator = estimator

    def _set_fit_signature(self, sign: inspect.Signature, doc=None):
        self.fit = MethodType(replace_signature(self.__class__.fit, sign, doc=doc), self)

    def _set_transform_signature(self, sign: inspect.Signature, doc=None):
        self.transform = MethodType(replace_signature(self.__class__.transform, sign, doc=doc), self)

    def _reset_fit_signature(self):
        self.fit = MethodType(self.__class__.fit, self)
        self.fit.__doc__ = self.__class__.fit.__doc__

    def _reset_transform_signature(self):
        self.transform = MethodType(self.__class__.transform, self)
        self.transform.__doc__ = self.__class__.transform.__doc__

    def __repr__(self):
        if self._estimator is None:
            return '<{}: {}>'.format(self.__class__.__name__, self.name)
        return '<{}({}): {}>'.format(self.__class__.__name__, self.estimator, self.name)

    def fit(self, *args, **kwargs):
        self.estimator.fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        return self.estimator.transform(*args, **kwargs)


def as_node(name, estimator) -> NodeWrapper:
    return NodeWrapper(name=name, estimator=estimator)


class SelectDtypes(NodeBase):
    SUBSET_OPTIONS = [
        'mixed',
        'str_or_na',
        'category',
        'float_na_inf',
        'float_na_int',
    ]

    def __init__(self, name=None, include=None, exclude=None):
        super().__init__(name)
        self._include = list()
        self._exclude = list()

        # fittable parameters
        self.selected_columns_ = None

        self.include = include
        self.exclude = exclude

    @staticmethod
    def _validate_subset(arr):
        if arr is None:
            return list()

        if isinstance(arr, str):
            arr = [arr, ]

        try:
            if SelectDtypes.is_numpy_dtype(arr):
                arr = [arr, ]
        except TypeError:
            pass

        try:
            it = iter(arr)
        except TypeError:
            raise DaskPipesException("Expected list; received {}".format(arr.__class__.__name__))

        for k in it:
            if arr == 'str':
                raise DaskPipesException("Cannot select str columns, 'str_or_na' or 'object' instead")
            if k is not None and k not in SelectDtypes.SUBSET_OPTIONS and not SelectDtypes.is_numpy_dtype(k):
                raise DaskPipesException(
                    "subset must be one of {} or dtype; received {}".format(SelectDtypes.SUBSET_OPTIONS, repr(k)))
        return copy(arr)

    def _check_include_exclude_intersection(self):
        if len(set(self.include).intersection(self.exclude)) != 0:
            raise DaskPipesException(
                "Include and exclude have overlapping elements: {}".format(
                    set(self.include).intersection(self.exclude)))

    @property
    def include(self):
        return copy(self._include)

    @include.setter
    def include(self, include):
        self._include = self._validate_subset(include)
        self._check_include_exclude_intersection()

    @property
    def exclude(self):
        return copy(self._exclude)

    @exclude.setter
    def exclude(self, exclude):
        self._exclude = self._validate_subset(exclude)
        self._check_include_exclude_intersection()

    @staticmethod
    def is_numpy_dtype(dtype):
        try:
            return np.issubdtype(dtype, np.generic) and dtype not in SelectDtypes.SUBSET_OPTIONS
        except TypeError:
            return False

    @staticmethod
    def matches_dtype(column: dd.Series, dtype):
        # Assuming dtype is valid
        if SelectDtypes.is_numpy_dtype(dtype):
            return np.issubdtype(column.dtype, dtype)
        if dtype == 'category':
            return pd.api.types.is_categorical(dtype)
        if dtype == 'float_inf':
            if not np.issubdtype(dtype, np.floating):
                return False
            return da.isinf(column).max().compute()
        if dtype == 'float_int':
            if not np.issubdtype(dtype, np.floating):
                return False
            return (column.fillna(0).astype(int) == column.fillna(0)).min().compute()

        if column.dtype != object:
            return False

        all_str = column.apply(lambda x: isinstance(x, str) or pd.isna(x), meta=(column.name, bool)).min().compute()
        if dtype == 'str_or_na':
            return all_str
        if dtype == 'mixed':
            return not all_str
        raise DaskPipesException("cannot subset by {}; not implemented.".format(dtype))

    def matches_dtypes(self, column, include, exclude):
        included = False
        for dtype in include:
            if self.matches_dtype(column, dtype):
                included = True
                break
        excluded = False
        for dtype in exclude:
            if self.matches_dtype(column, dtype):
                excluded = True
                break
        return included and not excluded

    def fit(self, dataset: dd.DataFrame):
        # Detect mixed types on subset
        self.selected_columns_ = [col for col in dataset.columns
                                  if self.matches_dtypes(dataset[col],
                                                         self.include,
                                                         self.exclude)]
        return self

    def transform(self, dataset: dd.DataFrame) -> {"selected": dd.DataFrame, "other": dd.DataFrame}:
        other_columns = [i for i in dataset.columns if i not in self.selected_columns_]
        return {"selected": dataset[self.selected_columns_],
                "other": dataset[other_columns]}


class MergeColumns(NodeBase):
    """
    Combine multiple dataframes to single
    """

    def __init__(self, name=None):
        super().__init__(name)

    def fit(self, *args):
        return pd.merge(args)

    def transform(self, *args):
        return pd.merge(args)


class RemoveInfrequent(NodeBase):
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
                 min_coverage=0.6,
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

    def get_col_stats(self, col: dd.Series):
        val_counts = col.value_counts().compute()
        top_vals = val_counts.index[:self.max_categories]
        min_count = val_counts[top_vals].min()

        # Select top values by count.
        # Drop last values that have identical counts
        if (val_counts >= min_count).sum() <= self.max_categories:
            top_vals = val_counts[val_counts >= min_count].index
        elif (val_counts > min_count).sum() > 0:
            top_vals = val_counts[val_counts > min_count].index
        else:
            return {}, 0

        val_counts_top = val_counts[top_vals]
        categories = set(top_vals)
        coverage = val_counts_top.sum() / val_counts.sum()
        return categories, coverage

    def fit(self, dataset):
        self.columns_ = self.columns
        self.frequent_vals_ = dict()
        self.drop_columns_ = list()

        if self.columns_ is None:
            self.columns_ = dataset.select_dtypes(object).columns

        include_columns = self.include_columns or list()
        exclude_columns = self.exclude_columns or list()
        self.columns_ = [i for i in dataset.columns if
                         (i in self.columns_ or i in include_columns)
                         and i not in exclude_columns]

        for col_name in self.columns_:
            categories, coverage = self.get_col_stats(dataset[col_name])
            if self.min_coverage <= coverage:
                self.frequent_vals_[col_name] = categories
            else:
                self.drop_columns_.append(col_name)
        return self

    def transform(self, dataset):
        def get_col_transformer(frequent_val_set, repl_val):
            def check_frequent(x):
                return x if x in frequent_val_set or pd.isna(x) else repl_val

            return check_frequent

        for col_name in dataset.columns:
            if col_name in self.frequent_vals_:
                col_frequent_vals = self.frequent_vals_[col_name]

                dataset[col_name] = dataset[col_name].apply(
                    get_col_transformer(col_frequent_vals, self.replacement),
                    meta=(col_name, object))

            elif col_name in self.drop_columns_:
                if self.drop:
                    dataset.drop(col_name, axis=1)
        return dataset


class FillnaMulti(NodeBase):
    NUM_STRATEGIES = ('median', 'mean')
    STR_STRATEGIES = ('const',)

    def __init__(self,
                 name=None,
                 num_fillna_strategy='median',
                 num_fillna_add_indicator=True,
                 str_fillna_strategy='const',
                 str_fillna_const='<Unknown>',
                 str_fillna_add_indicator=False,
                 ):
        super().__init__(name=name)
        self.num_fillna_strategy = num_fillna_strategy
        self.num_fillna_add_indicator = num_fillna_add_indicator
        self.str_fillna_strategy = str_fillna_strategy
        self.str_fillna_const = str_fillna_const
        self.str_fillna_add_indicator = str_fillna_add_indicator

        # Fittable parameters
        self.fillna_vals_ = None

    def get_fillna_val(self, column: dd.Series):
        if np.issubdtype(column.dtype, np.number):
            if self.num_fillna_strategy == 'median':
                return column.quantile(0.5).compute()
            elif self.num_fillna_strategy == 'mean':
                return column.mean().compute()
        elif np.issubdtype(column.dtype, np.datetime64) or np.issubdtype(column.dtype, np.timedelta64):
            if self.num_fillna_strategy == 'median':
                return column.dropna().astype(int).quantile(0.5).compute().astype(column.dtype)
            elif self.num_fillna_strategy == 'mean':
                return column.dropna().astype(int).mean().compute().astype(column.dtype)
        elif column.dtype == object:
            if self.str_fillna_strategy == 'const':
                return self.str_fillna_const
            else:
                raise NotImplementedError()
        elif np.issubdtype(column.dtype, np.bool_):
            return None

        return None

    def fit(self, dataset: dd.DataFrame):
        self.fillna_vals_ = dict()
        for col_name in dataset.columns:
            col = dataset[col_name]
            fillna_val = self.get_fillna_val(col)
            if not pd.isna(fillna_val):
                self.fillna_vals_[col_name] = fillna_val

    def transform(self, dataset: dd.DataFrame):
        for col_name in dataset.columns:
            if col_name in self.fillna_vals_:
                dataset[col_name] = dataset[col_name].fillna(self.fillna_vals_[col_name])
        return dataset


class NNPrepare(NodeBase):
    """
    Prepare arbitrary dataset for feeding into neural network
    """

    def __init__(self,
                 name=None,
                 str_fillna_strategy='const',
                 str_fillna_const='<Unknown>',
                 str_add_indicator=False,
                 num_fillna_strategy='median',
                 num_add_indicator=True,
                 ordinal_cols=None,
                 one_hot_cols=None,
                 ordinal_max_categories=100,
                 ordinal_min_coverage=0.6,
                 one_hot_max_categories=10,
                 one_hot_min_coverage=0.6,
                 one_hot_drop_first=True,
                 one_hot_include_columns=None,
                 one_hot_exclude_columns=None,
                 ordinal_exclude_columns=None,
                 ordinal_include_columns=None,
                 drop=True):
        super().__init__(name=name)
        self.str_fillna_strategy = str_fillna_strategy
        self.str_fillna_const = str_fillna_const
        self.str_add_indicator = str_add_indicator
        self.num_fillna_strategy = num_fillna_strategy
        self.num_add_indicator = num_add_indicator
        self.ordinal_cols = ordinal_cols
        self.one_hot_cols = one_hot_cols
        self.ordinal_max_categories = ordinal_max_categories
        self.ordinal_min_coverage = ordinal_min_coverage
        self.one_hot_max_categories = one_hot_max_categories
        self.one_hot_min_coverage = one_hot_min_coverage
        self.one_hot_drop_first = one_hot_drop_first
        self.one_hot_include_columns = one_hot_include_columns
        self.one_hot_exclude_columns = one_hot_exclude_columns
        self.ordinal_exclude_columns = ordinal_exclude_columns
        self.ordinal_include_columns = ordinal_include_columns
        self.drop = drop

        # Fittable parameters
        self.one_hot_infrequent_remover_: Optional[RemoveInfrequent] = None
        self.ordinal_infrequent_remover_: Optional[RemoveInfrequent] = None
        self.fillna_multi_: Optional[FillnaMulti] = None
        self.dummy_encoder_ = None
        self.ordinal_encoder_ = None
        self.fillna_cols_ = None
        self.one_hot_cols_ = None
        self.ordinal_cols_ = None

    def fit(self, dataset: dd.DataFrame):
        # Fit remove infrequent

        self.one_hot_infrequent_remover_ = RemoveInfrequent(
            name='one_hot',
            max_categories=self.one_hot_max_categories,
            min_coverage=self.one_hot_min_coverage,
            exclude_columns=self.one_hot_exclude_columns,
            include_columns=self.one_hot_include_columns,
            drop=False
        )
        self.one_hot_infrequent_remover_.fit(dataset)

        self.one_hot_cols_ = [i for i in dataset.columns
                              if i in self.one_hot_infrequent_remover_.frequent_vals_]

        test_ordinal_cols = [i for i in dataset.select_dtypes(object).columns if i not in self.one_hot_cols_]
        self.ordinal_infrequent_remover_ = RemoveInfrequent(
            name='ordinal',
            max_categories=self.ordinal_max_categories,
            min_coverage=self.ordinal_min_coverage,
            exclude_columns=self.ordinal_exclude_columns,
            include_columns=self.ordinal_include_columns,
            drop=False,
            columns=test_ordinal_cols)
        self.ordinal_infrequent_remover_.fit(dataset)
        self.ordinal_cols_ = [i for i in dataset.columns
                              if i in self.ordinal_infrequent_remover_.frequent_vals_]

        # Fillna
        self.fillna_multi_ = FillnaMulti(
            name='fillna',
            num_fillna_strategy=self.num_fillna_strategy,
            num_fillna_add_indicator=self.num_add_indicator,
            str_fillna_strategy=self.str_fillna_strategy,
            str_fillna_const=self.str_fillna_const,
            str_fillna_add_indicator=self.str_add_indicator,
        )
        self.fillna_multi_.fit(dataset)

        # Fit one-hot encoder
        one_hot_prep = self.one_hot_infrequent_remover_.transform(dataset)
        one_hot_prep = self.ordinal_infrequent_remover_.transform(one_hot_prep)
        one_hot_prep = self.fillna_multi_.transform(one_hot_prep)
        self.dummy_encoder_ = dask_ml.preprocessing.DummyEncoder(drop_first=self.one_hot_drop_first)
        self.dummy_encoder_.fit(one_hot_prep[self.one_hot_cols_].categorize())

        # Fit ordinal
        self.ordinal_encoder_ = dask_ml.preprocessing.OrdinalEncoder()
        self.ordinal_encoder_.fit(one_hot_prep[self.ordinal_cols_].categorize())

        # Find binary columns

        # Fit standard-scaler

    def transform(self, dataset):
        rv = self.one_hot_infrequent_remover_.transform(dataset)
        rv = self.ordinal_infrequent_remover_.transform(rv)
        rv = self.fillna_multi_.transform(rv)
        if len(self.ordinal_cols_) > 0:
            ordinal_cols = self.ordinal_encoder_.transform(rv[self.ordinal_cols_].categorize())
            for colname in ordinal_cols.columns:
                rv[colname] = ordinal_cols[colname]
        if len(self.one_hot_cols_) > 0:
            dummy_cols = self.dummy_encoder_.transform(rv[self.one_hot_cols_].categorize())
            for colname in dummy_cols.columns:
                rv[colname] = dummy_cols[colname]
            for colname in self.one_hot_cols_:
                rv = rv.drop(colname, axis=1)
        return rv
