from ..base import NodeBase
from ..exceptions import DaskPipesException
from ..utils import replace_signature
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import pandas.api.types
import numpy as np
from types import MethodType
import inspect
from copy import copy

__all__ = ['as_node', 'NodeWrapper', 'SelectDtypes']


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

        for k in arr:
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
                                  if self.matches_dtypes(dataset[col], self.include, self.exclude)]

    def transform(self, dataset: dd.DataFrame):
        return dataset[self.selected_columns_]
