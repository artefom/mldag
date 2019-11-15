from typing import Dict, Any, Optional, Set

from datetime import datetime

import yaml

import logging

import os

import pandas as pd
import dask.dataframe as dd

from .utils import *

import importlib

logger = logging.getLogger(__name__)

CLASS_DEF_FILE = 'class.yaml'

__all__ = ['DaskProcessor', 'DaskColumnMapper', 'cleanup_persist', 'transform']


class DaskColumnMapper:
    """
    Base class for arbitrary column processing, used by base.`dask_pipes.processors.ColumnMap`
    """

    def get_stats(self, column: dd.Series, force_categorical=False) -> Dict[str, Any]:
        """
        Compute necessary stats for transform step
        :param column: column to apply transformation to
        :param force_categorical: does current column should be considered categorical? (default: False)
        :return: dictionary of arbitrary statistics to be written to file.
        may contain numbers, strings or even pd.DataFrame. This is later passed to transform method
        Example:
        >>> def get_stats(...):
        >>>     ...
        >>>     return {'mean': 1, 'df': pd.DataFrame([[1,2],[3,4]],columns=['a','b'])}
        >>> ...
        >>> def transform(column, params):
        >>>     print(params['mean'])  # prints '1'
        >>>     print(params['df']['a'].iloc[1])  # prints '3'
        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        """
        Transform specific column based on params - previously saved dictionary of statistics
        This method must return dask future element
        # Standard scale example
        >>> def transform(...):
        >>>     return (column-params['mean'])/params['std']

        Notice:
            transform is a class method, so it be applied without having to create class instance

        :param column: column to apply transformation to
        :param params: Dictionary previously returned by get_stats
        :return: Dask Future of processed column
        """
        raise NotImplementedError()


# DaskPipes

class DaskProcessorMeta(type):
    """
    Meta-class for wrapping fit function
    """

    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, meta_folder, persist_folder, dataset, dataset_name, *args, **kwargs):
            class_fname = CLASS_DEF_FILE

            with open(os.path.join(meta_folder, class_fname), 'w') as f:
                yaml.dump({
                    'module': self.__module__,
                    'class': self.__class__.__name__,
                    'modified': datetime.now().isoformat()
                }, f)

            start_d = datetime.now()
            try_create_folder(meta_folder)
            try_create_folder(persist_folder)
            rv = func(self, meta_folder, persist_folder, dataset.copy(), dataset_name, *args, **kwargs)
            end_d = datetime.now()

            with open(os.path.join(meta_folder, class_fname), 'w') as f:
                yaml.dump({
                    'module': self.__module__,
                    'class': self.__class__.__name__,
                    'modified': datetime.now().isoformat(),
                    'last_run_elapsed_seconds': (end_d - start_d).total_seconds(),
                }, f)
            return rv

        return fit_wrapped

    def __new__(mcs, name, bases, attrs):
        """If the class has a 'run' method, wrap it"""
        if 'fit' in attrs:
            attrs['fit'] = mcs.wrap_fit(attrs['fit'])
        return super().__new__(mcs, name, bases, attrs)


class DaskProcessor(metaclass=DaskProcessorMeta):
    """
    Base class for arbitrary processing code
    """

    def __init__(self, categorical_columns: Optional[Set[str]] = None):
        """
        :param categorical_columns: list of column names to be forced categorical
        """
        self.categorical_columns = None
        self.set_categorical_columns(categorical_columns)

    def set_categorical_columns(self, categorical_columns: Set[str]):
        if categorical_columns is not None:
            if not isinstance(categorical_columns, set):
                categorical_columns = set(categorical_columns)
            self.categorical_columns = categorical_columns
        else:
            self.categorical_columns = set()

    def get_categorical_columns(self) -> Set[str]:
        return self.categorical_columns

    def fit(self, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name: str):
        """
        Fit dataset and store all necessary data in meta_folder and parquet folder
        :param meta_folder: str, path to directory. used for storing information during fit stage
        :param persist_folder: str, path to directory. used for storing intermediate .parquet files
        :param dataset: dataset for fitting
        :param dataset_name: dataset name for unique file names for each dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def cleanup_persist(self,
                        meta_folder,
                        persist_folder,
                        dataset_name):
        """
        Called when there is a need to remove all intermediate files used in dask
        :return:
        """
        pass

    @classmethod
    def _transform(cls, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name) -> dd.DataFrame:
        """
        Transform specified dataset
        :param meta_folder: folder, containing all fitting artifacts
        :param persist_folder: folder, where to write dask persistent tables
        :param dataset_name: used for naming files
        :param dataset: dataset to transform
        :return: transformed datast
        """
        raise NotImplementedError()

    def fit_transform(self, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name: str):
        """
        Fit dataset and store all necessary data in meta_folder and parquet folder
        :param meta_folder: str, path to directory. used for storing information during fit stage
        :param persist_folder: str, path to directory. used for storing intermediate .parquet files
        :param dataset: dataset for fitting
        :param dataset_name: dataset name for unique file names for each dataset
        :return:
        """
        self.fit(meta_folder, persist_folder, dataset, dataset_name)
        return transform(meta_folder, persist_folder, dataset, dataset_name)


def import_processor(meta_folder) -> DaskProcessor:
    class_meta = load_yaml(os.path.join(meta_folder, CLASS_DEF_FILE))
    module_name = class_meta['module']
    class_name = class_meta['class']
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    cls: DaskProcessor = getattr(module, class_name)
    return cls


def cleanup_persist(meta_folder,
                    persist_folder,
                    dataset_name):
    cls = import_processor(meta_folder)
    return cls.cleanup_persist(meta_folder,
                               persist_folder,
                               dataset_name)


def transform(meta_folder: str, persist_folder: str, dataset: dd.DataFrame, dataset_name: str) -> dd.DataFrame:
    try_create_folder(meta_folder)
    try_create_folder(persist_folder)
    cls = import_processor(meta_folder)
    return cls._transform(meta_folder, persist_folder, dataset.copy(), dataset_name)
