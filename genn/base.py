from typing import Dict, Any, Optional, List, Set, Union

from datetime import datetime

from enum import Enum, auto
import yaml

import logging

import os

import pandas as pd
import dask.dataframe as dd

from .utils import *

import importlib

logger = logging.getLogger(__name__)

CLASS_DEF_FILE = 'class.yaml'


class DaskColumnProcessorMixin:

    def get_stats(self, column: dd.Series, categorical=False) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def transform(column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        raise NotImplementedError()


def fit_wrapper(func):
    def fit_wrapped(*args, **kwargs):
        print("Fitting!")
        return func(*args, **kwargs)

    return fit_wrapped


class DaskProcessorMeta(type):
    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, *args, **kwargs):
            class_fname = CLASS_DEF_FILE

            start_d = datetime.now()
            rv = func(self, *args, **kwargs)
            end_d = datetime.now()

            with open(os.path.join(self.meta_folder, class_fname), 'w') as f:
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

    def __init__(self,
                 meta_folder: Optional[str] = None,
                 persist_folder: Optional[str] = None,
                 categorical_columns: Optional[Set[str]] = None,
                 persist=False):
        self.meta_folder = None
        self.categorical_columns = None
        self.persist_folder = None
        self.persist = persist

        self.set_categorical_columns(categorical_columns)
        self.set_meta_folder(meta_folder)
        self.set_persist_folder(persist_folder)

    def set_categorical_columns(self, categorical_columns: Set[str]):
        if categorical_columns is not None:
            if not isinstance(categorical_columns, set):
                categorical_columns = set(categorical_columns)
            self.categorical_columns = categorical_columns
        else:
            self.categorical_columns = set()

    def get_categorical_columns(self) -> Set[str]:
        return self.categorical_columns

    @staticmethod
    def _try_create_folder(folder):
        if folder is not None and not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except FileNotFoundError:
                raise FileNotFoundError("Could not create {} folder because {} does not exist".format(
                    folder,
                    os.path.split(folder)[0])) from None

    def set_meta_folder(self, meta_folder):
        self.meta_folder = meta_folder
        DaskProcessor._try_create_folder(self.meta_folder)

    def get_meta_folder(self):
        return self.meta_folder

    def set_persist_folder(self, persist_folder: str):
        self.persist_folder = persist_folder
        DaskProcessor._try_create_folder(persist_folder)

    def get_persist_folder(self) -> str:
        return self.persist_folder

    def fit(self, dataset: dd.DataFrame, dataset_name: str):
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
    def transform(cls,
                  meta_folder,
                  persist_folder,
                  dataset_name,
                  dataset: dd.DataFrame) -> dd.DataFrame:
        """
        Transform specified dataset
        :param meta_folder: folder, containing all fitting artifacts
        :param persist_folder: folder, where to write dask persistent tables
        :param dataset_name: used for naming files
        :param dataset: dataset to transform
        :return: transformed datast
        """
        raise NotImplementedError()


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


def transform(meta_folder: str,
              persist_folder: str,
              dataset_name: str,
              dataset: dd.DataFrame) -> dd.DataFrame:
    cls = import_processor(meta_folder)
    return cls.transform(meta_folder,
                         persist_folder,
                         dataset_name,
                         dataset)
