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
                    'elapsed_seconds': (end_d - start_d).total_seconds(),
                }, f)

            return rv

        return fit_wrapped

    def __new__(cls, name, bases, attrs):
        """If the class has a 'run' method, wrap it"""
        if 'fit' in attrs:
            attrs['fit'] = cls.wrap_fit(attrs['fit'])
        return super().__new__(cls, name, bases, attrs)


class DaskProcessor(metaclass=DaskProcessorMeta):

    def __init__(self,
                 meta_folder: Optional[str] = None,
                 categorical_columns: Optional[Set[str]] = None):
        self.meta_folder = None
        self.categorical_columns = None

        self.set_categorical_columns(categorical_columns)
        self.set_meta_folder(meta_folder)

    def set_categorical_columns(self, categorical_columns: Set[str]):
        if categorical_columns is not None:
            if not isinstance(categorical_columns, set):
                categorical_columns = set(categorical_columns)
            self.categorical_columns = categorical_columns
        else:
            self.categorical_columns = set()

    def get_categorical_columns(self) -> Set[str]:
        return self.categorical_columns

    def set_meta_folder(self, meta_folder):
        self.meta_folder = meta_folder
        if self.meta_folder is not None and not os.path.exists(self.meta_folder):
            try:
                os.mkdir(self.meta_folder)
            except FileNotFoundError:
                raise FileNotFoundError("Could not create {} folder because {} does not exist".format(
                    self.meta_folder,
                    os.path.split(self.meta_folder)[0])) from None

    def get_meta_folder(self):
        return self.meta_folder

    def get_file_loc(self, filename):
        return os.path.join(self.get_meta_folder(),
                            filename)

    def fit(self, dataset: dd.DataFrame):
        raise NotImplementedError()

    @classmethod
    def transform(cls, meta_folder, dataset: dd.DataFrame):
        raise NotImplementedError()


def transform(meta_folder, dataset: dd.DataFrame) -> dd.DataFrame:
    # Read class data from folder
    class_meta = load_yaml(os.path.join(meta_folder, CLASS_DEF_FILE))
    module_name = class_meta['module']
    class_name = class_meta['class']
    cls: DaskProcessor = getattr(importlib.import_module(module_name), class_name)
    return cls.transform(meta_folder, dataset)
