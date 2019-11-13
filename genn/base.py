from typing import Dict, Any, Optional, List, Set, Union

from enum import Enum, auto
import yaml

import logging

import os

import pandas as pd
import dask.dataframe as dd

logger = logging.getLogger(__name__)


class DaskColumnProcessorMixin:

    def get_stats(self, column: dd.Series, categorical=False) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def transform(column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        raise NotImplementedError()


class DaskProcessor:
    META_FILE_NAME = 'meta.yaml'

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

    @staticmethod
    def save_meta(meta_folder, meta: Union[list, dict]):
        meta_file = os.path.join(meta_folder, DaskProcessor.META_FILE_NAME)
        with open(meta_file, 'w') as f:
            yaml.dump(meta, f)

    @staticmethod
    def load_meta(meta_folder) -> Dict[Any, Any]:
        meta_file = os.path.join(meta_folder, DaskProcessor.META_FILE_NAME)
        with open(meta_file, 'r') as f:
            return yaml.load(f)

    @staticmethod
    def dump_tables(meta_folder, params: Dict[str, pd.DataFrame]):
        table_ext = 'csv'
        for table_name, table in params.items():
            file_name = os.path.join(meta_folder, '{}.{}'.format(table_name, table_ext))
            if os.path.exists(file_name):
                logger.warning("Overwriting {}".format(file_name))
            table.to_csv(file_name, index=False)

    @staticmethod
    def load_tables(meta_folder) -> Dict[str, pd.DataFrame]:
        files = os.listdir(meta_folder)
        tables = dict()
        for file in files:
            table_name = os.path.splitext(file)[0]
            path = os.path.join(meta_folder, file)
            if os.path.isdir(path):
                continue
            assert os.path.exists(path), "Could not find file %s" % path
            try:
                tables[table_name] = pd.read_csv(path)
            except Exception as ex:
                raise ValueError("Error reading {}".format(path)) from ex
        return tables

    def fit(self, dataset: dd.DataFrame):
        raise NotImplementedError()

    @classmethod
    def transform(cls, meta_folder, dataset: dd.DataFrame):
        raise NotImplementedError()
