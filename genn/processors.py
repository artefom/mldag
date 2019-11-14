from typing import List, Dict, Any, Optional

import os

import yaml

import logging

import importlib

from .utils import *
from .base import DaskColumnProcessorMixin, DaskProcessor
from .exceptions import ProcessingException

from uuid import uuid4

import pandas as pd
import dask.dataframe as dd

logger = logging.getLogger(__name__)


class ColumnProcessor(DaskProcessor):
    """
    Class for applying transformations to columns
    """
    COLUMNS_DATA_FILE = 'columns.csv'
    COLUMN_PROCESSORS_FILE = 'column_processors.yaml'

    def __init__(self,
                 column_mixins: Optional[List[DaskColumnProcessorMixin]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.column_mixins = column_mixins if column_mixins is not None else []

    def fit(self, dataset: dd.DataFrame):
        """
        Processes dataset and writes data to disk into meta_folder
        :param dataset: dataset to transform
        :return:
        """
        total_tables = dict()

        # Dump column mixins
        #  Dump pipeline to file
        column_processors = [{'module': mi.__module__,
                              'class': mi.__class__.__name__}
                             for mi in self.column_mixins]

        dump_yaml(os.path.join(self.get_meta_folder(), self.COLUMN_PROCESSORS_FILE), column_processors)

        column_data = list()

        cat_columns = self.get_categorical_columns()
        for column in dataset.columns:
            print("Processing column {}".format(column))
            col_stats = dict()
            col_stats['column'] = column
            categorical = column in cat_columns
            for mixin in self.column_mixins:
                try:
                    for k, v in mixin.get_stats(dataset[column], categorical).items():
                        assert k not in col_stats, "Parameter %s is already in column stats!" % k
                        col_stats[k] = v
                except ProcessingException:
                    pass

            column_data.append(col_stats)

        column_data = pd.DataFrame(column_data)
        column_data.to_csv(os.path.join(self.get_meta_folder(), self.COLUMNS_DATA_FILE), index=False)

    @classmethod
    def transform(cls, meta_folder, dataset: dd.DataFrame) -> dd.DataFrame:
        """
        Uses previously fitted data from meta_folder to process dataframe and save it to disk
        :param meta_folder:
        :param dataset:
        :return: copy of dataset with transformation
        """
        dataset = dataset.copy()
        columns = pd.read_csv(os.path.join(meta_folder, cls.COLUMNS_DATA_FILE))
        meta = load_yaml(os.path.join(meta_folder, cls.COLUMN_PROCESSORS_FILE))

        column_mixins = []
        for column_mixin_def in meta:
            module_name, class_name = column_mixin_def['module'], column_mixin_def['class']
            column_mixins.append(getattr(importlib.import_module(module_name), class_name))

        for loc, row in columns.iterrows():
            column = row['column']
            params = {k: v for k, v in row.iteritems() if k != 'column'}
            for mixin in column_mixins:
                try:
                    dataset[column] = mixin.transform(dataset[column], params)
                except ProcessingException:
                    pass

        return dataset


class PartitionMapper(DaskProcessor):
    PROC_FUNC_DEF = 'proc_func_def.yaml'
    PERSIST_FILE = 'persist.parquet'

    def __init__(self,
                 processing_module: str,
                 process_func: str,
                 persist=True,
                 **kwargs):
        """
        :param processing_module: path to processing python module, containing process_func
        :param process_func: foo(ds: pd.DataFrame) -> pd.DataFrame - processing function from processing_module
        """
        super().__init__(**kwargs)
        self.processing_module = processing_module
        self.processing_func = process_func
        self.persist = persist

    def fit(self, dataset: dd.DataFrame):
        meta = {'module': self.processing_module,
                'func': self.processing_func,
                'persist': self.persist}
        dump_yaml(os.path.join(self.get_meta_folder(), self.PROC_FUNC_DEF), meta)

    @classmethod
    def transform(cls, meta_folder, dataset: dd.DataFrame):
        # Load processing func
        meta = load_yaml(os.path.join(meta_folder, cls.PROC_FUNC_DEF))
        processing_module = meta['module']
        processing_func = meta['func']
        persist = meta['persist']

        proc_module = importlib.import_module(processing_module)
        proc_func = getattr(proc_module, processing_func)
        out_df = dataset.map_partitions(proc_func)
        ind_name = out_df.index.name

        out_df = out_df.reset_index().set_index(ind_name)

        if persist:
            persist_file = os.path.join(meta_folder, cls.PERSIST_FILE)
            out_df.to_parquet(persist_file)
            out_df = dd.read_parquet(persist_file)

        return out_df
