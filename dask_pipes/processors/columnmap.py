import importlib
import os
from typing import Optional, List

import pandas as pd
from dask import dataframe as dd

import dask_pipes as dp
from utils import dump_yaml, load_yaml

import logging

logger = logging.getLogger(__name__)


class ColumnMap(dp.DaskProcessor):
    """
    Class for applying transformations to columns
    """
    COLUMNS_DATA_FILE = 'columns.csv'
    COLUMN_PROCESSORS_FILE = 'column_processors.yaml'

    def __init__(self,
                 column_mixins: Optional[List[dp.DaskColumnMapper]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.column_mixins = column_mixins if column_mixins is not None else []

    def fit(self, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name):
        """
        Processes dataset and writes data to disk into meta_folder
        :param meta_folder:
        :param persist_folder:
        :param dataset: dataset to transform
        :param dataset_name: dataset name
        :return:
        """
        total_tables = dict()

        # Dump column mixins
        #  Dump pipeline to file
        column_processors = [{'module': mi.__module__,
                              'class': mi.__class__.__name__}
                             for mi in self.column_mixins]

        dump_yaml(os.path.join(meta_folder, self.COLUMN_PROCESSORS_FILE), column_processors)

        column_data = list()

        cat_columns = self.get_categorical_columns()
        logger.info(f"Computing column statistics")
        for column in dataset.columns:
            logging.info("Processing column {}".format(column))
            col_stats = dict()
            col_stats['column'] = column
            categorical = column in cat_columns

            prev_col = dataset[column]
            for mixin in self.column_mixins:
                try:
                    mixin_res = mixin.get_stats(prev_col, categorical)
                    for k, v in mixin_res.items():
                        assert k not in col_stats, "Parameter %s is already in column stats!" % k
                        col_stats[k] = v
                    prev_col = mixin.transform(prev_col, mixin_res)
                except dp.ProcessingException:
                    pass

            column_data.append(col_stats)

        column_data = pd.DataFrame(column_data)
        column_data.to_csv(os.path.join(meta_folder, self.COLUMNS_DATA_FILE), index=False)

    @classmethod
    def _transform(cls, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name) -> dd.DataFrame:
        """
        Uses previously fitted data from meta_folder to process dataframe and save it to disk
        :param meta_folder:
        :param persist_folder:
        :param dataset_name:
        :param dataset:
        :return: copy of dataset with transformation
        """
        dataset = dataset.copy()
        columns = pd.read_csv(os.path.join(meta_folder, cls.COLUMNS_DATA_FILE))
        meta = load_yaml(os.path.join(meta_folder, cls.COLUMN_PROCESSORS_FILE))

        column_mixins = []
        for column_mixin_def in meta:
            module_name, class_name = column_mixin_def['module'], column_mixin_def['class']
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            column_mixins.append(getattr(module, class_name))

        for loc, row in columns.iterrows():
            column = row['column']
            params = {k: v for k, v in row.iteritems() if k != 'column'}
            for mixin in column_mixins:
                try:
                    dataset[column] = mixin.transform(dataset[column], params)
                except dp.ProcessingException:
                    pass

        return dataset
