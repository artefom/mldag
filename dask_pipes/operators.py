from typing import Optional, List
from dask import dataframe as dd
import pandas as pd
from .base import BaseOperator, StorageBase
from dask_pipes.base.columnmapper import DaskColumnMapper
from .exceptions import ProcessingException
import logging
import importlib

logger = logging.getLogger(__name__)

__all__ = ['ColumnMapper']


class ColumnMapper(BaseOperator):
    """
    Class for applying transformations to columns
    """
    COLUMNS_DATA_FILE = 'columns.csv'
    COLUMN_PROCESSORS_FILE = 'column_processors.yaml'

    def __init__(self,
                 task_id,
                 column_mixins: Optional[List[DaskColumnMapper]] = None,
                 categorical_columns=None,
                 **kwargs):
        super().__init__(task_id, **kwargs)
        self.column_mixins = column_mixins if column_mixins is not None else []
        self.categorical_columns = categorical_columns

    def get_categorical_columns(self):
        return self.categorical_columns or []

    def fit(self, params_storage: StorageBase, persist_storage: StorageBase, dataset: dd.DataFrame):
        params_storage['mixins'] = [{'module': mi.__module__,
                                     'class': mi.__class__.__name__}
                                    for mi in self.column_mixins]

        column_data = list()
        cat_columns = self.get_categorical_columns()

        logger.info("Computing column statistics")
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
                except ProcessingException:
                    pass

            column_data.append(col_stats)
        column_data = pd.DataFrame(column_data)
        params_storage['column_data'] = column_data

    @classmethod
    def clear_params(cls, params_folder, persist_folder):
        pass

    @classmethod
    def clear_persist(cls, params_folder, persist_folder):
        pass

    @classmethod
    def transform(cls, params_folder: StorageBase, persist_folder: StorageBase, dataset, run_name=None):

        columns = params_folder['column_data']
        mixins = params_folder['mixins']

        column_mixins = []
        for column_mixin_def in mixins:
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
                except ProcessingException:
                    pass

        return dataset
