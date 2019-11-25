import importlib
from typing import Optional, List, Iterable
from collections import OrderedDict

import pandas as pd
from dask import dataframe as dd

from ..base import OperatorBase, ColumnMapperBase
from ..exceptions import DaskPipesException

import logging

logger = logging.getLogger(__name__)


class ColumnMap(OperatorBase):
    """
    Class for applying transformations to columns
    """

    COLUMNS_DATA_FILE = 'columns.csv'
    COLUMN_PROCESSORS_FILE = 'column_processors.yaml'

    def __init__(self,
                 name,
                 column_mixins: List[ColumnMapperBase]):
        super().__init__(name)
        self.column_mixins = column_mixins

    def fit(self, dataset: dd.DataFrame):
        """
        :param dataset: dataset to transform
        :return:
        """

        column_processors = [{'module': mi.__module__,
                              'class': mi.__class__.__name__}
                             for mi in self.column_mixins]

        logger.info("Computing column statistics")

        col_stats = OrderedDict()

        # ColumnMapper may return dataframe, keep track of which mapper returned which dataframe
        columnmap_ds_inputs = dict()

        additional_data = dict()
        for mixin_id, mixin in enumerate(self.column_mixins):
            skip_cols = set()
            for column in dataset.columns:
                if column in skip_cols:
                    continue

                if column not in col_stats:
                    col_stats[column] = dict()

                logging.info("Processing column {}".format(column))

                col_stats[column]['column'] = column

                try:
                    mixin_res = mixin.get_stats(dataset, dataset[column])
                    for k, v in mixin_res.items():
                        if isinstance(v, pd.DataFrame):
                            df_name = k
                            if df_name in additional_data:
                                new_df_name = None
                                counter = 2
                                while new_df_name is None or df_name in additional_data:
                                    new_df_name = '{}_{}'.format(df_name, counter)
                                    counter += 1
                                df_name = new_df_name

                            assert df_name not in columnmap_ds_inputs
                            if mixin_id not in columnmap_ds_inputs:
                                columnmap_ds_inputs[mixin_id] = dict()

                            columnmap_ds_inputs[mixin_id][k] = df_name

                            # Save dataframe value
                            additional_data[df_name] = v
                        else:
                            assert k not in col_stats[column]
                            col_stats[column][k] = v
                    new_cols = mixin.transform(dataset[column], mixin_res)
                    for col in new_cols:
                        dataset[col.name] = col
                        skip_cols.add(col.name)
                except DaskPipesException:
                    pass

        column_data = pd.DataFrame(list(col_stats.values()))

        rv = {
            'columns': column_data,
            'meta': column_processors,
            'columnmap_ds_inputs': columnmap_ds_inputs
        }
        for k, v in additional_data.items():
            assert k not in rv
            rv[k] = v
        return rv

    @classmethod
    def transform(cls, params, dataset: dd.DataFrame) -> {'result': dd.DataFrame}:
        """
        :param params: Parameters received from fit
        :param dataset: dataset to transform
        :return: copy of dataset with transformation
        """
        columns = params['columns']
        meta = params['meta']
        columnmap_ds_inputs = params['columnmap_ds_inputs']
        additional_params = {k: v for k, v in params.items() if k != 'columns'
                             and k != 'meta' and k != 'column_input_dataframes'}

        dataset = dataset.copy()

        column_mixins = []
        for column_mixin_def in meta:
            module_name, class_name = column_mixin_def['module'], column_mixin_def['class']
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            column_mixins.append(getattr(module, class_name))

        for mixin_id, mixin in enumerate(column_mixins):
            for loc, row in columns.iterrows():
                column = row['column']
                params = {k: v for k, v in row.iteritems() if k != 'column'}

                if mixin_id in columnmap_ds_inputs:
                    for k, v in columnmap_ds_inputs[mixin_id].items():
                        params[k] = additional_params[v]

                try:
                    new_cols = mixin.transform(dataset[column], params)
                    for col in new_cols:
                        dataset[col.name] = col
                except DaskPipesException:
                    pass

        return dataset
