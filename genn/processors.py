import importlib
import os
import shutil
from typing import List, Tuple, Optional

from .exceptions import ProcessingException
from .utils import *
from .base import DaskProcessor, transform, cleanup_persist, DaskColumnProcessorMixin

import pandas as pd
import dask.dataframe as dd

import logging

logger = logging.getLogger(__name__)

__all__ = ['ColumnProcessor', 'PartitionMapper', 'DaskPipeline']


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

    def fit(self, dataset: dd.DataFrame, dataset_name):
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
    def transform(cls,
                  meta_folder,
                  persist_folder,
                  dataset_name,
                  dataset: dd.DataFrame) -> dd.DataFrame:
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


class PartitionMapper(DaskProcessor):
    PROC_FUNC_DEF = 'proc_func_def.yaml'
    PERSIST_FILENAME = 'persist.parquet'

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

    def fit(self, dataset: dd.DataFrame, dataset_name):
        meta = {'module': self.processing_module,
                'func': self.processing_func,
                'persist': self.persist}
        dump_yaml(os.path.join(self.get_meta_folder(), self.PROC_FUNC_DEF), meta)

    @classmethod
    def get_persist_filename(cls, persist_folder, dataset_name):
        return os.path.join(persist_folder, f"{dataset_name}_{cls.PERSIST_FILENAME}")

    @classmethod
    def cleanup_persist(cls,
                        meta_folder,
                        persist_folder,
                        dataset_name):
        persist_file = cls.get_persist_filename(persist_folder, dataset_name)
        if os.path.exists(persist_file):
            shutil.rmtree(persist_file)

    @classmethod
    def transform(cls,
                  meta_folder,
                  persist_folder,
                  dataset_name,
                  dataset: dd.DataFrame):
        # Load processing func
        meta = load_yaml(os.path.join(meta_folder, cls.PROC_FUNC_DEF))
        processing_module = meta['module']
        processing_func = meta['func']
        persist = meta['persist']

        proc_module = importlib.import_module(processing_module)
        proc_module = importlib.reload(proc_module)
        proc_func = getattr(proc_module, processing_func)
        out_df = dataset.map_partitions(proc_func)
        ind_name = out_df.index.name

        out_df = out_df.reset_index().set_index(ind_name)

        if persist:
            persist_file = cls.get_persist_filename(persist_folder, dataset_name)
            out_df.to_parquet(persist_file)
            out_df = dd.read_parquet(persist_file)

        return out_df


class DaskPipeline(DaskProcessor):
    """
    Pipeline for piping multiple processors
    """

    PIPELINE_FILENAME = 'pipeline_params.yaml'
    PERSIST_FILENAME = 'persist.parquet'

    def __init__(self,
                 pipeline: List[Tuple[str, DaskProcessor]],
                 persist=True, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.persist = persist
        self.setup_pipeline()

    def setup_pipeline(self):
        for step_name, processor in self.pipeline:
            processor_meta_folder = os.path.join(self.get_meta_folder(), step_name)
            processor.set_meta_folder(processor_meta_folder)
            processor.set_persist_folder(os.path.join(self.get_persist_folder(), step_name))
            processor.set_categorical_columns(self.get_categorical_columns())

    @classmethod
    def get_persist_filename(cls, persist_folder, dataset_name):
        return os.path.join(persist_folder, f"{dataset_name}_{cls.PERSIST_FILENAME}")

    def fit(self, dataset: dd.DataFrame, dataset_name):
        assert self.get_meta_folder() is not None, "Metadata folder is not set! Use meta_folder='path'"

        #  Dump pipeline to file
        pipeline_params = {
            'pipeline_order': [name for name, _ in self.pipeline],
            'persist': self.persist
        }
        dump_yaml(os.path.join(self.get_meta_folder(), self.PIPELINE_FILENAME), pipeline_params)

        rv_before = None
        rv = dataset
        for step_name, processor in self.pipeline:
            # Fit
            processor.fit(rv, dataset_name)

            # Transform
            rv_before = rv
            rv = transform(processor.get_meta_folder(),
                           processor.get_persist_folder(),
                           dataset_name,
                           rv
                           )
            assert rv is not rv_before, f"DaskProcessor {str(processor.__class__.__name__)} at step {step_name} " \
                                        f"must not transform DataFrames inplace!"
            del rv_before

        del rv
        # Cleanup everything
        for step_name, processor in self.pipeline:
            processor.cleanup_persist(processor.get_meta_folder(),
                                      processor.get_persist_folder(),
                                      dataset_name)

    @classmethod
    def transform(cls,
                  meta_folder,
                  persist_folder,
                  dataset_name,
                  dataset: dd.DataFrame):
        # Load pipeline
        pipeline_params = load_yaml(os.path.join(meta_folder, cls.PIPELINE_FILENAME))
        calculation_order = pipeline_params['pipeline_order']
        persist = pipeline_params['persist']

        rv = dataset
        for step_name in calculation_order:
            rv = transform(os.path.join(meta_folder, step_name),
                           os.path.join(persist_folder, step_name),
                           dataset_name,
                           rv)

        if persist:
            persist_filename = cls.get_persist_filename(persist_folder, dataset_name)
            rv.to_parquet(persist_filename)
            rv = dd.read_parquet(persist_filename)

            # Cleanup everything
            for step_name in calculation_order:
                meta_dir = os.path.join(meta_folder, step_name)
                persist_folder = os.path.join(persist_folder, step_name)
                cleanup_persist(meta_dir,
                                persist_folder,
                                dataset_name)

        return rv
