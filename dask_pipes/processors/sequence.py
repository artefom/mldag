import os
import shutil
from typing import List, Tuple

from dask import dataframe as dd
from dask.diagnostics import ProgressBar

import dask_pipes as dp
from dask_pipes.utils import dump_yaml, load_yaml, cleanup_empty_dirs

import logging

logger = logging.getLogger(__name__)


class Sequence(dp.DaskProcessor):
    """
    Pipeline for piping multiple processors
    """

    PIPELINE_FILENAME = 'pipeline_params.yaml'
    PERSIST_FILENAME = 'persist.parquet'

    def __init__(self,
                 pipeline: List[Tuple[str, dp.DaskProcessor]],
                 persist=True, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.persist = persist
        self.setup_pipeline()

    def setup_pipeline(self):
        for step_name, processor in self.pipeline:
            processor.set_categorical_columns(self.get_categorical_columns())

    @classmethod
    def get_persist_filename(cls, persist_folder, dataset_name):
        return os.path.join(persist_folder, f"{dataset_name}_{cls.PERSIST_FILENAME}")

    def fit(self, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name):
        assert meta_folder is not None, "Metadata folder is not set! Use meta_folder='path'"

        #  Dump pipeline to file
        pipeline_params = {
            'pipeline_order': [name for name, _ in self.pipeline],
            'persist': self.persist
        }
        dump_yaml(os.path.join(meta_folder, self.PIPELINE_FILENAME), pipeline_params)

        rv = dataset
        for step_name, processor in self.pipeline:
            rv = processor.fit_transform(os.path.join(meta_folder, step_name),
                                         os.path.join(persist_folder, step_name),
                                         rv,
                                         dataset_name)
        del rv

        # Cleanup everything
        dp.cleanup_persist(meta_folder, persist_folder, dataset_name)

    @classmethod
    def load_pipeline(cls, meta_folder):
        pipeline_params = load_yaml(os.path.join(meta_folder, cls.PIPELINE_FILENAME))
        calculation_order = pipeline_params['pipeline_order']
        persist = pipeline_params['persist']
        return calculation_order, persist

    @classmethod
    def cleanup_persist(cls,
                        meta_folder,
                        persist_folder,
                        dataset_name):
        cleanup_empty_dirs(meta_folder)
        cleanup_empty_dirs(persist_folder)

        pipeline, persist = cls.load_pipeline(meta_folder)
        for step_name in pipeline:
            meta_dir = os.path.join(meta_folder, step_name)
            persist_folder = os.path.join(persist_folder, step_name)
            dp.cleanup_persist(meta_dir,
                               persist_folder,
                               dataset_name)

        if persist:
            persist_filename = cls.get_persist_filename(persist_folder, dataset_name)
            if os.path.exists(persist_filename):
                shutil.rmtree(persist_filename)

    @classmethod
    def _transform(cls, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name):
        # Load pipeline
        calculation_order, persist = cls.load_pipeline(meta_folder)

        rv = dataset
        for step_name in calculation_order:
            rv = dp.transform(os.path.join(meta_folder, step_name),
                              os.path.join(persist_folder, step_name),
                              rv,
                              dataset_name)

        if persist:
            persist_filename = cls.get_persist_filename(persist_folder, dataset_name)
            logger.info(f"Saving pipeline output to {persist_filename}")
            with ProgressBar():
                rv.to_parquet(persist_filename)
            rv = dd.read_parquet(persist_filename)

            # Cleanup everything
            for step_name in calculation_order:
                dp.cleanup_persist(os.path.join(meta_folder, step_name),
                                   os.path.join(persist_folder, step_name),
                                   dataset_name)

        # Cleanup empty persist directories
        for d in os.listdir(persist_folder):
            d = os.path.join(persist_folder, d)
            if os.path.isdir(d) and len(os.listdir(d)) == 0:
                shutil.rmtree(d)

        return rv
