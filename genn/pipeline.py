from typing import List, Tuple, Any, Optional
from .base import DaskProcessor, transform
from .utils import *
import dask.dataframe as dd
import importlib
import os


class DaskPipeline(DaskProcessor):
    """
    Pipeline for piping multiple processors
    """

    PIPELINE_FNAME = 'pipeline.yaml'

    def __init__(self, pipeline: List[Tuple[str, DaskProcessor]] = 10, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def fit(self, dataset: dd.DataFrame):
        assert self.get_meta_folder() is not None, "Metadata folder is not set! Use meta_folder='path'"

        #  Dump pipeline to file
        pipeline_order = [name for name, _ in self.pipeline]
        dump_yaml(os.path.join(self.get_meta_folder(), self.PIPELINE_FNAME), pipeline_order)

        rv = dataset
        for step_name, processor in self.pipeline:
            processor_meta_folder = os.path.join(self.get_meta_folder(), step_name)
            processor.set_meta_folder(processor_meta_folder)
            processor.set_categorical_columns(self.get_categorical_columns())
            processor.fit(rv)

            rv_before = rv
            rv = processor.transform(processor.get_meta_folder(), rv)
            assert rv is not rv_before, f"DaskProcessor {str(processor.__class__.__name__)} at step {step_name} " \
                                        f"must not transform DataFrames inplace!"

    @classmethod
    def transform(cls, meta_folder, dataset: dd.DataFrame):
        # Load pipeline
        calculation_order = load_yaml(os.path.join(meta_folder, cls.PIPELINE_FNAME))

        rv = dataset
        for step_name in calculation_order:
            meta_dir = os.path.join(meta_folder, step_name)
            rv = transform(meta_dir, rv)
        return rv
