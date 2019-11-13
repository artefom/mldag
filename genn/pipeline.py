from typing import List, Tuple, Any, Optional
from .base import DaskProcessor
import dask.dataframe as dd
import importlib
import os


class DaskPipeline(DaskProcessor):
    """
    Pipeline for piping multiple processors
    """

    def __init__(self, pipeline: List[Tuple[str, DaskProcessor]], **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline

    def fit(self, dataset: dd.DataFrame):
        assert self.get_meta_folder() is not None, "Metadata folder is not set! Use meta_folder='path'"

        #  Dump pipeline to file
        meta = list()
        for step_name, processor in self.pipeline:
            processor_meta = {
                'name': step_name,
                'module': processor.__module__,
                'class': processor.__class__.__name__,
            }
            meta.append(processor_meta)
        self.save_meta(self.get_meta_folder(), meta)

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
        pipeline = []
        meta = cls.load_meta(meta_folder)
        for processor_def in meta:
            step_name = processor_def['name']
            module_name = processor_def['module']
            class_name = processor_def['class']
            pipeline.append((step_name, getattr(importlib.import_module(module_name), class_name)))

        rv = dataset
        for step_name, processor in pipeline:
            meta_dir = os.path.join(meta_folder, step_name)
            rv = processor.transform(meta_dir, rv)
        return rv
