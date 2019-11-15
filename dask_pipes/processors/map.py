import importlib
import os
import shutil

from dask import dataframe as dd
from dask.diagnostics import ProgressBar

import dask_pipes as dp

from dask_pipes.utils import dump_yaml, load_yaml

import logging

logger = logging.getLogger(__name__)


class Map(dp.DaskProcessor):
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

    def fit(self, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name):
        meta = {'module': self.processing_module,
                'func': self.processing_func,
                'persist': self.persist}
        dump_yaml(os.path.join(meta_folder, self.PROC_FUNC_DEF), meta)

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
    def _transform(cls, meta_folder, persist_folder, dataset: dd.DataFrame, dataset_name):
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
            logger.info(f"Mapping partitions...")
            with ProgressBar():
                out_df.to_parquet(persist_file)
            out_df = dd.read_parquet(persist_file)

        return out_df
