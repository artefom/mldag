from .base import PipelineBase, StorageBase
from .exceptions import DaskPipesException
from .utils import try_create_folder, load_yaml, dump_yaml
import pandas as pd
import dask.dataframe as dd
import os

__all__ = ['FolderPipeline']


class DirectoryStorage(StorageBase):
    PARAMS_FILE = 'params.yaml'

    def __init__(self, folder: str):
        self.folder = folder
        self.params = None
        try_create_folder(self.folder)
        self.load_params()

    def load_params(self):
        self.params = dict()
        try:
            self.params = load_yaml(self.get_path(DirectoryStorage.PARAMS_FILE))
            for fname in os.listdir(self.folder):
                if fname == DirectoryStorage.PARAMS_FILE or fname[0] == '.':
                    continue
                full_path = self.get_path(fname)
                key, ext = os.path.splitext(fname)
                if key in self.params:
                    raise DaskPipesException("Duplicate parameter for {}".format(key))
                if ext == '.csv':
                    self.params[key] = pd.read_csv(full_path)
                elif ext == '.parquet':
                    self.params[key] = pd.read_parquet(full_path)
                else:
                    raise DaskPipesException("Unknown file extension: {}".format(fname))
        except FileNotFoundError:
            pass

    def get_path(self, x):
        return os.path.join(self.folder, x)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __del__(self):
        self.dump()

    def dump(self):
        yaml_dump = dict()
        for key, value in self.params.items():
            if isinstance(value, dd.DataFrame):
                value.to_parquet(self.get_path('{}.parquet'.format(key)))
            elif isinstance(value, pd.DataFrame):
                value.to_csv(self.get_path('{}.csv'.format(key)), index=False)
            elif isinstance(value, int) or \
                    isinstance(value, float) or \
                    isinstance(value, str) or \
                    isinstance(value, list) or \
                    isinstance(value, dict):
                yaml_dump[key] = value
            else:
                raise TypeError("Unknown type: {}".format(value.__class__.__name__))
        if len(yaml_dump) > 0:
            dump_yaml(self.get_path(DirectoryStorage.PARAMS_FILE), yaml_dump)


class FolderPipeline(PipelineBase):

    def __init__(self, params_folder, persist_folder):
        super().__init__()
        self.params_folder = params_folder
        self.persist_folder = persist_folder

    def get_params_storage(self, task_id):
        try_create_folder(self.params_folder)
        return DirectoryStorage(os.path.join(self.params_folder, task_id))

    def get_persist_storage(self, task_id):
        try_create_folder(self.persist_folder)
        return DirectoryStorage(os.path.join(self.persist_folder, task_id))

    def transform(self, *args, **kwargs):
        pass
