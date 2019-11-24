from typing import Dict, Any
import os
import yaml
import pandas as pd
import dask.dataframe as dd
import numpy as np
from .exceptions import *

from .utils import *

from .base import Pipeline

__all__ = ['DirPipeline']


def dump_dict(path, suffix, d: dict) -> None:
    csv_dump = {k: v for k, v in d.items() if isinstance(v, pd.DataFrame)}
    parquet_dump = {k: v for k, v in d.items() if isinstance(v, dd.DataFrame)}
    yaml_dump = {k: v for k, v in d.items() if
                 isinstance(v, dict) or
                 isinstance(v, list) or
                 isinstance(v, int) or
                 isinstance(v, float) or
                 isinstance(v, str)}

    # Unrecognised format
    recognised = set(csv_dump.keys()).union(parquet_dump.keys()).union(yaml_dump.keys())
    unrecognised = {k: v for k, v in d.items() if k not in recognised}
    if len(unrecognised) > 0:
        raise DaskPipesException(
            "Unrecognized format: {}".format({k: v.__class__.__name__
                                              for k, v in unrecognised.items()}))

    for csv_name, ds in csv_dump.items():
        ds: pd.DataFrame
        csv_path = os.path.join(path, '{}{}.csv'.format(csv_name, suffix))
        ds.to_csv(csv_path)

    for parquet_name, ds in parquet_dump.items():
        ds: dd.DataFrame
        parquet_path = os.path.join(path, '{}{}.parquet'.format(parquet_name, suffix))
        ds.to_parquet(parquet_path)

    yaml_name = 'params'
    yaml_path = os.path.join(path, '{}{}.yaml'.format(yaml_name, suffix))
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dump, f)


def load_dict(path, suffix) -> Dict[str, Any]:
    rv = dict()
    for fname in os.listdir(path):
        file_path = os.path.join(path, fname)
        file_name, file_ext = os.path.splitext(fname)

        if len(suffix) >= len(file_name):
            continue

        if file_name[-len(suffix):] != suffix:
            continue

        resource_name = file_name[:-len(suffix)]

        if file_ext == '.csv':
            try:
                ds = pd.read_csv(file_path)
                ds.set_index(ds.columns[0])
                rv[resource_name] = ds
            except Exception as ex:
                raise DaskPipesException("Error reading {}".format(file_path)) from ex
        elif file_ext == '.parquet':
            try:
                rv[resource_name] = dd.read_parquet(file_path)
            except Exception as ex:
                raise DaskPipesException("Error reading {}".format(file_path)) from ex
        elif file_ext == '.yaml':
            if resource_name != 'params':
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as ex:
                raise DaskPipesException("Error reading {}".format(file_path)) from ex
            for k, v in params.items():
                rv[k] = v
        else:
            raise DaskPipesException("Unrecognized file extension: {}".format(file_ext))

    return rv


class DirPipeline(Pipeline):

    def __init__(self, params_dir, persist_dir):
        super().__init__()

        # Get abspath immediately after receiving, since working directory can change
        self.params_dir = os.path.abspath(params_dir)
        self.persist_dir = os.path.abspath(persist_dir)

        # Create folders
        self.init_dirs()

    def init_dirs(self):
        try_create_dir(self.params_dir)
        try_create_dir(self.persist_dir)

    def dump_params(self, run_name: str, vertex_name: str, params: Dict[str, Any]):
        dir_name = os.path.join(self.params_dir, vertex_name)
        try_create_dir(dir_name)
        suffix = '_{}'.format(run_name)
        dump_dict(dir_name, suffix, params)

    def dump_outputs(self, run_name: str, vertex_name: str, outputs: Dict[str, Any]):
        dir_name = os.path.join(self.persist_dir, vertex_name)
        try_create_dir(dir_name)
        suffix = '_{}'.format(run_name)
        dump_dict(dir_name, suffix, outputs)

    def load_params(self, run_name: str, vertex_name: str) -> Dict[str, Any]:
        dir_name = os.path.join(self.params_dir, vertex_name)
        suffix = '_{}'.format(run_name)
        return load_dict(dir_name, suffix)

    def load_outputs(self, run_name: str, vertex_name: str) -> Dict[str, Any]:
        dir_name = os.path.join(self.persist_dir, vertex_name)
        suffix = '_{}'.format(run_name)
        return load_dict(dir_name, suffix)
