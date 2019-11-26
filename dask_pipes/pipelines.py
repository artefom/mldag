from typing import Dict, Any, Optional
import os
import yaml
import pandas as pd
import dask.dataframe as dd
from .exceptions import DaskPipesException

from .utils import try_create_dir

from .base import PipelineBase

__all__ = ['DirPipeline']


def dump_dict(path, suffix, d: dict) -> Dict[str, str]:
    csv_dump = {k: v for k, v in d.items() if isinstance(v, pd.DataFrame)}
    parquet_dump = {k: v for k, v in d.items() if isinstance(v, dd.DataFrame)}
    yaml_dump = {k: v for k, v in d.items() if
                 isinstance(v, dict) or
                 isinstance(v, list) or
                 isinstance(v, int) or
                 isinstance(v, float) or
                 isinstance(v, str)}

    out_files = dict()

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
        out_files[csv_name] = csv_path

    for parquet_name, ds in parquet_dump.items():
        ds: dd.DataFrame
        parquet_path = os.path.join(path, '{}{}.parquet'.format(parquet_name, suffix))
        ds.to_parquet(parquet_path)
        out_files[parquet_name] = parquet_path

    if len(yaml_dump) > 0:
        yaml_name = 'params'
        yaml_path = os.path.join(path, '{}{}.yaml'.format(yaml_name, suffix))
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_dump, f)
        out_files[yaml_name] = yaml_path

    return out_files


def _read_csv(resource_name, file_path):
    try:
        ds = pd.read_csv(file_path)
        ds.set_index(ds.columns[0])
        return {resource_name: ds}
    except Exception as ex:
        raise DaskPipesException("Error reading {}".format(file_path)) from ex


def _read_parquet(resource_name, file_path):
    try:
        return {resource_name: dd.read_parquet(file_path)}
    except Exception as ex:
        raise DaskPipesException("Error reading {}".format(file_path)) from ex


def _read_yaml(resource_name, file_path):
    if resource_name != 'params':
        return dict()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as ex:
        raise DaskPipesException("Error reading {}".format(file_path)) from ex


ext_reader_mapping = {
    '.csv': _read_csv,
    '.parquet': _read_parquet,
    '.yaml': _read_yaml,
    '.yml': _read_yaml
}


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

        if file_ext not in ext_reader_mapping:
            raise DaskPipesException("Unknown extension: {}".format(file_ext))
        for k, v in ext_reader_mapping[file_ext](resource_name, file_path).items():
            assert k not in rv
            rv[k] = v

    return rv


class DirPipeline(PipelineBase):

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

    def dump_params(self, vertex_name: str, params: Dict[str, Any], run_name: Optional[str] = None):
        dir_name = os.path.join(self.params_dir, vertex_name)
        try_create_dir(dir_name)
        suffix = '_{}'.format(run_name) if run_name else ''
        dump_dict(dir_name, suffix, params)

    def dump_outputs(self, vertex_name: str, outputs: Dict[str, Any], run_name: Optional[str] = None):
        dir_name = os.path.join(self.persist_dir, vertex_name)
        try_create_dir(dir_name)
        suffix = '_{}'.format(run_name) if run_name else ''
        dump_dict(dir_name, suffix, outputs)

    def load_params(self, vertex_name: str, run_name: Optional[str] = None) -> Dict[str, Any]:
        dir_name = os.path.join(self.params_dir, vertex_name)
        suffix = '_{}'.format(run_name) if run_name else ''
        return load_dict(dir_name, suffix)

    def load_outputs(self, vertex_name: str, run_name: Optional[str] = None) -> Dict[str, Any]:
        dir_name = os.path.join(self.persist_dir, vertex_name)
        suffix = '_{}'.format(run_name) if run_name else ''
        return load_dict(dir_name, suffix)
