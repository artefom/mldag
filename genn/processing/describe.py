import dask.dataframe as dd
import pandas as pd
import logging
import numpy as np
import pickle as pkl
from enum import Enum, auto

logger = logging.getLogger(__name__)

__all__ = ['VariableType', 'ProcessingType', 'numeric_processing_types', 'categorical_processing_types',
           'train_processing_types', 'infer_processing', 'describe']


class VariableType(Enum):
    ORDERED = auto()
    CATEGORICAL = auto()
    NUMERIC = auto()


class ProcessingType(Enum):
    SKIP = auto()
    INDEX = auto()
    PARTITION = auto()
    FLAG = auto()
    ONE_HOT = auto()
    NORMALISED = auto()
    EMBEDDING = auto()


numeric_processing_types = {i.name for i in {ProcessingType.NORMALISED}}

categorical_processing_types = {i.name for i in {ProcessingType.FLAG,
                                                 ProcessingType.ONE_HOT,
                                                 ProcessingType.EMBEDDING}}

train_processing_types = {i.name for i in {ProcessingType.FLAG,
                                           ProcessingType.ONE_HOT,
                                           ProcessingType.NORMALISED,
                                           ProcessingType.EMBEDDING}}

cache_fname = 'cache.pkl'
try:
    with open(cache_fname, 'rb') as f:
        describe_col_cache = pkl.load(f)
except FileNotFoundError:
    describe_col_cache = dict()


def describe_col(col: dd.Series):
    global describe_col_cache
    if col.name in describe_col_cache:
        return describe_col_cache[col.name]

    rv = {
        'dtype': col.dtype,
        'n_unique': col.drop_duplicates().count().compute(),
        'n_na': col.isna().sum().compute(),
    }

    vc = col.value_counts().compute().reset_index()
    vc.columns = [col.name, 'count']
    vc['frac'] = vc['count'] / vc['count'].sum()
    vc['frac_cumsum'] = np.cumsum(vc['frac'])
    rv['max_count'] = vc['count'].max()
    rv['min_count'] = vc['count'].min()
    if col.dtype == object:
        rv['max_len'] = col.str.len().max().compute()
    else:
        rv['max_len'] = None

    try:
        vc = vc[vc['count'] > 900]
        rv['n_unique_cutoff'] = len(vc)
        rv['coverage'] = vc.iloc[-1]['frac_cumsum']
        rv['cutoff_min_count'] = vc.iloc[-1]['count']
    except IndexError:
        rv['coverage'] = 0
        rv['n_unique_cutoff'] = 0
        rv['cutoff_min_count'] = None
    try:
        rv['mean'] = col.mean().compute()
        rv['std'] = col.std().compute()
        rv['median'] = col.dropna().quantile(0.5).compute()
        rv['min'] = col.min().compute()
        rv['max'] = col.max().compute()
    except ValueError:
        pass

    describe_col_cache[col.name] = rv
    with open(cache_fname, 'wb') as f:
        pkl.dump(describe_col_cache, f)
    return rv


def describe(ds: dd.DataFrame):
    ds[ds.index.name] = ds.index
    col_stats = dict()
    for col_i, col in enumerate(ds.columns):
        logger.info("Processing column {} ({} of {})".format(col, col_i + 1, len(ds.columns)))
        col_stats[col] = describe_col(ds[col])
    return pd.DataFrame(col_stats).T


def get_var_type(col_params):
    if np.issubdtype(col_params['params']['dtype'], np.number) and \
            col_params['params'].name not in col_params['cat_features']:
        return VariableType.NUMERIC.name
    return VariableType.CATEGORICAL.name


def is_nullable(col_params):
    if col_params['params']['n_na'] > 0 or \
            (col_params['nullable'] is not None and
             col_params['params'].name in col_params['nullable']):
        return True
    return False


def get_n_unique(col_params):
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['params']['n_unique']
    return None


def get_mean(col_params):
    if get_var_type(col_params) == VariableType.NUMERIC.name:
        return col_params['params']['mean']
    return None


def get_std(col_params):
    if get_var_type(col_params) == VariableType.NUMERIC.name:
        return col_params['params']['std']


def get_n_na(col_params):
    if is_nullable(col_params):
        return col_params['params']['n_na']
    return None


def get_n_unique_cutoff(col_params):
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['params']['n_unique_cutoff']


def get_coverage(col_params):
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['params']['coverage']


def get_fillna_value(col_params):
    if is_nullable(col_params):
        if get_var_type(col_params) == VariableType.CATEGORICAL.name:
            return '<Unknown>'
        if get_var_type(col_params) == VariableType.NUMERIC.name:
            return col_params['params']['median']
        if get_var_type(col_params) == VariableType.ORDERED.name:
            return col_params['params']['median']
        raise ValueError("Cannot get fillna value for column {}".format(col_params['params'].name))
    else:
        return None


def get_class_repl_value(col_params):
    coverage = get_coverage(col_params)
    proc_type = get_processing_type(col_params)
    if coverage is not None and coverage < 1 and \
            proc_type != ProcessingType.PARTITION.name and \
            proc_type != ProcessingType.INDEX.name:
        return '<Other>'
    return None


def get_max_count(col_params):
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['params']['max_count']
    return None


def get_min_count(col_params):
    if get_n_unique_cutoff(col_params) is not None:
        return col_params['params']['cutoff_min_count']
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['min_count']
    return None


def get_max_len(col_params):
    if get_var_type(col_params) == VariableType.CATEGORICAL.name:
        return col_params['params']['max_len']
    return None


def get_processing_type(col_params):
    col = col_params['params']
    n_unique = get_n_unique(col_params)
    if col['max_count'] == 1:
        return ProcessingType.INDEX.name
    if col.name == col_params['ds'].index.name:
        return ProcessingType.PARTITION.name
    if not is_nullable(col_params) and col['n_unique'] == 2:
        return ProcessingType.FLAG.name
    if n_unique is not None and n_unique < 10 and \
            get_coverage(col_params) > 0.5:
        return ProcessingType.ONE_HOT.name
    if get_var_type(col_params) == VariableType.CATEGORICAL.name and get_coverage(col_params) > 0.5:
        return ProcessingType.EMBEDDING.name
    if get_var_type(col_params) == VariableType.NUMERIC.name:
        return ProcessingType.NORMALISED.name
    return ProcessingType.SKIP.name


def get_encoding_ids(col_params):
    ds = col_params['ds']
    p_type = get_processing_type(col_params)
    if p_type == ProcessingType.ONE_HOT.name or \
            p_type == ProcessingType.EMBEDDING.name or \
            p_type == ProcessingType.FLAG.name:
        logger.info(col_params['params'].name)
        val_counts = ds[col_params['params'].name].value_counts().compute()
        top_n_unique = get_n_unique_cutoff(col_params)
        val_counts = val_counts[:top_n_unique]

        val_counts = val_counts.sort_index().reset_index().reset_index()
        val_counts.columns = ['enc', 'value', 'count']

        repl_val = get_class_repl_value(col_params)
        if repl_val is not None:
            col = ds[col_params['params'].name]
            num_other = ((~col.isin(val_counts['value'])) & (~col.isna())).sum().compute()
            other_row = pd.DataFrame([[0, repl_val, num_other]], columns=['enc', 'value', 'count'])
            val_counts['enc'] += 1
            val_counts = other_row.append(val_counts).reset_index(drop=True)

        fillna_val = get_fillna_value(col_params)
        if fillna_val is not None:
            num_na = ds[col_params['params'].name].isna().sum().compute()
            na_row = pd.DataFrame([[0, fillna_val, num_na]], columns=['enc', 'value', 'count'])
            val_counts['enc'] += 1
            val_counts = na_row.append(val_counts).reset_index(drop=True)

        return val_counts
    return None


def infer_processing(ds, ds_descr, cat_features=None, nullable=None):
    if cat_features is None:
        cat_features = set()

    rv = dict()
    rv_cols = []
    for col_name, col_params in ds_descr.iterrows():
        col_params = {'params': col_params, 'cat_features': cat_features, 'ds': ds, 'nullable': nullable}
        rv[col_name] = {
            'dtype': col_params['params']['dtype'],
            'var_type': get_var_type(col_params),
            'processing_type': get_processing_type(col_params),
            'nullable': is_nullable(col_params),
            'n_unique': get_n_unique(col_params),
            'n_na': get_n_na(col_params),
            'mean': get_mean(col_params),
            'std': get_std(col_params),
            'n_unique_cutoff': get_n_unique_cutoff(col_params),
            'coverage': get_coverage(col_params),
            'fillna': get_fillna_value(col_params),
            'repl': get_class_repl_value(col_params),
            'max_count': get_max_count(col_params),
            'min_count': get_min_count(col_params),
            'max_len': get_max_len(col_params),
        }
        val_counts = get_encoding_ids(col_params)
        if val_counts is not None:
            val_counts['column'] = col_name
            rv_cols.append(val_counts)

    rv = pd.DataFrame(rv).T
    rv.index.name = 'column'

    rv_cols = pd.concat(rv_cols).reset_index(drop=True).set_index('column')
    return rv_cols, rv
