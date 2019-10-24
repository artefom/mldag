from enum import Enum, auto
import numpy as np
import pandas as pd
import os
import dask.dataframe as dd

__all__ = ['ProcessingClass', 'is_numeric', 'get_processing_class',
           'get_fillna_value', 'get_train_columns', 'get_ordered_cols', 'get_categorical_cols',
           'get_mean_value', 'get_std_value', 'categorical_classes', 'numeric_classes', 'no_na_classes',
           'index_classes', 'target_classes', 'read_file', 'partition_classes', 'skip_classes',
           'get_min_itemsize']


class ProcessingClass(Enum):
    SKIP = auto()
    NUMERIC = auto()
    ONE_HOT = auto()
    BINARY = auto()
    EMBEDDING = auto()
    INDEX = auto()
    PARTITION = auto()
    TARGET_CLASS = auto()
    TARGET_NUMERIC = auto()


skip_classes = {i.name for i in {ProcessingClass.SKIP}}
index_classes = {i.name for i in {ProcessingClass.INDEX}}
partition_classes = {i.name for i in {ProcessingClass.PARTITION}}
target_classes = {i.name for i in {ProcessingClass.TARGET_CLASS,
                                   ProcessingClass.TARGET_NUMERIC}}
categorical_classes = {i.name for i in {ProcessingClass.TARGET_CLASS,
                                        ProcessingClass.ONE_HOT,
                                        ProcessingClass.BINARY,
                                        ProcessingClass.EMBEDDING}}
numeric_classes = {i.name for i in {ProcessingClass.NUMERIC,
                                    ProcessingClass.TARGET_NUMERIC}}
no_na_classes = {i.name for i in {ProcessingClass.TARGET_NUMERIC,
                                  ProcessingClass.TARGET_CLASS,
                                  ProcessingClass.INDEX}}


def read_file(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.csv':
        return None, dd.read_csv(file)
    if ext == '.hdf':
        rv = dd.read_hdf(file, 'df')
        return rv.index.name, rv
    if ext == '.parquet':
        rv = dd.read_parquet(file)
        return rv.index.name, rv
    raise ValueError(f"Extension {ext} not recognized")


def is_numeric(col_name, dtype, cat_cols=None):
    if cat_cols is not None and col_name in cat_cols:
        return False
    return np.issubdtype(dtype, np.number)


def get_processing_class(col_name, dtype, n_unique, n_reps, val_count_th, n_reps_th,
                         nullable=False,
                         cat_cols=None,
                         index_col=None,
                         target_col=None,
                         partition_col=None):
    if len({index_col}.intersection({target_col}).intersection({partition_col})) > 0:
        raise ValueError("index column, target column and partition column must be different")

    if index_col is not None and col_name == index_col:
        return ProcessingClass.INDEX.name
    if partition_col is not None and col_name == partition_col:
        return ProcessingClass.PARTITION.name
    if target_col is not None and col_name == target_col:
        if is_numeric(col_name, dtype, cat_cols):
            return ProcessingClass.TARGET_NUMERIC.name
        if n_unique < val_count_th:
            return ProcessingClass.TARGET_CLASS.name
        raise ValueError("Too many target classes!")
    if is_numeric(col_name, dtype, cat_cols):
        return ProcessingClass.NUMERIC.name
    if n_unique <= 2:
        return ProcessingClass.BINARY.name
    if n_unique < val_count_th:
        return ProcessingClass.ONE_HOT.name
    if n_reps < n_reps_th:
        return ProcessingClass.EMBEDDING.name

    return ProcessingClass.SKIP.name


def get_ordered_cols(column_meta):
    num_columns = set(column_meta[column_meta['processing_class'].isin(numeric_classes)].index)
    return num_columns


def get_min_itemsize(column_meta):
    return {col: max_len for col, max_len in column_meta['max_len'].dropna().iteritems()}


def get_categorical_cols(column_meta):
    cat_columns = set(column_meta[column_meta['processing_class'].isin(categorical_classes)].index)
    return cat_columns


def get_fillna_value(column_meta, col):
    row = column_meta.loc[col]
    rv = None
    if row['processing_class'] in no_na_classes:
        return None
    if row['processing_class'] in numeric_classes:
        rv = row['fillna_num']
    if row['processing_class'] in categorical_classes:
        rv = row['fillna_cat']
    if pd.isna(rv):
        raise ValueError("Can't get fillna value for column {}".format(col))
    return rv


def get_mean_value(column_meta, col):
    return column_meta.loc[col, 'mean']


def get_std_value(column_meta, col):
    return column_meta.loc[col, 'std']


def get_train_columns(column_meta):
    skip_cols = index_classes.union(target_classes).union(partition_classes).union(skip_classes)
    rv = []
    for col_name, row in column_meta.iterrows():
        if row['processing_class'] not in skip_cols:
            rv.append(col_name)
    return rv
