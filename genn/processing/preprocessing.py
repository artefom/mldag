import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn import preprocessing
from genn.processing.describe import *
import hashlib
import os
import shutil
import logging

logger = logging.getLogger(__name__)

__all__ = ['get_label_encoder', 'preprocess_chunk', 'preprocess_ds', 'load_ds']

COLUMNS_META_NAME = 'column_meta.csv'
VAL_META_NAME = 'val_meta.csv'


def get_label_encoder(val_meta, col):
    values = val_meta.loc[col, ['value', 'enc']].sort_values('enc')
    ranks = values['enc'].values
    values = values['value'].astype(str).values
    for i, rank in enumerate(ranks):
        if not i == rank:
            raise ValueError("Invalid encodings for column {}".format(col))
    rv = preprocessing.LabelEncoder()
    rv.classes_ = values
    return rv


def load_ds(*args,
            columns=None,
            partition_col=None,
            process_func=None,
            **kwargs):
    """
    Load ds and process if necessary by partition
    :param args: arguments passed to dask.dataframe.read_csv
    :param columns: columns to read from file
    :param partition_col: column to partition by
    :param process_func: processing function
    :param kwargs: keyword arguments passed to dask.dataframe.read_csv
    :return:
    """
    if partition_col is not None and partition_col in columns:
        raise ValueError("Partition column {} is in columns".format(partition_col))
    fname = args[0]

    ds: dd.DataFrame = dd.read_csv(*args, **kwargs)
    if columns is not None:
        all_columns = columns
        if partition_col is not None:
            all_columns = all_columns + [partition_col]
        ds = ds[all_columns]
    if partition_col is not None:
        ds = ds.set_index(partition_col)

    if process_func is not None:
        logger.info("Applying processing func")
        ds = ds.map_partitions(process_func)
        intermediate_file = '{}_process_cache.parquet'.format(os.path.splitext(fname)[0])
        if os.path.exists(intermediate_file):
            shutil.rmtree(intermediate_file)
        ds.to_parquet(intermediate_file)
        ds = dd.read_parquet(intermediate_file)

    return ds


def get_categorical_cols(column_meta):
    return list(column_meta[column_meta['processing_type'].isin(categorical_processing_types)].index)


def get_train_columns(column_meta):
    return list(column_meta[column_meta['processing_type'].isin(train_processing_types)].index)


def get_numeric_cols(column_meta):
    return list(column_meta[column_meta['processing_type'].isin(numeric_processing_types)].index)


def get_fillna_value(column_meta, col):
    return column_meta.loc[col]['fillna']


def get_mean_value(column_meta, col):
    return column_meta.loc[col]['mean']


def get_std_value(column_meta, col):
    return column_meta.loc[col]['std']


def get_min_itemsize(column_meta):
    return {col: max_len for col, max_len in column_meta['max_len'].dropna().iteritems()}


def get_replace_val(column_meta, col):
    return column_meta.loc[col]['repl']


def preprocess_chunk(column_meta: pd.DataFrame, val_meta: pd.DataFrame, part: pd.DataFrame, hash_index=False):
    train_columns = get_train_columns(column_meta)
    cat_columns = get_categorical_cols(column_meta)
    num_columns = get_numeric_cols(column_meta)

    label_encoders = {col: get_label_encoder(val_meta, col) for col in cat_columns if col in train_columns}
    fillna_values = {col: get_fillna_value(column_meta, col) for col in train_columns
                     if not pd.isna(get_fillna_value(column_meta, col))}

    # FIllna
    logger.info("Filling nan values")
    part = part.fillna(fillna_values)

    # Encode labels
    logger.info("Encoding labels")
    for col, enc in label_encoders.items():
        enc: preprocessing.LabelEncoder
        repl_val = get_replace_val(column_meta, col)
        if not pd.isna(repl_val):
            part[col] = part.where(~((~part[col].astype(str).isin(enc.classes_)) & (~part[col].isna())), other=repl_val)
        try:
            part[col] = enc.transform(part[col].astype(str))
        except ValueError as ex:
            msg = str(ex)
            raise ValueError("Col {}: {}".format(col, msg)) from None
    # Normalize
    logger.info("Normalizing")
    for col in num_columns:
        part[col] = (part[col] - get_mean_value(column_meta, col)) / get_std_value(column_meta, col)

    if hash_index:
        part.index = pd.Series(part.index).apply(
            lambda x: hashlib.sha3_256(str(x).encode('utf-8')).hexdigest()[:5] + '_{:0>8}'.format(x))
    logger.info("Success")
    return part


def preprocess_ds(ds,
                  val_meta,
                  column_meta,
                  out_fname,
                  rewrite=False):
    assert column_meta is not None
    assert val_meta is not None
    assert ds is not None
    assert out_fname is not None
    with ProgressBar():

        source_n_partitions = ds.npartitions

        processed_ds_out = os.path.join(os.path.splitext(out_fname)[0], '{}.hdf'.format(out_fname))
        processed_ds_out_parquet = out_fname

        if os.path.exists(processed_ds_out):
            os.remove(processed_ds_out)

        if rewrite:
            if os.path.exists(processed_ds_out_parquet):
                os.remove(processed_ds_out_parquet)
        elif os.path.exists(processed_ds_out_parquet):
            raise ValueError("File {} already exist!".format(processed_ds_out_parquet))

        min_item_sizes = get_min_itemsize(column_meta)
        if ds.index.name in min_item_sizes:
            min_item_sizes['index'] = min_item_sizes[ds.index.name]
            del min_item_sizes[ds.index.name]
        logger.info("Min itmsizes: {}".format(min_item_sizes))
        for part_i, part in enumerate(ds.partitions):
            logger.info('Processing chunk {} of {}'.format(part_i + 1, source_n_partitions))
            part_processed = preprocess_chunk(column_meta, val_meta, part.compute())
            part_processed_dd: dd.DataFrame = dd.from_pandas(part_processed, npartitions=source_n_partitions)
            logger.info("Saving to disk")
            part_processed_dd.to_hdf(processed_ds_out, 'df', mode='a',
                                     append=True, min_itemsize=min_item_sizes)

        # Convert to parquet
        logger.info("Converting to parquet")
        dd.read_hdf(processed_ds_out, 'df').to_parquet(processed_ds_out_parquet)

        # Remove intermediate hdf file
        os.remove(processed_ds_out)
