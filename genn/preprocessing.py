import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn import preprocessing
from .utils import *
import hashlib
import os
import shutil
import logging

logger = logging.getLogger(__name__)

__all__ = ['get_label_encoder', 'preprocess_chunk', 'preprocess_file', 'save_dataset_meta', 'load_ds']

COLUMNS_META_NAME = 'column_meta.csv'
VAL_META_NAME = 'val_meta.csv'


def get_label_encoder(val_meta, col):
    values = val_meta.loc[col, ['value', 'rank']].sort_values('rank')
    ranks = values['rank'].values
    values = values['value'].astype(str).values
    for i, rank in enumerate(ranks):
        if not i == rank:
            raise ValueError("Invalid ranks for column {}".format(col))
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


def preprocess_chunk(column_meta: pd.DataFrame, val_meta: pd.DataFrame, part: pd.DataFrame, hash_index=False):
    train_columns = get_train_columns(column_meta)
    cat_columns = get_categorical_cols(column_meta)
    num_columns = get_ordered_cols(column_meta)
    label_encoders = {col: get_label_encoder(val_meta, col) for col in cat_columns if col in train_columns}
    fillna_values = {col: get_fillna_value(column_meta, col) for col in train_columns
                     if not pd.isna(get_fillna_value(column_meta, col))}

    # FIllna
    logger.info("Filling nan values")
    part = part.fillna(fillna_values)

    # Encode labels
    logger.info("Encoding labels")
    for col, enc in label_encoders.items():
        part[col] = enc.transform(part[col].astype(str))

    # Normalize
    logger.info("Normalizing")
    for col in num_columns:
        print(col)
        print(get_mean_value(column_meta, col))
        print(get_std_value(column_meta, col))
        part[col] = (part[col] - get_mean_value(column_meta, col)) / get_std_value(column_meta, col)

    if hash_index:
        part.index = pd.Series(part.index).apply(
            lambda x: hashlib.sha3_256(str(x).encode('utf-8')).hexdigest()[:5] + '_{:0>8}'.format(x))
    logger.info("Success")
    return part


def save_dataset_meta(input_file,
                      out_dir,
                      index_col=None,
                      target_col=None,
                      cat_cols=None,
                      drop_dir=False):
    with ProgressBar():
        partition_col, ds_source = read_file(input_file)

        logging.info("Dtypes:\n{}".format(ds_source.dtypes))
        out_dir = os.path.abspath(out_dir)
        if os.path.exists(out_dir):
            if drop_dir:
                shutil.rmtree(out_dir)
            else:
                raise IOError("Directory {} already exists!".format(out_dir))
        logger.info("Getting column and value metadata")
        column_meta, val_meta = get_dataset_meta(ds_source,
                                                 cat_cols=cat_cols,
                                                 index_col=index_col,
                                                 target_col=target_col,
                                                 partition_col=partition_col)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        column_meta_out_file = os.path.join(out_dir, COLUMNS_META_NAME)
        val_meta_out_file = os.path.join(out_dir, VAL_META_NAME)

        logger.info("Saving data to disk")
        column_meta.to_csv(column_meta_out_file)
        val_meta.to_csv(val_meta_out_file)


def preprocess_file(fname,
                    meta_dir,
                    rewrite=False):
    with ProgressBar():
        column_meta_out_file = os.path.join(meta_dir, COLUMNS_META_NAME)
        val_meta_out_file = os.path.join(meta_dir, VAL_META_NAME)

        column_meta = pd.read_csv(column_meta_out_file).set_index('column')
        val_meta = pd.read_csv(val_meta_out_file).set_index('column')

        partition_col = list(column_meta[column_meta['processing_class'].isin(partition_classes)].index)
        if len(partition_col) != 1:
            raise NotImplementedError()
        partition_col = partition_col[0]

        source_partition, ds_source = read_file(fname)
        if source_partition is None:
            ds_source.set_index(partition_col)
        else:
            if ds_source.index.name != partition_col:
                raise ValueError("ds_source.index does not match partition col!")
        source_n_partitions = ds_source.npartitions

        out_fname = os.path.splitext(os.path.split(os.path.abspath(fname))[1])[0]
        processed_ds_out = os.path.join(meta_dir, '{}.hdf'.format(out_fname))
        processed_ds_out_parquet = os.path.join(meta_dir, '{}.parquet'.format(out_fname))

        if os.path.exists(processed_ds_out):
            os.remove(processed_ds_out)

        if rewrite:
            if os.path.exists(processed_ds_out_parquet):
                os.remove(processed_ds_out_parquet)
        elif os.path.exists(processed_ds_out_parquet):
            raise ValueError("File {} already exist!".format(processed_ds_out_parquet))

        min_item_sizes = get_min_itemsize(column_meta)
        for part_i, part in enumerate(ds_source.partitions):
            logger.info('Processing chunk {} of {}'.format(part_i + 1, source_n_partitions))
            part_processed = preprocess_chunk(column_meta, val_meta, part.compute())
            part_processed_dd: dd.DataFrame = dd.from_pandas(part_processed, npartitions=source_n_partitions)
            logger.info("Saving to disk")
            part_processed_dd.to_hdf(processed_ds_out, 'df', mode='a',
                                     append=True, min_itemsize=min_item_sizes)

        # Convert to parquet
        logger.info("Converting to parquet")
        dd.read_hdf(processed_ds_out, 'df').to_parquet(processed_ds_out_parquet)
        os.remove(processed_ds_out)
