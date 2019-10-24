import dask.dataframe as dd
import numpy as np
import pandas as pd
import logging
from .utils import *

__all__ = ['get_dataset_meta']

logger = logging.getLogger(__name__)


def get_column_data(ds: dd.DataFrame, val_count_th=10, n_reps_th=30, fillna_label='Unknown',
                    cat_cols=None, index_col=None, target_col=None, partition_col=None):
    assert isinstance(ds, dd.DataFrame)

    logger.info("Processing dataframe")

    # Get column types
    ds_meta = pd.DataFrame(ds.dtypes, columns=['type'])
    ds_meta.loc[ds.index.name, 'type'] = ds.index.dtype

    ds_len = ds.shape[0].compute()
    logger.info("Dataset length: {}".format(ds_len))

    # Get value counts for each column
    val_counts = ds_meta.apply(lambda x: None if is_numeric(x.name, x['type'], cat_cols) else x, axis=1).dropna().index

    if index_col is not None:
        # Drop index from counting num unique
        val_counts = [col for col in val_counts if col != index_col and col != ds.index.name]
        non_index_cols = [col for col in ds.columns if col != index_col]
    else:
        non_index_cols = ds.columns

    logger.info("Total partitions: {}".format(ds.npartitions))

    if index_col is not None:
        val_counts.append(index_col)
    if partition_col is not None:
        val_counts.append(partition_col)

    logger.info("Computing full stats for {} columns: {}".format(len(val_counts), val_counts))
    val_counts_dict = dict()
    val_median_dict = dict()
    val_num_na_dict = dict()
    val_max_len_dict = dict()
    for col_i, col in enumerate(val_counts):
        logger.info("Counting unique, max len for column {} ({} of {})".format(col, col_i + 1, len(val_counts)))
        if ds.index.name == col:
            n_unique = ds.index.drop_duplicates().count().compute()
            val_counts_dict[col] = {
                'n_unique': n_unique,
                'avg_reps': round(ds_len / n_unique, 2)
            }
            try:
                val_max_len_dict[col] = pd.Series(ds.index).apply(lambda x: len(x),
                                                                  meta=(col, 'int64')).max().compute()
            except TypeError:
                pass
        else:
            n_unique = ds[col].drop_duplicates().count().compute()
            val_counts_dict[col] = {
                'n_unique': n_unique,
                'avg_reps': round(ds_len / n_unique, 2)
            }
            try:
                val_max_len_dict[col] = ds[col].apply(lambda x: len(x) if x is not pd.isna(x) else 0,
                                                      meta=(col, 'int64')).max().compute()
            except TypeError:
                pass

    it_items = [(col, dtype) for col, dtype in ds.dtypes.iteritems()
                if is_numeric(col, dtype, cat_cols) and col in non_index_cols]
    for col_i, (col, dtype) in enumerate(it_items):
        logger.info("Computing median for column {} ({} of {})".format(col, col_i + 1, len(it_items)))
        val_median_dict[col] = ds[col].drop_duplicates().dropna().quantile(0.5).compute()

    for col in list(ds.columns) + [ds.index.name]:
        if col == ds.index.name:
            val_num_na_dict[col] = ds.index.isna().sum().compute()
        else:
            val_num_na_dict[col] = ds[col].isna().sum().compute()

    val_counts = pd.DataFrame(val_counts_dict).T
    val_median = pd.DataFrame([val_median_dict], index=['median']).T
    val_num_na = pd.DataFrame([val_num_na_dict], index=['num_na']).T
    val_max_len = pd.DataFrame([val_max_len_dict], index=['max_len']).T

    logger.info("Computing statistics for normalization")

    # Compute statistics for normalization
    ds_meta = ds_meta \
        .join(val_num_na) \
        .join(pd.DataFrame(ds[non_index_cols].mean().compute(), columns=['mean'])) \
        .join(val_median) \
        .join(pd.DataFrame(ds[non_index_cols].std().compute(), columns=['std'])) \
        .join(val_counts) \
        .join(val_max_len)
    logger.info("Getting processing classes")
    ds_meta['processing_class'] = ds_meta.apply(lambda x: get_processing_class(col_name=x.name,
                                                                               dtype=x['type'],
                                                                               n_unique=x['n_unique'],
                                                                               val_count_th=val_count_th,
                                                                               n_reps_th=n_reps_th,
                                                                               n_reps=x['n_reps'],
                                                                               cat_cols=cat_cols,
                                                                               index_col=index_col,
                                                                               target_col=target_col,
                                                                               partition_col=partition_col),
                                                axis=1)
    logger.info("Getting fillna values")
    ds_meta['fillna_num'] = ds_meta.apply(
        lambda x: x['median'] if x['processing_class'] == ProcessingClass.NUMERIC.name else None, axis=1)
    ds_meta['fillna_cat'] = ds_meta.apply(
        lambda x: fillna_label if x['processing_class'] == ProcessingClass.ONE_HOT.name else None, axis=1)
    ds_meta.index.name = 'column'

    return ds_meta[['type', 'processing_class', 'num_na', 'mean',
                    'std', 'n_unique', 'avg_reps', 'max_len', 'fillna_num', 'fillna_cat']]


def get_val_meta(ds: dd.DataFrame, column_meta: pd.DataFrame):
    val_meta = []
    cat_columns = list(column_meta[column_meta['processing_class'].isin(categorical_classes)].index)
    cat_columns = cat_columns
    for col in cat_columns:
        proc_class = column_meta.loc[col, 'processing_class']

        if proc_class in index_classes:
            # Skip encoding for index columns
            continue

        fillna_val = column_meta.loc[col, 'fillna_cat']
        n_na = column_meta.loc[col, 'num_na']
        if col == ds.index.name:
            val_counts = ds.index.value_counts().compute()
        else:
            val_counts = ds[col].value_counts().compute()
        val_counts = pd.DataFrame(val_counts).sort_index()

        # Can be none?
        if proc_class in no_na_classes:
            if n_na > 0:
                raise ValueError("Column {} contains NaN values, though it can't!".format(col))
            ranks = np.arange(0, len(val_counts))
        else:
            none = pd.DataFrame([{col: n_na}], index=[fillna_val])
            val_counts = none.append(val_counts)
            ranks = np.arange(0, len(val_counts))
        val_counts.index.name = 'value'
        val_counts.columns = ['count']
        val_counts = val_counts.reset_index()
        val_counts['column'] = col
        val_counts['rank'] = ranks

        val_meta.append(val_counts)
    val_meta = pd.concat(val_meta).sort_values(['column', 'rank', 'value'])
    val_meta = val_meta[['column', 'value', 'rank', 'count']].infer_objects()
    val_meta = val_meta.set_index('column')
    val_meta.index.name = 'column'
    return val_meta


def get_dataset_meta(ds: dd.DataFrame, val_count_th=10,
                     cat_cols=None, index_col=None, target_col=None, partition_col=None):
    column_meta = get_column_data(ds, val_count_th,
                                  cat_cols=cat_cols,
                                  index_col=index_col,
                                  target_col=target_col,
                                  partition_col=partition_col)
    val_meta = get_val_meta(ds, column_meta)
    return column_meta, val_meta
