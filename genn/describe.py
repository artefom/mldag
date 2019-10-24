import dask.dataframe as dd


def describe_col(col: dd.Series):
    return {
        'n_unique': col.drop_duplicates().count().compute(),
        'n_na': col.isna().sum().compute(),
        'mean': col.mean().compute()
    }


def describe(ds: dd.DataFrame):
    pass
