import dask_ml.compose
import dask_ml.impute
import dask_ml.preprocessing

from dask_pipes.nodes import AddNaIndicator, RobustCategoriser, DateProcessor, AddNaCategory
from . import column_selection as cs
from .nodes import as_node
from .pipeline import Pipeline

__all__ = ['prepareNN']


def sort_columns(X):
    return X[sorted(X.columns)]


def prepareNN(date_retro_date_mapping=None):
    pipeline = Pipeline()

    categorize = as_node(RobustCategoriser())
    convert_dates = as_node(DateProcessor(retro_date_mapping=date_retro_date_mapping))

    add_nullable_indicator = as_node(dask_ml.compose.ColumnTransformer([
        ('add_na_indicator', as_node(AddNaIndicator()), cs.Numeric & cs.Nullable)
    ], remainder='passthrough'), name='add_na_indicator')

    add_na_cat = as_node(AddNaCategory())

    # FILLNA TRANSFORMER
    fill_na = as_node(dask_ml.compose.ColumnTransformer([
        ('fillna_numeric', dask_ml.impute.SimpleImputer(strategy='mean'),
         cs.Numeric),
        ('fillna_str', dask_ml.impute.SimpleImputer(strategy='constant', fill_value='<Unknown>'),
         cs.Categorical & cs.Nullable),
    ], remainder='passthrough'), name="fill_na")

    # STANDARD SCALER
    scale_numeric = as_node(dask_ml.compose.ColumnTransformer([
        ('scale_numeric', dask_ml.preprocessing.StandardScaler(),
         cs.Numeric),
    ], remainder='passthrough'), name="scale_numeric")

    one_hot = as_node(dask_ml.preprocessing.DummyEncoder(drop_first=True))

    scale_binary = as_node(dask_ml.compose.ColumnTransformer([
        ('scale_binary', dask_ml.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
         cs.Numeric & cs.Binary),
    ], remainder='passthrough'), name="scale_binary")

    (
            pipeline['X'] >>
            categorize['X'] >>
            convert_dates >>
            # sort_columns >>
            add_nullable_indicator >>
            add_na_cat >>
            fill_na >>
            scale_numeric >>
            one_hot >>
            scale_binary >>
            # sort_columns >>
            pipeline['normalized']
    )

    return pipeline
