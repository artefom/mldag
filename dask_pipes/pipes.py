import dask_ml.compose
import dask_ml.impute
import dask_ml.preprocessing

from dask_pipes.nodes import AddNaIndicator, RobustCategoriser, DateProcessor, AddNaCategory
from . import column_selection as cs
from .nodes import as_node
from .pipeline import Pipeline

__all__ = ['prepareNN']


def prepareNN():
    pipeline = Pipeline()

    categorize = RobustCategoriser()
    convert_dates = DateProcessor(retro_date_mapping={'c5': 'c7'})

    add_nullable_indicator = as_node(dask_ml.compose.ColumnTransformer([
        ('add_na_indicator', AddNaIndicator(), cs.Numeric & cs.Nullable)
    ], remainder='passthrough'))

    add_na_cat = AddNaCategory()

    # FILLNA TRANSFORMER
    fillna = as_node(dask_ml.compose.ColumnTransformer([
        ('fillna_numeric', dask_ml.impute.SimpleImputer(strategy='mean'),
         cs.Numeric),
        ('fillna_str', dask_ml.impute.SimpleImputer(strategy='constant', fill_value='<Unknown>'),
         cs.Categorical & cs.Nullable),
    ], remainder='passthrough'), name="FillNa")

    # STANDARD SCALER
    scale = as_node(dask_ml.compose.ColumnTransformer([
        ('scale_numeric', dask_ml.preprocessing.StandardScaler(),
         cs.Numeric),
    ], remainder='passthrough'), name="ScaleNumeric")

    one_hot = as_node(dask_ml.preprocessing.DummyEncoder(drop_first=True))

    scale_one_hot = as_node(dask_ml.compose.ColumnTransformer([
        ('scale_one_hot', dask_ml.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
         cs.Numeric & cs.Binary),
    ], remainder='passthrough'), name="ScaleNumericBinary")

    (
            pipeline >>
            categorize >>
            convert_dates >>
            add_nullable_indicator >>
            add_na_cat >>
            fillna >>
            scale >>
            one_hot >>
            scale_one_hot >>
            pipeline['normalized']
    )

    return pipeline
