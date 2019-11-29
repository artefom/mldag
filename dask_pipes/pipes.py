import dask_pipes as dp
import dask_ml.preprocessing
import dask_ml.compose
import dask_ml.impute

__all__ = ['prepareNN']


def prepareNN():
    pipeline = dp.Pipeline()

    categorize = dp.RobustCategoriser()
    convert_dates = dp.DateProcessor(retro_date_mapping={'c5': 'c7'})

    add_nullable_indicator = dp.as_node("add_na_indicator", dask_ml.compose.ColumnTransformer([
        ('add_na_indicator', dp.AddNaIndicator(), dp.numeric_nullable)
    ], remainder='passthrough'))

    add_na_cat = dp.AddNaCategory()

    # FILLNA TRANSFORMER
    fillna = dp.as_node('fillna', dask_ml.compose.ColumnTransformer([
        ('fillna_numeric', dask_ml.impute.SimpleImputer(strategy='mean'), dp.numeric),
        ('fillna_str', dask_ml.impute.SimpleImputer(strategy='constant', fill_value='<Unknown>'), dp.categorical)
    ], remainder='passthrough'))

    # STANDARD SCALER
    scale = dp.as_node('scale', dask_ml.compose.ColumnTransformer([
        ('scale_numeric', dask_ml.preprocessing.StandardScaler(), dp.numeric)
    ], remainder='passthrough'))

    one_hot = dp.as_node('one_hot', dask_ml.preprocessing.DummyEncoder(drop_first=True))

    scale_one_hot = dp.as_node('scale_one_hot', dask_ml.compose.ColumnTransformer([
        ('scale_one_hot', dask_ml.preprocessing.MinMaxScaler(feature_range=(-1, 1)), dp.numeric_binary)
    ], remainder='passthrough'))

    pipeline >> categorize >> convert_dates >> add_nullable_indicator >> \
    add_na_cat >> fillna >> scale >> one_hot >> scale_one_hot

    return pipeline
