Dask-pipes
====================================
**architectural sugar for dask out of memory processing workflow**

![](https://github.com/artefom/dask-pipes/workflows/unit-test/badge.svg?branch=master)

Usage
------------------------------------

### Processing pipeline example
```python
import numpy as np
import pandas as pd
import dask.dataframe as dd

# Use functions from dask ml
import dask_ml

# Dask-pipes stuff
import dask_pipes as dp
from dask_pipes import column_selection as cs
from dask_pipes.nodes import AddNaIndicator, RobustCategoriser, DateProcessor, AddNaCategory

def get_sample_df():
    """ Creates sample dataset """
    part_1 = pd.DataFrame([['a',  2.0,    2],
                           ['a',  4.0,    100]],columns=['c1','c2','c3'])
    part_2 = pd.DataFrame([[None, 2.0,    np.nan],
                           ['c',  np.nan, 20]],columns=['c1','c2','c3'])
    return dd.concat([part_1, part_2])

pipeline = dp.Pipeline()

categorize = RobustCategoriser()
convert_dates = DateProcessor(retro_date_mapping={'c5': 'c7'})

add_nullable_indicator = dp.as_node(dask_ml.compose.ColumnTransformer([
    ('add_na_indicator', AddNaIndicator(), cs.Numeric & cs.Nullable)
], remainder='passthrough'))

add_na_cat = AddNaCategory()

# FILLNA TRANSFORMER
fillna = dp.as_node(dask_ml.compose.ColumnTransformer([
    ('fillna_numeric', dask_ml.impute.SimpleImputer(strategy='mean'),
     cs.Numeric),
    ('fillna_str', dask_ml.impute.SimpleImputer(strategy='constant', fill_value='<Unknown>'),
     cs.Categorical & cs.Nullable),
], remainder='passthrough'), name="FillNa")
	
# STANDARD SCALER
scale = dp.as_node(dask_ml.compose.ColumnTransformer([
    ('scale_numeric', dask_ml.preprocessing.StandardScaler(),
     cs.Numeric),
], remainder='passthrough'), name="ScaleNumeric")

one_hot = dp.as_node(dask_ml.preprocessing.DummyEncoder(drop_first=True))

scale_one_hot = dp.as_node(dask_ml.compose.ColumnTransformer([
    ('scale_one_hot', dask_ml.preprocessing.MinMaxScaler(feature_range=(-1, 1)),
     cs.Numeric & cs.Binary),
], remainder='passthrough'), name="ScaleNumericBinary")

# Create pipeline
(pipeline >> categorize >> add_nullable_indicator >>
 add_na_cat >> fillna >> scale >> one_hot >> scale_one_hot >> pipeline['normalized'])


ds = get_sample_df()
pipeline.fit(ds)
res = pipeline.transform(ds)
res.outputs['normalized'].compute()
# -1.0  -1.0   1.0  -1.0  -0.816497  -1.048240
# -1.0  -1.0   1.0  -1.0   1.632993   1.608507
# -1.0   1.0  -1.0  -1.0  -0.816497   0.000000
#  1.0  -1.0  -1.0   1.0   0.000000  -0.560266
```

Visualisation
------------------------------------