Dask-pipes
====================================
*architectural sugar for dask out of memory processing workflow*

Usage
------------------------------------

### Processing pipeline example
```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
import dask_pipes as dp

def get_sample_df():
    """ Creates sample dataset """
    part_1 = pd.DataFrame([['a',  2.0,    2],
                           ['a',  4.0,    100]],columns=['c1','c2','c3'])
    part_2 = pd.DataFrame([[None, 2.0,    np.nan],
                           ['c',  np.nan, 20]],columns=['c1','c2','c3'])
    return dd.concat([part_1, part_2])

# Define locations for storing
# Intermediate processing data and datasets during processing
meta_folder = './meta'  # Folder to store metadata
persist_folder = './persist'  # Folder to store persist parquet files
dataset_name = 'sample_dataset'  # For file naming purposes

# Standard scaler + Fillna pipeline
process_pipeline = dp.Sequence([
        ('transform_columns', dp.ColumnMap([dp.StandardScaler(), 
                                            dp.FillNa()]))
    ], 
    categorical_columns=['c2'],
)

ds = get_sample_df()
print(ds.compute())
#      c1   c2     c3
# 0     a  2.0    2.0
# 1     a  4.0  100.0
# 0  None  2.0    NaN
# 1     c  NaN   20.0

# Fit pipeline
process_pipeline.fit(
    meta_folder    = meta_folder,
    persist_folder = persist_folder,
    dataset        = ds,
    dataset_name   = dataset_name)

# Transform dataset
# Notice: we do not need process_pipeline on transform stage
# We now use transform method anywhere
out_ds = dp.transform(
    meta_folder    = meta_folder,
    persist_folder = persist_folder,
    dataset        = ds,
    dataset_name   = dataset_name)

print(out_ds.compute())
#               c1  c2        c3
# index                         
# 0              a   2 -0.741218
# 1              a   4  1.137386
# 0      <Unknown>   2  0.198084
# 1              c  -1 -0.396168
```