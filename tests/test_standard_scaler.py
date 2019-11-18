import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytest
import os

import dask_pipes as dp
import dask_pipes.columnmappers
import dask_pipes.storage

ds1 = pd.DataFrame([['cat5', -0.08791349765766582, 1],
                    ['cat2', -0.45607955436914216, np.nan],
                    ['cat4', 1.0365671323315593, 0],
                    ['cat1', -0.024157518723391634, 1],
                    ['cat4', -1.0746881596620674, 1],
                    ['cat2', -1.3745769333109847, 1],
                    ['cat2', -0.8096348940348146, 1],
                    ['cat2', 0.9389351138213718, 1],
                    ['cat1', 0.0816240934021167, 0],
                    ['cat2', 0.23782656204987004, 1]],
                   columns=['cat', 'normal', 'normal2'])
ds2 = pd.DataFrame([['cat4', 0.0925574898889439, -1],
                    ['cat3', 0.5267352833224139, np.nan],
                    ['cat3', -0.6058660301330128, 1],
                    ['cat1', 0.8961509434493576, 1],
                    ['cat3', -0.0027012581656900036, 1],
                    ['cat3', 0.021680712905233424, np.nan],
                    ['cat3', -1.348967911605108, 1],
                    ['cat2', 1.6863322137777539, np.nan],
                    ['cat5', -0.5088200779053001, 1],
                    ['cat1', -0.16265239148925334, np.nan]],
                   columns=['cat', 'normal', 'normal2'])
ds1['normal'] += 2
ds2['normal'] += 5
test_ds = dd.concat([ds1, ds2])

tmp_folder = 'tmp'
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

params_folder = os.path.join(tmp_folder, 'meta')
persist_folder = os.path.join(tmp_folder, 'persist')

if not os.path.exists(params_folder):
    os.mkdir(params_folder)
if not os.path.exists(persist_folder):
    os.mkdir(persist_folder)

ds_name = 'test_ds'


def test_standard_scaler():
    cp = dp.columnmappers.StandardScaler()

    stats = cp.get_stats(test_ds['normal'])
    mean = stats['mean']
    std = stats['std']

    assert pytest.approx(mean, 1E-6) == 3.4531175658946096
    assert pytest.approx(std, 1E-6) == 1.830683795745193


def test_pipeline():
    # Define dask processor
    pipeline = dp.FolderPipeline(params_folder, persist_folder)

    scale_fillna = dp.ColumnMap('transform_columns', [dp.columnmappers.StandardScaler(), dp.columnmappers.FillNa()])

    pipeline >> scale_fillna

    # Fit dask processor
    pipeline.fit(test_ds)

    rv = pipeline.transform(test_ds).compute()

    assert pytest.approx(rv['normal'].mean(), 1E-6) == 0
    assert pytest.approx(rv['normal'].std(), 1E-6) == 1


def test_processor():
    # Define dask processor
    processor = dp.ColumnMap('column_map',
                             column_mixins=[dp.columnmappers.StandardScaler(),
                                            dp.columnmappers.FillNa()])

    params_storage = dp.MemoryStorage()
    persist_storage = dp.MemoryStorage()

    processor.fit(params_storage, persist_storage, test_ds)

    rv = dp.ColumnMap.transform(params_storage, persist_storage, test_ds).compute()

    print(rv['normal'].std())
    print(rv['normal'].mean())

    print(rv['normal2'].std())
    print(rv['normal2'].mean())
    print(rv)
