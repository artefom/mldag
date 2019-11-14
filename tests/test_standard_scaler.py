import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytest
import yaml
from io import StringIO

from genn.column_processors import StandardScaler, FillNa
from genn.processors import ColumnProcessor
from genn.pipeline import DaskPipeline

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


def test_standard_scaler():
    cp = StandardScaler()

    stats = cp.get_stats(test_ds['normal'])
    mean = stats['mean']
    std = stats['std']

    assert pytest.approx(mean, 1E-6) == 3.4531175658946096
    assert pytest.approx(std, 1E-6) == 1.830683795745193


def test_pipeline():
    process_pipeline = DaskPipeline([
        ('transform_columns', ColumnProcessor([StandardScaler(),
                                               FillNa()]))
    ],
        meta_folder='meta',
        persist_folder='persist')

    ds_name = 'test_ds'

    process_pipeline.fit(test_ds, ds_name)
    rv = test_ds.compute()

    rv = process_pipeline.transform(
        process_pipeline.get_meta_folder(),
        process_pipeline.get_persist_folder(),
        ds_name,
        test_ds).compute()

    assert pytest.approx(rv['normal'].mean(), 1E-6) == 0
    assert pytest.approx(rv['normal'].std(), 1E-6) == 1


def test_processor():
    # Define dask processor
    processor = ColumnProcessor(meta_folder='meta',
                                column_mixins=[StandardScaler(),
                                               FillNa()])
    ds_name = 'test_ds'
    # Fit dask processor
    processor.fit(test_ds, ds_name)

    transf = processor.transform(processor.get_meta_folder(),
                                 processor.get_persist_folder(),
                                 ds_name,
                                 test_ds)
    rv = transf.compute()
    print(rv['normal'].std())
    print(rv['normal'].mean())

    print(rv['normal2'].std())
    print(rv['normal2'].mean())
    print(rv)
