import io
from datetime import datetime, timedelta
from typing import Optional

import dask.dataframe as dd
import pandas as pd

import dask_pipes as dp

TEST_DS: Optional[pd.DataFrame] = None


def get_ds() -> pd.DataFrame:
    global TEST_DS
    if TEST_DS is None:
        ds = """
        index c0 c1  c2   c3  c4   c5
        2     0  a   1    3.4 s1   2019-01-01T02:11:59
        2     1  a   100  5   s2
        1     0  b   9    nan s4   2019-02-01T20:33:32
        3     0  nan 100  -1  s4   
        3     1  c   100  3.1 s4   2019-01-23T10:57:1
        """
        ds = dd.from_pandas(pd.read_csv(io.StringIO(ds), sep=r'\s+'), npartitions=3)
        # ds['c1'] = ds['c1'].apply(lambda x: x if x != 'b' else -1, meta=('c1',object)).compute()
        ds['c5'] = dd.to_datetime(ds['c5'])
        ds['c6'] = pd.Series([timedelta(1.12), None, timedelta(0), timedelta(-10)])
        ds['c7'] = datetime(2019, 7, 1)
        ds = ds.set_index('index').persist()
        # ds = ds.compute()
        TEST_DS = ds
    return TEST_DS.copy()


def test_pipeline():
    ds = get_ds()

    p = dp.pipes.prepareNN(date_retro_date_mapping={'c5': 'c7'})
    p.mixins = [dp.extensions.CacheMixin('tmp')]
    f_run = p.fit(ds)
    t_run = p.transform(ds)

    assert p.input_names == ['X', 'run_id']
    assert p.output_names == ['normalized']


def test_args():
    @dp.returns(['some_result'])
    def foo(a, a_default=None, *var_pos, b, b_default=None, **var_key):
        return (a, a_default, var_pos, b, b_default, var_key),

    p = dp.Pipeline()
    a = dp.as_node(foo, name='a')

    p['a'] >> a['a']
    p['a_default'] >> a['a_default']
    p['args'] >> a['var_pos']
    p['b'] >> a['b']
    p['kwargs'] >> a['var_key']
    a['some_result'] >> p['result']

    run = p.transform(1, 2, 3, 4, b=10, kwarg=10)
    assert run.outputs['result'] == (1, 2, (3, 4), 10, None, {'kwarg': 10})


if __name__ == '__main__':
    test_pipeline()
    test_args()
