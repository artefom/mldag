from ..base import NodeBase
import dask.dataframe as dd
import pandas as pd
import pandas.api.types
import numpy as np

__all__ = ['ColumnTransformer']


class ColumnTransformer(NodeBase):

    def __init__(self, name=None):
        super().__init__(name=name)

    def fit(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        pass
