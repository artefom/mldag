from typing import Dict, Any

from ..base import OperatorBase


class PartitionMapper(OperatorBase):
    """
    Map dask datafame with custom function by partition
    """

    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @classmethod
    def transform(cls, *args, **kwargs):
        pass
