from typing import Dict, Any

from dask import dataframe as dd

__all__ = ['SingleColumnMapper']


class SingleColumnMapper:
    """
    Base class for arbitrary column processing, used by base.`dask_pipes.processors.ColumnMap`
    """

    def get_stats(self, column: dd.Series, force_categorical=False) -> Dict[str, Any]:
        """
        Compute necessary stats for transform step
        :param column: column to apply transformation to
        :param force_categorical: does current column should be considered categorical? (default: False)
        :return: dictionary of arbitrary statistics to be written to file.
        may contain numbers, strings or even pd.DataFrame. This is later passed to transform method
        Example:
        >>> def get_stats(...):
        >>>     ...
        >>>     return {'mean': 1, 'df': pd.DataFrame([[1,2],[3,4]],columns=['a','b'])}
        >>> ...
        >>> def transform(column, params):
        >>>     print(params['mean'])  # prints '1'
        >>>     print(params['df']['a'].iloc[1])  # prints '3'
        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> dd.Series:
        """
        Transform specific column based on params - previously saved dictionary of statistics
        This method must return dask future element
        # Standard scale example
        >>> def transform(...):
        >>>     return (column-params['mean'])/params['std']

        Notice:
            transform is a class method, so it be applied without having to create class instance

        :param column: column to apply transformation to
        :param params: Dictionary previously returned by get_stats
        :return: Dask Future of processed column
        """
        raise NotImplementedError()
