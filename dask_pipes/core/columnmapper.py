from typing import Dict, Any, List

import dask.dataframe as dd

__all__ = ['ColumnMapperBase']


class ColumnMapperBase:
    """
    Base class for arbitrary column processing, used by core.`dask_pipes.processors.ColumnMap`
    """

    def __init__(self, skip_cols=None):
        self.skip_cols = skip_cols

    def get_stats(self, dataset: dd.DataFrame, column: dd.Series) -> Dict[str, Any]:
        """
        Compute necessary stats for transform step

        Parameters
        ----------
        dataset
            Dataset for gathering info from other columns
        column
            column to apply transformation to

        Returns
        -------
        stats
            dictionary of arbitrary statistics to be written to file.
            may contain numbers, strings or even pd.DataFrame. This is later passed to transform method

        Examples
        -------
        >>> def get_stats(...):
        ...     ...
        ...     return {'mean': 1, 'df': pd.DataFrame([[1,2],[3,4]],columns=['a','b'])}
        ... ...
        ... def transform(column, params):
        ...     print(params['mean'])  # prints '1'
        ...     print(params['df']['a'].iloc[1])  # prints '3'

        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, column: dd.Series, params: Dict[str, Any]) -> List[dd.Series]:
        """
        Transform specific column based on params - previously saved dictionary of statistics
        This method must return dask future element

        .. notice:
            transform is a class method, so it be applied without having to create class instance

        Parameters
        ----------
        column
            column to apply transformation to
        params
            Dictionary previously returned by get_stats

        Returns
        -------
            Dask Future of processed column


        Examples
        ----------

        Standard scale example
        >>> def transform(...):
        ...     return (column-params['mean'])/params['std']

        """
        raise NotImplementedError()
