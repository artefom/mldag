from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, _SpecialForm

__all__ = ['OperatorConnection', 'OperatorBase', 'Pipeline']


class OperatorConnection(EdgeBase):
    pass


class OperatorBaseMeta(type):

    @staticmethod
    def get_outputs(func):
        return_type = func.__annotations__['return']
        if isinstance(return_type, tuple) or isinstance(return_type, list):
            pass
        elif isinstance(return_type, dict):
            pass
        elif isinstance(return_type, _SpecialForm):
            pass

    @staticmethod
    def get_inputs(func):
        pass

    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return fit_wrapped

    def __new__(mcs, name, bases, attrs):
        """If the class has a 'run' method, wrap it"""
        if 'fit' in attrs:
            attrs['fit'] = mcs.wrap_fit(attrs['fit'])
        return super().__new__(mcs, name, bases, attrs)


class OperatorBase(VertexBase, metaclass=OperatorBaseMeta):

    def fit(self, *args, **kwargs):
        """
        Infer parameters prior to transforming dataset
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @classmethod
    def transform(cls, params, **kwargs):
        """
        Transform dataset
        :param params:
        :param kwargs:
        :return:
        """
        pass


class Pipeline(Graph):

    def validate_edge(self, edge: OperatorConnection):
        if not isinstance(edge, OperatorConnection):
            raise DaskPipesException(
                "Expected {}; got {}".format(OperatorConnection.__name__, edge.__class__.__name__))

    def validate_vertex(self, vertex):
        if not isinstance(vertex, OperatorBase):
            raise DaskPipesException(
                "Expected {}; got {}".format(OperatorBase.__name__, vertex.__class__.__name__))

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline. Sequentially calls fit_transform of child operators
        :param args:
        :param kwargs:
        :return:
        """
        for v in VertexWidthFirst(self):
            pass

    def transform(self, *args, **kwargs):
        """
        Method for transforming the data.
        Does not alter the fitted parameters
        :param args:
        :param kwargs:
        :return:
        """
        for v in VertexWidthFirst(self):
            pass
