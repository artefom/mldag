from ..base import NodeBase
from ..exceptions import DaskPipesException
from ..utils import replace_signature

from types import MethodType
import inspect


class NodeWrapper(NodeBase):

    def __init__(self, name=None, estimator=None):
        super().__init__(name)
        self._estimator = None
        self.estimator = estimator

    @property
    def estimator(self):
        if not self._estimator:
            raise DaskPipesException("{} does not have assigned estimator".format(self))
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if estimator is not None:
            if not hasattr(estimator, 'fit'):
                raise DaskPipesException("{} must implement fit".format(estimator))
            if not hasattr(estimator, 'transform'):
                raise DaskPipesException("{} must implement transform".format(estimator))
            fit_sign = inspect.signature(estimator.fit.__func__)
            fit_sign = inspect.Signature(
                parameters=list(fit_sign.parameters.values()),
                return_annotation=NodeWrapper
            )
            self._set_fit_signature(fit_sign)
            self._set_transform_signature(inspect.signature(estimator.transform.__func__))
        else:
            self._reset_transform_signature()
            self._reset_fit_signature()
        self._estimator = estimator

    def _set_fit_signature(self, sign: inspect.Signature):
        self.fit = MethodType(replace_signature(self.__class__.fit, sign), self)

    def _set_transform_signature(self, sign: inspect.Signature):
        self.transform = MethodType(replace_signature(self.__class__.transform, sign), self)

    def _reset_fit_signature(self):
        self.fit = MethodType(self.__class__.fit, self)

    def _reset_transform_signature(self):
        self.transform = MethodType(self.__class__.transform, self)

    def __repr__(self):
        if self._estimator is None:
            return '<{}: {}>'.format(self.__class__.__name__, self.name)
        return '<{}({}): {}>'.format(self.__class__.__name__, self.estimator, self.name)

    def fit(self, *args, **kwargs):
        self.estimator.fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        return self.estimator.transform(*args, **kwargs)


def as_node(name, estimator):
    return NodeWrapper(name=name, estimator=estimator)
