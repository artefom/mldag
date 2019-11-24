from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, Dict, List, Tuple, Type, Optional, NamedTuple
from collections import namedtuple
import inspect
from ..utils import *
from types import MethodType
import sys

__all__ = ['OperatorConnection', 'OperatorBase', 'Pipeline', 'OperatorBaseMeta', 'ExampleOperator']

SLOT_UNNAMED = '<unnamed>'

OperatorInput = namedtuple("OperatorInput", ['input_arg', 'upstream_output_name', 'upstream_operator'])

PipelineInput = namedtuple("PipelineInput", ['arg_name', 'downstream_slot', 'downstream_operator'])


class OperatorConnection(EdgeBase):

    def __init__(self, upstream, downstream, upstream_slot: str, downstream_slot: str):
        """
        :param upstream:
        :type upstream: OperatorBase
        :param downstream:
        :type downstream: OperatorBase
        :param upstream_slot:
        :param downstream_slot:
        """
        super().__init__(upstream, downstream)
        if not isinstance(upstream_slot, str):
            raise DaskPipesException("upstream slot must be str, got {}".format(upstream_slot.__class__.__name__))
        if not isinstance(upstream_slot, str):
            raise DaskPipesException("downstream slot must be str, got {}".format(downstream_slot.__class__.__name__))

        if downstream_slot not in {i.name for i in downstream.inputs}:
            raise DaskPipesException("Invalid value for downstream_slot: {} does not have '{}' input. "
                                     "Please, provide one of {}".format(downstream, downstream_slot,
                                                                        [i.name for i in downstream.inputs]))

        if upstream_slot not in {i.name for i in upstream.outputs}:
            raise DaskPipesException("Invalid value for upstream_slot: {} does not have '{}' output. "
                                     "Please, provide one of {}".format(upstream, upstream_slot,
                                                                        [i.name for i in upstream.outputs]))

        self.upstream_slot = upstream_slot
        self.downstream_slot = downstream_slot

    @staticmethod
    def validate(operator_con):
        assert_subclass(operator_con, OperatorConnection)

    def to_dict(self):
        d = super().to_dict()
        d['upstream_slot'] = self.upstream_slot
        d['downstream_slot'] = self.downstream_slot
        return d

    @classmethod
    def params_from_dict(cls, graph, d):
        args, kwargs = super().params_from_dict(graph, d)
        kwargs['upstream_slot'] = d.get('upstream_slot')
        kwargs['downstream_slot'] = d.get('downstream_slot')
        return args, kwargs

    def __repr__(self):
        return "<{}({}['{}'] -> {}['{}']>".format(self.__class__.__name__,
                                                  self._v1,
                                                  self.upstream_slot,
                                                  self._v2,
                                                  self.downstream_slot)


class OperatorBaseMeta(type):

    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        fit_wrapped.__signature__ = inspect.signature(func)  # transfer signature
        fit_wrapped.__doc__ = func.__doc__  # preserve docstring
        return fit_wrapped

    def __new__(mcs, name, bases, attrs):
        if 'fit' in attrs:
            attrs['fit'] = mcs.wrap_fit(attrs['fit'])
        else:
            raise DaskPipesException("Class does not have fit method")

        # Check transform parameters
        if 'transform' in attrs:
            if not isinstance(attrs['transform'], classmethod):
                raise DaskPipesException("Transform must be a classmethod")
        else:
            raise DaskPipesException("Class does not have transform method")

        return super().__new__(mcs, name, bases, attrs)


class OperatorBase(VertexBase, metaclass=OperatorBaseMeta):

    @staticmethod
    def validate(operator):
        assert_subclass(operator, OperatorBase)

    @property
    def outputs(self):
        ret_descr = get_return_description(self.fit)
        return ret_descr

    @property
    def inputs(self):
        arg_descr = get_arguments_description(self.fit)
        return arg_descr

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        """
        :param other:
        :type other: OperatorBase
        :param upstream_slot:
        :param downstream_slot:
        :return:
        """
        super().set_upstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def set_downstream(self, other,
                       upstream_slot: Optional[str] = None,
                       downstream_slot: Optional[str] = None, ):
        """
        :param other:
        :type other: OperatorBase
        :param upstream_slot:
        :param downstream_slot:
        :return:
        """
        super().set_downstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def fit(self, *args, **kwargs):
        """
        Infer parameters prior to transforming dataset
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, *args, **kwargs):
        """
        Transform dataset
        :param params:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()


inspect.Signature()


class ExampleOperator(OperatorBase):

    def fit(self, dataset: "Descr"):
        """Some docstring"""
        return dataset

    @classmethod
    def transform(cls, params, dataset: "Descr"):
        """Some docstring"""
        return dataset


class Pipeline(Graph):

    def __init__(self):
        super().__init__()
        self.inputs: List[PipelineInput] = list()

    def validate_edge(self, edge):
        OperatorConnection.validate(edge)

    def validate_vertex(self, vertex):
        OperatorBase.validate(vertex)

    def set_input(self, operator: OperatorBase, suffix=None):

        if suffix is None:
            suffix = ''

        operator_inputs = operator.inputs
        for op_input in operator_inputs:
            downstream_slot = op_input.name

            if downstream_slot is None:
                operator_inputs = operator.inputs
                if len(operator_inputs) > 1:
                    raise DaskPipesException(
                        "{} has multiple inputs, "
                        "downstream_slot must be one of {}".format(operator, [i.name for i in operator_inputs]))
                downstream_slot = operator_inputs[0].name

            arg_name = '{}{}'.format(downstream_slot, suffix)

            self.inputs.append(PipelineInput(arg_name=arg_name,
                                             downstream_slot=downstream_slot,
                                             downstream_operator=operator))
        operator.graph = self

        fit_func = self.fit.__func__
        transform_func = self.transform.__func__
        sign = inspect.signature(fit_func)

        # Construct new fit parameters
        old_params = [i for i in sign.parameters.values()
                      if not i.kind == inspect.Parameter.VAR_KEYWORD and
                      not i.kind == inspect.Parameter.VAR_POSITIONAL]
        additional_params = list()
        # Create additional parameters
        for arg_descr in operator.inputs:
            arg_descr: ArgumentDescription
            annot = arg_descr.type
            if annot == object:
                annot = inspect._empty
            new_param = inspect.Parameter(
                name=arg_descr.name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annot,
                default=None
            )
            dup_param = None
            dup_params = [i for i in old_params if i.name == new_param.name]
            if len(dup_params) > 0:
                dup_param = dup_params[0]

            if dup_param is not None:
                # TODO: update annotation if necessary
                pass
            else:
                additional_params.append(new_param)

        # TODO: construct docstring for fit and tranform

        new_params = old_params + additional_params

        fit_func.__signature__ = inspect.Signature(
            parameters=new_params,
            return_annotation=sign.return_annotation)

        transform_func.__signature__ = inspect.Signature(
            parameters=new_params,
            return_annotation=sign.return_annotation)

    def connect(self,
                upstream: OperatorBase,
                downstream: OperatorBase,
                upstream_slot: Optional[str] = None,
                downstream_slot: Optional[str] = None) -> OperatorConnection:

        # Infer upstream and downstream objects
        if upstream_slot is None:
            upstream_outputs = upstream.outputs
            if len(upstream_outputs) > 1:
                raise DaskPipesException("Upstream has multiple outputs, cannot infer upstream_slot. "
                                         "Please, provide upstream_slot as one of {}".format(
                    [i.name for i in upstream_outputs]))
            upstream_slot = upstream_outputs[0].name

        if downstream_slot is None:
            downstream_inputs = downstream.inputs
            if len(downstream_inputs) > 1:
                raise DaskPipesException("Downstream has multiple inputs, cannot infer downstream_slot. "
                                         "Please, provide downstream_slot as one of {}".format(
                    [i.name for i in downstream_inputs]))
            downstream_slot = downstream_inputs[0].name

        # Create edge
        edge = OperatorConnection(upstream=upstream,
                                  downstream=downstream,
                                  upstream_slot=upstream_slot,
                                  downstream_slot=downstream_slot)
        return self.add_edge(edge)

    # Change signature of functions
    def add_edge(self, edge: OperatorConnection, edge_id=None) -> OperatorConnection:
        super().add_edge(edge, edge_id=edge_id)
        return edge

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline. Sequentially calls fit_transform of child operators
        """
        print("Imma fitting!")
        print(args)
        print(kwargs)

    def transform(self, *args, **kwargs):
        """
        Method for transforming the data.
        Does not alter the fitted parameters
        :param args:
        :param kwargs:
        :return:
        """
        params = {k: v for k, v in kwargs.items()}
        if len(args) > 1:
            raise DaskPipesException("More than one positional dataset not supported")
        elif len(args) == 1:
            params[None] = args[0]

        operator_outputs = dict()

        for operator in VertexWidthFirst(self):
            operator: OperatorBase
            pass
