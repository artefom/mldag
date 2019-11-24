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

    @property
    def outputs(self) -> List[ReturnDescription]:
        """
        Outputs of an operator (parses annotation of .transform function)
        :return:
        """
        ret_descr = get_return_description(self.transform)
        return ret_descr

    @property
    def inputs(self) -> List[ArgumentDescription]:
        """
        Inputs of an operator - parameters of self.fit function
        must be equal to parameters of self.transform
        :return:
        """
        arg_descr = get_arguments_description(self.fit)
        return arg_descr

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

    def __init__(self, name):
        super().__init__()
        self.name = name

    @property
    def inputs(self) -> List[ArgumentDescription]:
        return self.__class__.inputs

    @property
    def outputs(self) -> List[ReturnDescription]:
        return self.__class__.outputs

    @staticmethod
    def validate(operator):
        assert_subclass(operator, OperatorBase)

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

    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Infer parameters prior to transforming dataset

        Important!
        Reason behind returning dict instead of storing fitted data inside operator:
        1. One operator can be fitted multiple time in one pipeline
        2. Do not use pickle to save model parameters. instead, serialize them explicitly to yaml or csv files

        :param args:
        :param kwargs:
        :return: Dictionary of parameters to use in transform
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

    def __repr__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self.name)


inspect.Signature()


class ExampleOperator(OperatorBase):

    def fit(self, dataset) -> Dict[str, Any]:
        """Some docstring"""
        print("Fittin {}".format(self))
        return {'a': 1}

    @classmethod
    def transform(cls, params, dataset):
        """Some docstring"""
        print("Transformin {}".format(cls.__name__))
        return dataset


class Pipeline(Graph):

    def __init__(self):
        super().__init__()

        # Pipeline inputs. use set_input to add input
        self._inputs: List[PipelineInput] = list()

        self.operator_dict = dict()

    def add_vertex(self, vertex: OperatorBase, vertex_id=None):
        if vertex.name in self.operator_dict:
            if self.operator_dict[vertex.name] is not vertex:
                raise DaskPipesException("Duplicate name for operator {}".format(vertex))
        self.operator_dict[vertex.name] = vertex
        super().add_vertex(vertex, vertex_id=vertex_id)

    def remove_vertex(self, vertex) -> VertexBase:
        if vertex.name in self.operator_dict:
            if self.operator_dict[vertex.name] is not vertex:
                raise ValueError("Vertex does not equal to operator_dict['{}']".format(vertex.name))
            del self.operator_dict[vertex.name]
        else:
            raise ValueError("Invalid vertex name")
        self.remove_input(vertex)
        return super().remove_vertex(vertex)

    def validate_edge(self, edge):
        OperatorConnection.validate(edge)

    def validate_vertex(self, vertex):
        OperatorBase.validate(vertex)

    def dump_params(self, vertex_name: str, run_name: str, params: Dict[str, Any]):
        raise NotImplementedError()

    def dump_outputs(self, vertex_name: str, run_name: str, outputs: Dict[str, Any]):
        raise NotImplementedError()

    def load_params(self, vertex_name: str, run_name: str):
        raise NotImplementedError()

    def load_outputs(self, vertex_name: str, run_name: str):
        raise NotImplementedError()

    def remove_input(self, operator: OperatorBase):
        # Find inputs to remove
        self._inputs = [i for i in self._inputs if i.downstream_operator is not operator]
        self.update_fit_transform_signatures()

    def update_fit_transform_signatures(self):
        fit_func = self.fit.__func__
        transform_func = self.transform.__func__
        sign = inspect.signature(fit_func)
        self_param = next(iter(sign.parameters.values()))
        new_params = [self_param]
        seen_params = set()

        for inp in self._inputs:
            if inp.arg_name in seen_params:
                continue
            param = inspect.Parameter(
                name=inp.arg_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=inspect._empty,
                default=None)
            new_params.append(param)
            seen_params.add(inp.arg_name)

        fit_func.__signature__ = inspect.Signature(
            parameters=new_params,
            return_annotation=sign.return_annotation)

        transform_func.__signature__ = inspect.Signature(
            parameters=new_params,
            return_annotation=sign.return_annotation)

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

            self._inputs.append(PipelineInput(arg_name=arg_name,
                                              downstream_slot=downstream_slot,
                                              downstream_operator=operator))
        operator.graph = self

        self.update_fit_transform_signatures()

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

    def _parse_arguments(self, *args, **kwargs) -> Dict[str, Any]:

        expected_inputs = {i.arg_name for i in self._inputs}
        unseen_inputs = {i for i in expected_inputs}

        rv = {k: v for k, v in kwargs.items()}

        if len(args) > 0:
            if len(args) > 1:
                raise DaskPipesException("Multiple positional arguments not allowed")
            if len(rv) > 0:
                raise DaskPipesException("Cannot mix positional and key-word arguments")
            if len(expected_inputs) > 1:
                raise DaskPipesException(
                    "Positional arguments not allowed in case of multiple inputs needed. "
                    "Expected {} key-word arguments; got {} positional".format(len(expected_inputs),
                                                                               len(args)))
            rv[next(iter(expected_inputs))] = next(iter(args))

        for k, v in rv.items():
            if k not in expected_inputs:
                raise DaskPipesException(
                    "Unexpected argument: '{}'. Should be one of {}".format(
                        k, expected_inputs
                    ))
            unseen_inputs.remove(k)

        if len(unseen_inputs) > 0:
            raise DaskPipesException("Unfilled arguments: {}".format(unseen_inputs))

        return rv

    @staticmethod
    def _parse_operator_output(operator_cls, output):
        expected_result: List[ReturnDescription] = operator_cls.outputs
        expected_keys = [i.name for i in expected_result]
        if not isinstance(output, list) and not isinstance(output, tuple) and not isinstance(output, dict):
            output = (output,)

        if isinstance(output, list) or isinstance(output, tuple):  # In case returned list
            # Check that output length matches
            if len(output) != len(expected_result):
                raise DaskPipesException(
                    "{}.transform result length does not match expected. Expected {}, got {}".format(
                        operator_cls.__name__,
                        ['{}: {}'.format(i.name, i.type) for i in expected_result],
                        [i.__class__.__name__ for i in output]
                    ))
            output = {k: v for k, v in zip(expected_keys, output)}
        elif isinstance(output, dict):  # In case returned dict
            got_keys = set(output.keys())
            # Check that keys match
            if set(expected_keys) != got_keys:
                raise DaskPipesException(
                    "{}.transform result does not match expected. Expected keys: {}, received: {}".format(
                        operator_cls.__name__,
                        expected_keys, got_keys
                    ))
        else:
            raise DaskPipesException("Unknown return type: {}".format(output.__class__.__name__))

        return output

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline. Sequentially calls fit_transform of child operators
        """
        args = self._parse_arguments(*args, **kwargs)
        operator_arguments = {op: dict() for op in self.vertices}
        for inp in self._inputs:
            operator_arguments[inp.downstream_operator][inp.downstream_slot] = args[inp.arg_name]

        for op in VertexWidthFirst(self):
            op: OperatorBase
            op_args = operator_arguments[op]
            if len(op_args) == 0:
                raise DaskPipesException("No input for operator {}".format(op))

            # ----------------------
            # Fit operator
            # ----------------------
            params = op.fit(**op_args)

            # ----------------------
            # Save fitted parameters
            # ----------------------
            # TODO: Save params for future use

            # ----------------------
            # Transform operator
            # ----------------------
            op_result = Pipeline._parse_operator_output(op.__class__,
                                                        op.__class__.transform(params, **op_args))

            # Get downstream edges
            for downstream_op in self.get_downstream_vertices(op):
                edges = self.get_edges(op, downstream_op)
                for edge in edges:
                    edge: OperatorConnection
                    if edge.downstream_slot in operator_arguments[downstream_op]:
                        raise DaskPipesException(
                            "Duplicate argument for {}['{}']. "
                            "Already contains {}".format(downstream_op, edge.downstream_slot,
                                                         operator_arguments[downstream_op][edge.downstream_slot]))
                    operator_arguments[downstream_op][edge.downstream_slot] = op_result[edge.upstream_slot]

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
