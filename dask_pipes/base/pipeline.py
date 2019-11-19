from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, Dict, List, Tuple
import inspect

__all__ = ['OperatorConnection', 'OperatorBase', 'Pipeline', 'OperatorBaseMeta', 'ExampleOperator']


class OperatorConnection(EdgeBase):

    def __init__(self, *args, upstream_slot=None, downstream_slot=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream_slot = upstream_slot
        self.downstream_slot = downstream_slot

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


class OperatorBaseMeta(type):

    @staticmethod
    def get_outputs(func):
        func_name = func.__qualname__
        return_type = func.__annotations__.get('return', Any)
        if isinstance(return_type, tuple) or isinstance(return_type, list):
            if len(return_type) == 0:
                raise DaskPipesException("return type '{}' of {} not understood".format(repr(return_type), func_name))
            rv = list()
            for v in return_type:
                if isinstance(v, tuple) or isinstance(v, list):
                    if len(v) != 2:
                        raise NotImplementedError()
                    var_name = v[0]
                    var_type = v[1]
                elif isinstance(v, str):
                    var_name = v[0]
                    var_type = object
                else:
                    raise NotImplementedError()
                if var_name in set((i[0] for i in rv)):
                    raise DaskPipesException("duplicate return name: {} of {}".format(var_name, func_name))
                rv.append((var_name, var_type))
            return rv
        elif isinstance(return_type, dict):
            return tuple(((k, v) for k, v in return_type.items()))
        elif isinstance(return_type, str):
            return [(return_type, object)]
        else:
            return [(None, return_type)]

    @staticmethod
    def get_inputs(func, skip_first=1):
        fullargspec = inspect.getfullargspec(func)
        argspec = fullargspec.args[skip_first:]
        annotations = func.__annotations__
        rv = [(k, annotations.get(k, object)) for k in argspec]
        return rv

    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        fit_wrapped.__signature__ = inspect.signature(func)  # transfer signature
        fit_wrapped.__doc__ = func.__doc__  # preserve docstring
        return fit_wrapped

    def __new__(mcs, name, bases, attrs):
        global transform_test
        """If the class has a 'run' method, wrap it"""
        if 'fit' in attrs:
            # Infer inputs and outputs prior to wrapping method
            attrs['inputs'] = OperatorBaseMeta.get_inputs(attrs['fit'])
            attrs['outputs'] = OperatorBaseMeta.get_outputs(attrs['fit'])

            attrs['fit'] = mcs.wrap_fit(attrs['fit'])
        else:
            raise DaskPipesException("Class does not have fit method")

        # Check transform parameters
        if 'transform' in attrs:
            if not isinstance(attrs['transform'], classmethod):
                raise DaskPipesException("Transform must be a classmethod")
            transform_func = attrs['transform'].__func__
            func_name = transform_func.__name__
            transform_inputs = OperatorBaseMeta.get_inputs(transform_func, skip_first=2)
            transform_outputs = OperatorBaseMeta.get_outputs(transform_func)
            if transform_inputs != attrs['inputs']:
                raise DaskPipesException(
                    "{} inputs do not match fit inputs ({} != {})".format(func_name, transform_inputs,
                                                                          attrs['inputs']))
            if transform_outputs != attrs['outputs']:
                raise DaskPipesException(
                    "{} inputs do not match fit outputs ({} != {})".format(func_name,
                                                                           transform_outputs,
                                                                           attrs['outputs']))
        else:
            raise DaskPipesException("Class does not have transform method")

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
    def transform(cls, *args, **kwargs):
        """
        Transform dataset
        :param params:
        :param kwargs:
        :return:
        """
        pass


class ExampleOperator(OperatorBase):

    def fit(self, dataset: "Descr"):
        """Some docstring"""
        return dataset

    @classmethod
    def transform(cls, params, dataset: "Descr"):
        """Some docstring"""
        return dataset


class Pipeline(Graph):

    def validate_edge(self, edge: OperatorConnection):
        if not issubclass(edge.__class__, OperatorConnection):
            raise DaskPipesException(
                "Expected subclass of {}; got {}".format(OperatorConnection.__name__, edge.__class__.__name__))

    def validate_vertex(self, vertex: OperatorBase):
        if not issubclass(vertex.__class__, OperatorBase):
            raise DaskPipesException(
                "Expected subclass of {}; got {}".format(OperatorBase.__name__, vertex.__class__.__name__))

    def get_default_edge(self, *args, **kwargs):
        return OperatorConnection(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline. Sequentially calls fit_transform of child operators
        :param args:
        :param kwargs:
        :return:
        """
        params = {k: v for k, v in kwargs.items()}
        if len(args) > 1:
            raise DaskPipesException("More than one positional dataset not supported")
        elif len(args) == 1:
            params[None] = args[0]

        previous_outputs = dict()
        output_nodes = list()

        for operator in VertexWidthFirst(self):
            operator: OperatorBase
            operator_inputs: List[Tuple[str, Any]] = operator.inputs
            operator_input_names = [i[0] for i in operator_inputs]
            operator_input_types = [i[1] for i in operator_inputs]

            operator_outputs: List[Tuple[str, Any]] = operator.outputs
            operator_output_names = [i[0] for i in operator_outputs]
            operator_output_types = [i[1] for i in operator_outputs]

            # Infer inputs
            input_dict = dict()
            upstream_vertices = self.get_upstream_vertices(operator)
            downstream_vertices = self.get_downstream_vertices(operator)
            if len(downstream_vertices) == 0:
                output_nodes.append(operator)

            if len(upstream_vertices) > 0:
                for upstream_vertex in upstream_vertices:
                    v_outputs = previous_outputs[upstream_vertex]
                    print(v_outputs)
                    edges: List[OperatorConnection] = self.get_edges(upstream_vertex, operator)
                    slot_mapping = dict()
                    for edge in edges:
                        slot_mapping[edge.upstream_slot] = edge.downstream_slot

                    for k, v in v_outputs.items():
                        upstream_slot = k
                        try:
                            downstream_slot = slot_mapping[upstream_slot]
                        except KeyError:
                            expected_outputs = [{None: '<positional>'}.get(i, i) for i in slot_mapping.keys()]
                            raise DaskPipesException(
                                "{} contains unknown output {}, avaliable: {}".format(
                                    operator, repr(upstream_slot), repr(expected_outputs))) from None
                        if downstream_slot not in operator_input_names:
                            if downstream_slot is None:
                                raise DaskPipesException(
                                    "{} does not have positional input".format(operator))
                            else:
                                raise DaskPipesException(
                                    "{} does not have input '{}'".format(operator, downstream_slot))
                        if downstream_slot in input_dict:
                            if downstream_slot is None:
                                raise DaskPipesException(
                                    "Duplicate input for positional argument of {}".format(operator))
                            else:
                                raise DaskPipesException(
                                    "Duplicate input for slot '{}' of {}".format(k, operator))
                        else:
                            input_dict[downstream_slot] = v
            else:
                # Input vertex, assign from input
                input_dict = params

            args = tuple() if None not in input_dict else (input_dict[None],)
            kwargs = {k: v for k, v in input_dict.items() if k is not None}

            operator.fit(*args, **kwargs)
            params = None
            outputs = operator.transform(params, *args, **kwargs)

            if not isinstance(outputs, list) and not isinstance(outputs, dict) and not isinstance(outputs, tuple):
                outputs = (outputs,)

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                if len(outputs) != len(operator_output_names):
                    raise DaskPipesException(
                        "{} output does not match declared ({} != {})".format(operator,
                                                                              [i.__class__.__name__ for i in outputs],
                                                                              operator_output_types))
                output_dict = dict()
                for output, output_name in zip(outputs, operator_output_names):
                    output_dict[output_name] = output
                previous_outputs[operator] = output_dict
            elif isinstance(outputs, dict):
                output_dict = dict()
                for k in outputs.keys():
                    if k not in operator_output_names:
                        raise DaskPipesException(
                            "{} output contains unseen key {}".format(operator.__class__.__name__, k))
                for output_name in operator_output_names:
                    if output_name not in outputs:
                        raise DaskPipesException(
                            "{} output does not contain declared key {}".format(operator.__class__.__name__,
                                                                                output_name))
                    output = outputs[output_name]
                    output_dict[output_name] = output
                previous_outputs[operator] = output_dict

        return output_nodes

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
            operator_inputs: Dict[str, Any] = operator.inputs
            operator_outputs: Dict[str, Any] = operator.outputs
