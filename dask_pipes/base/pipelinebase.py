from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, Dict, List, Optional
from collections import namedtuple
import inspect
from ..utils import (ArgumentDescription,
                     get_arguments_description,
                     get_return_description,
                     ReturnDescription,
                     assert_subclass)

__all__ = ['OperatorConnection', 'OperatorBase', 'PipelineBase', 'OperatorBaseMeta', 'ExampleOperator']

SLOT_UNNAMED = '<unnamed>'

OperatorInput = namedtuple("OperatorInput", ['input_arg', 'upstream_output_name', 'upstream_operator'])

PipelineInput = namedtuple("PipelineInput", ['arg_name', 'downstream_slot', 'downstream_operator'])
PipelineOutput = namedtuple("PipelineOutput", ['output_name', 'upstream_slot', 'upstream_operator'])


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

        Since this is defined in meta, we can get operator outputs just based on it's class
        :return: List of named tuples 'ReturnDescription' with argument name, type and description
        """
        ret_descr = get_return_description(getattr(self, 'transform'))
        return ret_descr

    @property
    def inputs(self) -> List[ArgumentDescription]:
        """
        Inputs of an operator - parameters of self.fit function
        must be equal to parameters of self.transform

        Since this is defined in meta, we can get operator outputs just based on it's class
        :return:
        """
        arg_descr = get_arguments_description(getattr(self, 'fit'))
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
        """
        Used for decorating fit method and maybe basic checks
        In previous versions fit decorator measured computation time
        For the time being, it seems pretty much obsolete.
        Just left it here as codebase for future improvements
        """
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
        """
        Inputs of an operator - parameters of self.fit function
        must be equal to parameters of self.transform

        This duplicates OperatorBaseMeta's inputs property for convenience
        (being able to do obj.inputs instead of obj.__class__.inputs)
        :return: List of named tuples 'ReturnDescription' with argument name, type and description
        """
        return self.__class__.inputs

    @property
    def outputs(self) -> List[ReturnDescription]:
        """
        Outputs of an operator (parses annotation of .transform function)

        This duplicates OperatorBaseMeta's inputs property for convenience
        (being able to do obj.outputs instead of obj.__class__.outputs)
        :return:
        """
        return self.__class__.outputs

    @staticmethod
    def validate(operator):
        """
        Used by Graph to validate vertices when adding them
        Checks that operator is valid OperatorBase
            > check that operator is subclass of OperatorBase
        :param operator: Any
        :return:
        """
        assert_subclass(operator, OperatorBase)

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        """
        Make other Operator upstream of current
        Pipes output of specific slot of upstream operator to specific slot of current operator
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and operators
        >>> p = PipelineBase()
        >>> op1 = ExampleOperator('op1')
        >>> op2 = ExampleOperator('op2')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>>
        >>> # Set upstream node, add op2 to pipeline
        >>> # input ->> op1 ->> op2
        >>> op2.set_upstream(op1)

        :param other: Operator to set upstream
        :type other: OperatorBase
        :param upstream_slot: upstream's output (name of one of other.outputs) or None
        if None is passed and upstream has only one output, automatically deducts upstream_slot
        :param downstream_slot: this input slot (name of one of self.inputs) or None
        if None is passed and this operator has only one input, automatically deducts upstream_slot
        :return: None
        """
        super().set_upstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def set_downstream(self, other,
                       upstream_slot: Optional[str] = None,
                       downstream_slot: Optional[str] = None, ):
        """
        Make other Operator upstream of current
        Pipes output of specific slot of upstream operator to specific slot of current operator
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and operators
        >>> p = PipelineBase()
        >>> op1 = ExampleOperator('op1')
        >>> op2 = ExampleOperator('op2')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>>
        >>> # Set downstream node, add op2 to pipeline
        >>> # input ->> op1 ->> op2
        >>> op1.set_downstream(op2)

        :param other: Operator to set downstream
        :type other: OperatorBase
        :param upstream_slot: this operator output (name of one of self.outputs) or None
        if None is passed and this operator has only one output, automatically deducts upstream_slot
        :param downstream_slot: downstream's input slot (name of one of other.inputs) or None
        if None is passed and downstream operator has only one input, automatically deducts upstream_slot
        :return: None
        """
        super().set_downstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Infer parameters prior to transforming dataset
        To be implemented by subclass pipelines

        Signature of this function is changed dynamically.
        As user sets pipeline input operators, they are added to parameters of fit

        Important!
        Reason behind returning dict instead of storing fitted data inside operator:
        1. One operator can be fitted multiple time in one pipeline
        2. Do not use pickle to save model parameters. instead, serialize them explicitly to yaml or csv files

        # Example:
        >>> import pandas as pd
        >>> ds = pd.DataFrame([[1,2,3]])
        >>> p = PipelineBase()
        >>> op1 = ExampleOperator('op1')
        >>> op2 = ExampleOperator('op2')
        >>> p.set_input(op1)
        >>> op1.set_downstream(op2)
        >>> print(p.inputs) # ['dataset']
        >>> # since pipeline has single input, it's allowed to pass ds as positional
        >>> p.fit(ds)
        >>> # but it can be also passed as key-word (consult p.inputs)
        >>> p.fit(datset=ds)

        :param args:
        :param kwargs:
        :return: self
        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, *args, **kwargs):
        """
        To be implemented by subclass pipelines

        Signature of this function is changed dynamically.
        As user sets pipeline input operators, they are added to parameters of transform

        # Example
        >>> import pandas as pd
        >>> ds = pd.DataFrame([[1,2,3]])
        >>> p = PipelineBase()
        >>> op1 = ExampleOperator('op1')
        >>> op2 = ExampleOperator('op2')
        >>> p.set_input(op1)
        >>> op1.set_downstream(op2)
        >>> op1.fit(ds)
        >>>
        >>> print(p.outputs) # ['op3_<unnamed>'] '<unnamed>' is because transform has not defined result annotations
        >>> # since pipeline has single input, it's allowed to pass ds as positional
        >>> p.transform(ds) # since output of pipeline is single dataset, returns ds.
        >>> # If multiple datasets are returned, returns dictionary {dataset_name: dataset} (see p.outputs)
        >>> # but it can be also passed as key-word (consult p.inputs)
        >>> p.transform(datset=ds) # since output of pipeline is single dataset, returns ds

        :param args: placeholder for pipeline inputs.
        Positional arguments are allowed only if pipeline has single input.
        Positional arguments are not allowed to be mixed with key-word arguments
        This is for safety purposes, because order of inputs can be changed dynamically during runtime
        :param kwargs: placeholder for pipeline inputs.
        Key-word inputs to pipeline
        name of argument is equal to input operators' input slots + optional suffix
        single key-word argument can be piped to multiple operators
        :return: Operator outputs that were not piped anywhere
        """
        raise NotImplementedError()

    def __repr__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self.name)


class ExampleOperator(OperatorBase):

    def fit(self, dataset) -> Dict[str, Any]:
        """
        Example docstring
        :param dataset: dataset to fit to
        :return: Parameters passed to .transform
        """
        return {'a': 1}

    @classmethod
    def transform(cls, params, dataset):
        """
        Example docstring
        :param params: - obligatory parameter, containing output of fit
        :param dataset: dataset to transform
        :return:
        """
        assert params['a'] == 1
        return dataset


class PipelineBase(Graph):
    """
    Pipeline is a graph structure, containing relationships between OperatorBase (vertices)
    with OperatorConnection as edges

    Defines fit and transform methods which iterate vertices in width-first order,
    calling fit() and transform() methods of operators and
    piping outputs of upstream operators to inputs of downstream operators
    """

    def __init__(self):
        super().__init__()

        # Pipeline inputs. use set_input to add input
        self._inputs: List[PipelineInput] = list()

        self.operator_dict = dict()

    def add_vertex(self, operator: OperatorBase, vertex_id=None):
        """
        Add operator to current pipeline
        :param operator: operator to add
        :param vertex_id: operator's id or None (if None - autoincrement) (used for deserialization from disk)
        :type vertex_id: int
        :return:
        """
        if operator.name in self.operator_dict:
            if self.operator_dict[operator.name] is not operator:
                raise DaskPipesException("Duplicate name for operator {}".format(operator))
        self.operator_dict[operator.name] = operator
        super().add_vertex(operator, vertex_id=vertex_id)

    def remove_vertex(self, operator: OperatorBase) -> VertexBase:
        """
        Remove operator from pipeline
        :param operator: operator to remove (must be in current pipeline)
        :return:
        """
        if operator.graph is not self:
            raise DaskPipesException("Operator does not belong to pipeline")
        if operator.name in self.operator_dict:
            if self.operator_dict[operator.name] is not operator:
                raise ValueError("Operator does not equal to operator_dict['{}']".format(operator.name))
            del self.operator_dict[operator.name]
        else:
            raise ValueError("Invalid operator name")
        self.remove_input(operator)
        return super().remove_vertex(operator)

    @property
    def _outputs(self) -> List[PipelineOutput]:
        """
        Get detailed info about pipeline outputs
        :return: list out named tuples defining unused operators' outputs. output of .transform() method
        """
        outputs = []

        for v_id, vertex in self._vertices.items():

            vertex: OperatorBase

            v_outputs = {i.name for i in vertex.outputs}

            for edge in self.get_downstream_edges(vertex):
                edge: OperatorConnection
                if edge.upstream_slot in v_outputs:
                    v_outputs.remove(edge.upstream_slot)
                if len(v_outputs) == 0:
                    break

            for v_out in v_outputs:
                out = '{}_{}'.format(vertex.name, v_out)
                outputs.append(PipelineOutput(output_name=out, upstream_slot=v_out, upstream_operator=vertex))

        return outputs

    @property
    def inputs(self) -> List[str]:
        """
        Get human-friendly list of names of current pipeline inputs without duplicates
        :return: list of argument names for fit function
        """
        inputs = []
        for arg in self._inputs:
            if arg.arg_name not in inputs:
                inputs.append(arg.arg_name)
        return inputs

    @property
    def outputs(self) -> List[str]:
        """
        Get human-friendly list of output names of .transform() function without detailed info about operators
        :return: list of operators's outputs that have no downstream operators. Transform output
        """
        return [o.output_name for o in self._outputs]

    def validate_edge(self, edge):
        """
        Used by base graph class to check if edge can be added to current graph
        Checks if edge is subclass of OperatorConnection
        :param edge:
        :return:
        """
        OperatorConnection.validate(edge)

    def validate_vertex(self, vertex):
        """
        Used by base graph to check if vertex can be added to current graph
        Checks if vertex is subclass of OperatorBase
        :param vertex:
        :return:
        """
        OperatorBase.validate(vertex)

    def dump_params(self, run_name: str, vertex_name: str, params: Dict[str, Any]):
        """
        Save output of operator's fit method somewhere
        to be implemented in subclasses
        :param run_name: Name of run, specified in fit and transform methods
        :param vertex_name: vertex name, which params we're saving. Can be used to infer filenames
        :param params: params to save
        :return:
        """
        raise NotImplementedError()

    def dump_outputs(self, run_name: str, vertex_name: str, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save output of operator's transform method somewhere (used to persist dataframes after calculations)
        to be implemented in subclasses
        :param run_name: Name of run, specified in fit and transform methods
        :param vertex_name: Name of vertex which output we're dumping. Can used to infer filenames
        :param outputs: Dataframes to save
        :return:
        """
        raise NotImplementedError()

    def load_params(self, run_name: str, vertex_name: str) -> Dict[str, Any]:
        """
        Load parameters which fit function previously returned and which were dumped using dump_params
        Loaded params are shortly passed to .transform method
        to be implemented in subclasses
        :param run_name: Name of run, specified in fit and transform methods.
        :param vertex_name: Name of vertex which params we're loading. Can used to infer filenames
        :return: parameters
        """
        raise NotImplementedError()

    def load_outputs(self, run_name: str, vertex_name: str) -> Dict[str, Any]:
        """
        Load outputs of .transform method which previously was dumped using dump_params
        loaded outputs are shortly passed to .transform method or yielded as pipeline output
        :param run_name: Name of run, specified in fit and transform methods
        :param vertex_name: Name of vertex which output we're loading. Can used to infer filenames
        :return:
        """
        raise NotImplementedError()

    def set_input(self, operator: OperatorBase, suffix=None):
        """
        Register operator as pipeline's input.
        All operator inputs should be passed in fit function
        by default, uses fit argument names equal to operator input arguments.

        Caution:
        If multiple operators which have same input names are set as input,
        pipeline will pass data from argument to multiple operators
        suffix can be used to differentiate inputs for operators with same argument names

        Example:
        >>> # Construct pipeline and operators
        >>> p = PipelineBase()
        >>> op1 = ExampleOperator('op1')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>> # p.fit signature changed
        >>> # also, p.inputs now returns ['dataset']

        :param operator: operator to set as input
        :param suffix: suffix to add to operator's inputs before setting them as pipeline inputs
        :return:
        """

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

        self._update_fit_transform_signatures()

    def remove_input(self, operator: OperatorBase):
        """
        Unset operator as pipeline output
        :param operator: operator to unset as output
        :return:
        """
        # Find inputs to remove
        self._inputs = [i for i in self._inputs if i.downstream_operator is not operator]
        self._update_fit_transform_signatures()

    def _update_fit_transform_signatures(self):
        """
        Called when input operator is added or removed.
        Infers and updates signatures of fit and transform methods for user-friendly hints
        :return:
        """
        fit_func = getattr(self.fit, '__func__')  # Since fit is method, it has __func__
        transform_func = getattr(self.transform, '__func__')  # Since transform is method, it has __func__
        sign = inspect.signature(fit_func)
        new_params = list(sign.parameters.values())[:2]
        seen_params = set()

        for inp in self._inputs:
            if inp.arg_name in seen_params:
                continue
            param = inspect.Parameter(
                name=inp.arg_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None)
            new_params.append(param)
            seen_params.add(inp.arg_name)

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
        """
        General method for connecting tto operators
        Creates directed connection with upstream and downstream slots

        if connection between slots was already made, raises DaskPipesException
        if downstream_slot already has piped input, raises DaskPipesException,
        since multiple inputs for one slot are not allowed

        :param upstream: operator to set upstream
        :param downstream: operator to set downstream
        :param upstream_slot: slot of upstream operator to pass to downstream
        :param downstream_slot: slot of downstream operator to receive data into
        :return:
        """

        # Infer upstream and downstream objects
        if upstream_slot is None:
            upstream_outputs = upstream.outputs
            if len(upstream_outputs) > 1:
                raise DaskPipesException(
                    "Upstream has multiple outputs, cannot infer upstream_slot. "
                    "Please, provide upstream_slot as one of {}".format(
                        [i.name for i in upstream_outputs]))
            upstream_slot = upstream_outputs[0].name

        if downstream_slot is None:
            downstream_inputs = downstream.inputs
            if len(downstream_inputs) > 1:
                raise DaskPipesException(
                    "Downstream has multiple inputs, cannot infer downstream_slot. "
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
    def add_edge(self, op_connection: OperatorConnection, edge_id=None) -> OperatorConnection:
        """
        Add OperatorConnection to current pipeline
        op_connection must already have assigned upstream and downstream operators
        :param op_connection: connection to add
        :param edge_id: integer id of edge, auto-increment if None (used for deserialization from disk)
        :return:
        """
        super().add_edge(op_connection, edge_id=edge_id)
        return op_connection

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
        """
        Get dictionary of key-values from return of function

        Converts list, tuple or dict to key-value pairs
        makes sanity checks

        :param operator_cls: class of operator to read transform return annotation from
        :param output: return value of .transform() method
        :return:
        """
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

    def _check_arguements(self, op, edge, operator_arguments, downstream_op, op_result):
        if edge.downstream_slot in operator_arguments[downstream_op]:
            raise DaskPipesException(
                "Duplicate argument for {}['{}']. "
                "Already contains {}".format(downstream_op, edge.downstream_slot,
                                             operator_arguments[downstream_op][edge.downstream_slot]))
        if downstream_op not in operator_arguments:
            raise DaskPipesException("Pipeline does not have {}".format(downstream_op))
        if edge.upstream_slot not in op_result:
            raise DaskPipesException(
                "Operator {} did not return expected {}; "
                "recieved {}".format(op, edge.upstream_slot, list(op_result.keys())))

    def _iterate_graph(self, func, *args, **kwargs):
        """
        Helper function used in fit and transform methods
        :param func: function to apply during iteration
        :param args: pipeline positional input
        :param kwargs: pipeline key-word input
        :return:
        """
        args = self._parse_arguments(*args, **kwargs)
        operator_arguments = {op: dict() for op in self.vertices}
        for inp in self._inputs:
            operator_arguments[inp.downstream_operator][inp.downstream_slot] = args[inp.arg_name]

        outputs = dict()

        for op in VertexWidthFirst(self):
            op: OperatorBase
            op_args = operator_arguments[op]

            if len(op_args) == 0:
                raise DaskPipesException("No input for operator {}".format(op))

            downstream_edges = self.get_downstream_edges(op)
            has_downstream = len(downstream_edges) > 0

            op_result = func(op, has_downstream=has_downstream, **op_args)

            if op_result is None:
                continue

            if not isinstance(op_result, dict):
                raise DaskPipesException(
                    "Invalid return. Expected {}, received {}".format(dict.__name__,
                                                                      op_result.__class__.__name__))

            unused_output = set(op_result.keys())

            # Get downstream edges
            for edge in downstream_edges:
                downstream_op: OperatorBase = edge.downstream
                edge: OperatorConnection
                self._check_arguements(op, edge, operator_arguments, downstream_op, op_result)
                if edge.upstream_slot in unused_output:
                    unused_output.remove(edge.upstream_slot)
                operator_arguments[downstream_op][edge.downstream_slot] = op_result[edge.upstream_slot]

            for upstream_slot in unused_output:
                outputs[(op, upstream_slot)] = op_result[upstream_slot]

        return outputs

    def _fit(self, run_name, op, **op_args):
        """
        Helper function used for fit call and dumping params
        :param run_name:
        :param op:
        :param op_args:
        :return:
        """
        # ----------------------
        # Fit operator
        # ----------------------
        params = op.fit(**op_args)

        # ----------------------
        # Save fitted parameters
        # ----------------------
        try:
            self.dump_params(run_name, op.name, params)
        except Exception as ex:
            raise DaskPipesException("Error saving parameters of {}".format(op)) from ex

    def _transform(self, run_name, op, **op_args):
        """
        Helper function, used for transform call and dumping outputs
        :param run_name:
        :param op:
        :param op_args:
        :return:
        """
        params = self.load_params(run_name, op.name)
        if not isinstance(params, dict):
            raise DaskPipesException("Error after loading parameters. Expected {}, received {}".format(
                dict.__name__, params.__class__.__name__))

        # ----------------------
        # Transform operator
        # ----------------------
        op_result = PipelineBase._parse_operator_output(op.__class__,
                                                        op.__class__.transform(params, **op_args))
        if len(op_result) == 0:
            raise DaskPipesException("Operator {} did not return anything".format(op))

        # ----------------------
        # Save persist datasets
        # ----------------------
        try:
            self.dump_outputs(run_name, op.name, op_result)
        except Exception as ex:
            raise DaskPipesException("Error saving outputs of {}".format(op)) from ex

        try:
            op_result = self.load_outputs(run_name, op.name)
        except Exception as ex:
            raise DaskPipesException("Error loading outputs of {}".format(op)) from ex

        if not isinstance(params, dict):
            raise DaskPipesException("Error after loading persist. Expected {}, received {}".format(
                Dict[str, Any].__name__, params.__class__.__name__))
        if len(op_result) == 0:
            raise DaskPipesException("Error loading outputs {}. got empty dict".format(op.name))
        return op_result

    def fit(self, run_name, *args, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order
        :param run_name: name of run for logging and dumping data
        :param args: pipeline positional input to pass to input operators
        :param kwargs: pipeline key-word input to  pass to input operators
        :return: self
        """

        def func(op, has_downstream, **op_args):
            self._fit(run_name, op, **op_args)
            if has_downstream:
                return self._transform(run_name, op, **op_args)

        self._iterate_graph(func, *args, **kwargs)
        return self

    def transform(self, run_name, *args, **kwargs):
        """
        Method for transforming based on previously fitted parameters
        :param run_name: name of run for logging and dumping data
        :param args: pipeline positional input to pass to input operators
        :param kwargs: pipeline key-word input to  pass to input operators
        :return: dictionary of datasets, or single dataset.
        Output of operators that was not piped anywhere
        """

        def func(op, has_downstream, **op_args):
            return self._transform(run_name, op, **op_args)

        outputs = self._iterate_graph(func, *args, **kwargs)
        rv = {'{}_{}'.format(k[0].name, k[1]): v for k, v in outputs.items()}
        # Return just dataframe if len 1
        if len(rv) == 1:
            return next(iter(rv.values()))
        return rv
