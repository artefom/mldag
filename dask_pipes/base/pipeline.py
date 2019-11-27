from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, Dict, List, Optional
from types import MethodType
from collections import namedtuple
from sklearn.base import BaseEstimator
import inspect
from ..utils import (ArgumentDescription,
                     get_arguments_description,
                     get_return_description,
                     ReturnDescription,
                     assert_subclass,
                     replace_signature)

__all__ = ['NodeConnection', 'NodeBase', 'Pipeline', 'NodeBaseMeta', 'ExampleNode']

NodeInput = namedtuple("NodeInput", ['input_arg', 'upstream_output_name', 'upstream_node'])

PipelineInput = namedtuple("PipelineInput", ['arg_name', 'downstream_slot', 'downstream_node'])
PipelineOutput = namedtuple("PipelineOutput", ['output_name', 'upstream_slot', 'upstream_node'])


class NodeConnection(EdgeBase):

    def __init__(self, upstream, downstream, upstream_slot: str, downstream_slot: str):
        """
        :param upstream:
        :type upstream: NodeBase
        :param downstream:
        :type downstream: NodeBase
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
    def validate(node_con):
        assert_subclass(node_con, NodeConnection)

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


class NodeBaseMeta(type):

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
        if 'transform' not in attrs:
            raise DaskPipesException("Class {} does not have transform method".format(name))

        return super().__new__(mcs, name, bases, attrs)


class NodeBase(VertexBase, BaseEstimator, metaclass=NodeBaseMeta):

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name

    @property
    def outputs(self) -> List[ReturnDescription]:
        """
        Outputs of an node (parses annotation of .transform function)

        Since this is defined in meta, we can get node outputs just based on it's class
        :return: List of named tuples 'ReturnDescription' with argument name, type and description
        """
        ret_descr = get_return_description(getattr(self, 'transform'))
        return ret_descr

    @property
    def inputs(self) -> List[ArgumentDescription]:
        """
        Inputs of an node - parameters of self.fit function
        must be equal to parameters of self.transform

        Since this is defined in meta, we can get node outputs just based on it's class
        :return:
        """
        arg_descr = get_arguments_description(getattr(self, 'fit'))
        return arg_descr

    @staticmethod
    def validate(node):
        """
        Used by Graph to validate vertices when adding them
        Checks that node is valid NodeBase
            > check that node is subclass of NodeBase
        :param node: Any
        :return:
        """
        assert_subclass(node, NodeBase)

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        """
        Make other Node upstream of current
        Pipes output of specific slot of upstream node to specific slot of current node
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and nodes
        >>> p = Pipeline()
        >>> op1 = ExampleNode('op1')
        >>> op2 = ExampleNode('op2')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>>
        >>> # Set upstream node, add op2 to pipeline
        >>> # input ->> op1 ->> op2
        >>> op2.set_upstream(op1)

        :param other: Node to set upstream
        :type other: NodeBase
        :param upstream_slot: upstream's output (name of one of other.outputs) or None
        if None is passed and upstream has only one output, automatically deducts upstream_slot
        :param downstream_slot: this input slot (name of one of self.inputs) or None
        if None is passed and this node has only one input, automatically deducts upstream_slot
        :return: None
        """
        super().set_upstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def set_downstream(self, other,
                       upstream_slot: Optional[str] = None,
                       downstream_slot: Optional[str] = None, ):
        """
        Make other Node upstream of current
        Pipes output of specific slot of upstream node to specific slot of current node
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and nodes
        >>> p = Pipeline()
        >>> op1 = ExampleNode('op1')
        >>> op2 = ExampleNode('op2')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>>
        >>> # Set downstream node, add op2 to pipeline
        >>> # input ->> op1 ->> op2
        >>> op1.set_downstream(op2)

        :param other: Node to set downstream
        :type other: NodeBase
        :param upstream_slot: this node output (name of one of self.outputs) or None
        if None is passed and this node has only one output, automatically deducts upstream_slot
        :param downstream_slot: downstream's input slot (name of one of other.inputs) or None
        if None is passed and downstream node has only one input, automatically deducts upstream_slot
        :return: None
        """
        super().set_downstream(other, upstream_slot=upstream_slot, downstream_slot=downstream_slot)

    def fit(self, *args, **kwargs):
        """
        Infer parameters prior to transforming dataset
        To be implemented by subclass pipelines

        Signature of this function is changed dynamically.
        As user sets pipeline input nodes, they are added to parameters of fit

        Important!
        Reason behind returning dict instead of storing fitted data inside node:
        1. One node can be fitted multiple time in one pipeline
        2. Do not use pickle to save model parameters. instead, serialize them explicitly to yaml or csv files

        # Example:
        >>> import pandas as pd
        >>> ds = pd.DataFrame([[1,2,3]])
        >>> p = Pipeline()
        >>> op1 = ExampleNode('op1')
        >>> op2 = ExampleNode('op2')
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

    def transform(self, *args, **kwargs):
        """
        To be implemented by subclass pipelines

        Signature of this function is changed dynamically.
        As user sets pipeline input nodes, they are added to parameters of transform

        # Example
        >>> import pandas as pd
        >>> ds = pd.DataFrame([[1,2,3]])
        >>> p = Pipeline()
        >>> op1 = ExampleNode('op1')
        >>> op2 = ExampleNode('op2')
        >>> p.set_input(op1)
        >>> op1.set_downstream(op2)
        >>> op1.fit(ds)
        >>>
        >>> print(p.outputs) # ['op3_result'] 'result' is because transform has not defined result annotations
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
        name of argument is equal to input nodes' input slots + optional suffix
        single key-word argument can be piped to multiple nodes
        :return: Node outputs that were not piped anywhere
        """
        raise NotImplementedError()

    def __repr__(self):
        if self.name is None:
            return '<Unnamed {} at {}>'.format(self.__class__.__name__, hex(id(self)))

        return '<{}: {}>'.format(self.__class__.__name__, self.name)


class ExampleNode(NodeBase):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.params = None

    def fit(self, dataset):
        """
        Example docstring
        :param dataset: dataset to fit to
        :return: Parameters passed to .transform
        """
        self.params = {'a': 1}

    def transform(self, dataset):
        """
        Example docstring
        :param dataset: dataset to transform
        :return:
        """
        assert dataset is not None
        assert self.params['a'] == 1
        return dataset


class Pipeline(Graph):
    """
    Pipeline is a graph structure, containing relationships between NodeBase (vertices)
    with NodeConnection as edges

    Defines fit and transform methods which iterate vertices in width-first order,
    calling fit() and transform() methods of nodes and
    piping outputs of upstream nodes to inputs of downstream nodes
    """

    def __init__(self):
        super().__init__()

        # Pipeline inputs. use set_input to add input
        self._inputs: List[PipelineInput] = list()

        self.node_dict = dict()

        # For naming unnamed nodes
        self.node_name_counter = 0

    def add_vertex(self, node: NodeBase, vertex_id=None):
        """
        Add node to current pipeline
        :param node: node to add
        :param vertex_id: node's id or None (if None - autoincrement) (used for deserialization from disk)
        :type vertex_id: int
        :return:
        """
        if node.name is None:
            while '{}_unnamed{}'.format(node.__class__.__name__, self.node_name_counter) in self.node_dict:
                self.node_name_counter += 1
            node.name = '{}_unnamed{}'.format(node.__class__.__name__, self.node_name_counter)
            self.node_name_counter += 1
        if node.name in self.node_dict:
            if self.node_dict[node.name] is not node:
                raise DaskPipesException("Duplicate name for node {}".format(node))
        self.node_dict[node.name] = node
        super().add_vertex(node, vertex_id=vertex_id)

    def remove_vertex(self, node: NodeBase) -> VertexBase:
        """
        Remove node from pipeline
        :param node: node to remove (must be in current pipeline)
        :return:
        """
        if node.name is None:
            raise DaskPipesException("Node is unnamed")
        if node.graph is not self:
            raise DaskPipesException("Node does not belong to pipeline")
        if node.name in self.node_dict:
            if self.node_dict[node.name] is not node:
                raise ValueError("Node does not equal to node_dict['{}']".format(node.name))
            del self.node_dict[node.name]
        else:
            raise ValueError("Invalid node name")
        self.remove_input(node)
        return super().remove_vertex(node)

    @property
    def _outputs(self) -> List[PipelineOutput]:
        """
        Get detailed info about pipeline outputs
        :return: list out named tuples defining unused nodes' outputs. output of .transform() method
        """
        outputs = []

        for v_id, vertex in self._vertices.items():

            vertex: NodeBase

            v_outputs = {i.name for i in vertex.outputs}

            for edge in self.get_downstream_edges(vertex):
                edge: NodeConnection
                if edge.upstream_slot in v_outputs:
                    v_outputs.remove(edge.upstream_slot)
                if len(v_outputs) == 0:
                    break

            for v_out in v_outputs:
                out = '{}_{}'.format(vertex.name, v_out)
                outputs.append(PipelineOutput(output_name=out, upstream_slot=v_out, upstream_node=vertex))

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
        Get human-friendly list of output names of .transform() function without detailed info about nodes
        :return: list of nodes's outputs that have no downstream nodes. Transform output
        """
        return [o.output_name for o in self._outputs]

    def validate_edge(self, edge):
        """
        Used by base graph class to check if edge can be added to current graph
        Checks if edge is subclass of NodeConnection
        :param edge:
        :return:
        """
        NodeConnection.validate(edge)

    def validate_vertex(self, vertex):
        """
        Used by base graph to check if vertex can be added to current graph
        Checks if vertex is subclass of NodeBase
        :param vertex:
        :return:
        """
        NodeBase.validate(vertex)

    def set_input(self, node: NodeBase, suffix=None):
        """
        Register node as pipeline's input.
        All node inputs should be passed in fit function
        by default, uses fit argument names equal to node input arguments.

        Caution:
        If multiple nodes which have same input names are set as input,
        pipeline will pass data from argument to multiple nodes
        suffix can be used to differentiate inputs for nodes with same argument names

        Example:
        >>> # Construct pipeline and nodes
        >>> p = Pipeline()
        >>> op1 = ExampleNode('op1')
        >>>
        >>> # Assign input for op1 as pipeline input and add it to pipeline
        >>> p.set_input(op1)
        >>> # p.fit signature changed
        >>> # also, p.inputs now returns ['dataset']

        :param node: node to set as input
        :param suffix: suffix to add to node's inputs before setting them as pipeline inputs
        :return:
        """
        self.validate_vertex(node)

        if suffix is None:
            suffix = ''

        node_inputs = node.inputs
        if len(node_inputs) == 0:
            raise DaskPipesException("{} does not have any inputs.".format(node))
        for op_input in node_inputs:
            downstream_slot = op_input.name

            if downstream_slot is None:
                node_inputs = node.inputs
                if len(node_inputs) > 1:
                    raise DaskPipesException(
                        "{} has multiple inputs, "
                        "downstream_slot must be one of {}".format(node, [i.name for i in node_inputs]))
                downstream_slot = node_inputs[0].name

            arg_name = '{}{}'.format(downstream_slot, suffix)

            self._inputs.append(PipelineInput(arg_name=arg_name,
                                              downstream_slot=downstream_slot,
                                              downstream_node=node))
        node.graph = self

        self._update_fit_transform_signatures()

    def remove_input(self, node: NodeBase):
        """
        Unset node as pipeline output
        :param node: node to unset as output
        :return:
        """
        self.validate_vertex(node)

        # Find inputs to remove
        self._inputs = [i for i in self._inputs if i.downstream_node is not node]
        self._update_fit_transform_signatures()

    def _set_fit_signature(self, sign: inspect.Signature):
        self.fit = MethodType(replace_signature(Pipeline.fit, sign), self)

    def _set_transform_signature(self, sign: inspect.Signature):
        self.transform = MethodType(replace_signature(Pipeline.transform, sign), self)

    def _reset_fit_signature(self):
        self.fit = MethodType(self.__class__.fit, self)

    def _reset_transform_signature(self):
        self.transform = MethodType(self.__class__.transform, self)

    def _update_fit_transform_signatures(self):
        """
        Called when input node is added or removed.
        Infers and updates signatures of fit and transform methods for user-friendly hints
        :return:
        """
        fit_func = getattr(self.fit, '__func__')  # Since fit is method, it has __func__
        transform_func = getattr(self.transform, '__func__')  # Since transform is method, it has __func__
        fit_sign = inspect.signature(fit_func)
        transform_sign = inspect.signature(transform_func)

        original_prams = list(fit_sign.parameters.values())

        new_params_pos_only = [i for i in original_prams if i.kind == inspect.Parameter.POSITIONAL_ONLY]
        new_params_pos = [i for i in original_prams if
                          i.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                          i.default == inspect._empty]
        new_params_kwargs = [i for i in original_prams if
                             i.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                             i.default != inspect._empty]
        new_params_kwargs_only = [i for i in original_prams if i.kind == inspect.Parameter.KEYWORD_ONLY]
        seen_params = set()

        if len(self._inputs) > 0:
            for inp in self._inputs:
                if inp.arg_name in seen_params:
                    continue
                param = inspect.Parameter(
                    name=inp.arg_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                    if len(self._inputs) == 1 else inspect.Parameter.KEYWORD_ONLY,
                    default=None)

                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default == inspect._empty:
                    new_params_pos.append(param)
                elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    new_params_kwargs.append(param)
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    new_params_kwargs_only.append(param)
                else:
                    raise TypeError("Invalid parameter king")

                seen_params.add(inp.arg_name)
            new_params = new_params_pos_only + new_params_pos + new_params_kwargs + new_params_kwargs_only

            self._set_fit_signature(inspect.Signature(
                parameters=new_params,
                return_annotation=fit_sign.return_annotation))

            self._set_transform_signature(inspect.Signature(
                parameters=new_params,
                return_annotation=transform_sign.return_annotation))
        else:
            self._reset_fit_signature()
            self._reset_transform_signature()

    def connect(self,
                upstream: NodeBase,
                downstream: NodeBase,
                upstream_slot: Optional[str] = None,
                downstream_slot: Optional[str] = None) -> NodeConnection:
        """
        General method for connecting tto nodes
        Creates directed connection with upstream and downstream slots

        if connection between slots was already made, raises DaskPipesException
        if downstream_slot already has piped input, raises DaskPipesException,
        since multiple inputs for one slot are not allowed

        :param upstream: node to set upstream
        :param downstream: node to set downstream
        :param upstream_slot: slot of upstream node to pass to downstream
        :param downstream_slot: slot of downstream node to receive data into
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
                    "{} has multiple inputs, cannot infer downstream_slot. "
                    "Please, provide downstream_slot as one of {}".format(
                        downstream,
                        [i.name for i in downstream_inputs]))
            downstream_slot = downstream_inputs[0].name

        # Create edge
        edge = NodeConnection(upstream=upstream,
                              downstream=downstream,
                              upstream_slot=upstream_slot,
                              downstream_slot=downstream_slot)
        return self.add_edge(edge)

    # Change signature of functions
    def add_edge(self, op_connection: NodeConnection, edge_id=None) -> NodeConnection:
        """
        Add NodeConnection to current pipeline
        op_connection must already have assigned upstream and downstream nodes
        :param op_connection: connection to add
        :param edge_id: integer id of edge, auto-increment if None (used for deserialization from disk)
        :return:
        """
        super().add_edge(op_connection, edge_id=edge_id)
        return op_connection

    def _parse_arguments(self, *args, **kwargs) -> Dict[str, Any]:

        expected_inputs = {i.arg_name for i in self._inputs}
        if len(expected_inputs) == 0 and (len(args) > 0 or len(kwargs) > 0):
            raise DaskPipesException(
                "{} Does not have any inputs. (used .set_input(op) to set nodes as input)".format(self))

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

        return rv

    @staticmethod
    def _parse_node_output(node, output):
        """
        Get dictionary of key-values from return of function

        Converts list, tuple or dict to key-value pairs
        makes sanity checks

        :param node: class of node to read transform return annotation from
        :param output: return value of .transform() method
        :return:
        """
        expected_result: List[ReturnDescription] = node.outputs
        expected_keys = [i.name for i in expected_result]
        if not isinstance(output, list) and not isinstance(output, tuple) and not isinstance(output, dict):
            output = (output,)

        if isinstance(output, list) or isinstance(output, tuple):  # In case returned list
            # Check that output length matches
            if len(output) != len(expected_result):
                raise DaskPipesException(
                    "{}.transform result length does not match expected. Expected {}, got {}".format(
                        node,
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
                        node,
                        expected_keys, got_keys
                    ))
        else:
            raise DaskPipesException("Unknown return type: {}".format(output.__class__.__name__))

        return output

    def _check_arguements(self, op, edge, node_arguments, downstream_op, op_result):
        if edge.downstream_slot in node_arguments[downstream_op]:
            raise DaskPipesException(
                "Duplicate argument for {}['{}']. "
                "Already contains {}".format(downstream_op, edge.downstream_slot,
                                             node_arguments[downstream_op][edge.downstream_slot]))
        if downstream_op not in node_arguments:
            raise DaskPipesException("Pipeline does not have {}".format(downstream_op))
        if edge.upstream_slot not in op_result:
            if len(op_result) == 0:
                raise DaskPipesException("Node {} did not return anything!".format(op))
            raise DaskPipesException(
                "Node {} did not return expected {}; "
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
        node_arguments = {op: dict() for op in self.vertices}
        for inp in self._inputs:
            node_arguments[inp.downstream_node][inp.downstream_slot] = args.get(inp.arg_name, None)

        outputs = dict()

        for op in VertexWidthFirst(self):
            op: NodeBase
            op_args = node_arguments[op]

            if len(op_args) == 0:
                raise DaskPipesException("No input for node {}".format(op))

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
                downstream_op: NodeBase = edge.downstream
                edge: NodeConnection
                self._check_arguements(op, edge, node_arguments, downstream_op, op_result)
                if edge.upstream_slot in unused_output:
                    unused_output.remove(edge.upstream_slot)
                node_arguments[downstream_op][edge.downstream_slot] = op_result[edge.upstream_slot]

            for upstream_slot in unused_output:
                outputs[(op, upstream_slot)] = op_result[upstream_slot]

        return outputs

    def _fit(self, op, **op_args):
        """
        Helper function used for fit call and dumping params
        :param op:
        :param op_args:
        :return:
        """
        # ----------------------
        # Fit node
        # ----------------------
        # Make copy of arguments
        op_args = {k: v.copy() if v is not None else v for k, v in op_args.items()}
        op.fit(**op_args)

    def _transform(self, op, **op_args):
        """
        Helper function, used for transform call and dumping outputs
        :param op:
        :param op_args:
        :return:
        """

        # ----------------------
        # Transform node
        # ----------------------
        # Make copy of arguments
        op_args = {k: v.copy() if v is not None else v for k, v in op_args.items()}

        op_result = Pipeline._parse_node_output(op, op.transform(**op_args))
        if len(op_result) == 0:
            raise DaskPipesException("Node {} did not return anything".format(op))

        return op_result

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: self
        """

        def func(op, has_downstream, **op_args):
            self._fit(op, **op_args)
            if has_downstream:
                rv = self._transform(op, **op_args)
                return rv

        self._iterate_graph(func, *args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        """
        Method for transforming based on previously fitted parameters
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: dictionary of datasets, or single dataset.
        Output of nodes that was not piped anywhere
        """

        def func(op, has_downstream, **op_args):
            return self._transform(op, **op_args)

        outputs = self._iterate_graph(func, *args, **kwargs)
        rv = {'{}_{}'.format(k[0].name, k[1]): v for k, v in outputs.items()}
        # Return just dataframe if len 1
        if len(rv) == 1:
            return next(iter(rv.values()))
        return rv
