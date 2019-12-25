from .graph import Graph, VertexBase, EdgeBase, VertexWidthFirst
from ..exceptions import DaskPipesException
from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from .cache import dataframe_loader_factory, dataframe_dumper_factory
import inspect
from ..utils import (get_arguments_description,
                     get_return_description,
                     ReturnDescription,
                     assert_subclass)
from ._pipeline_utils import (PipelineInput,
                              PipelineOutput,
                              get_input_signature,
                              set_fit_signature,
                              set_transform_signature,
                              reset_transform_signature,
                              reset_fit_signature,
                              getcallargs_inverse,
                              validate_fit_transform)
from copy import deepcopy
import yaml
from yaml.constructor import ConstructorError

__all__ = ['NodeConnection', 'NodeBase', 'Pipeline', 'NodeBaseMeta', 'DummyNode', 'CacheMixin']


class NodeSlot:
    def __init__(self, node, slot: str):
        if not hasattr(node, 'set_downstream') or not hasattr(node, 'set_upstream'):
            raise DaskPipesException("node {} must implement set_downstream, set_upstream".format(node))

        self.node = node
        self.slot = slot

    def __rshift__(self, other):
        """
        self >> other
        :param other:
        :return:
        """
        if isinstance(other, NodeSlot):
            self.node.set_downstream(other.node, upstream_slot=self.slot, downstream_slot=other.slot)
            return other.node
        elif isinstance(other, NodeBase):
            self.node.set_downstream(other, upstream_slot=self.slot)
            return other
        elif isinstance(other, Pipeline):
            raise NotImplementedError()

    def __lshift__(self, other):
        """
        self << other
        :param other:
        :return:
        """
        if isinstance(other, NodeSlot):
            self.node.set_upstream(other.node, upstream_slot=other.slot, downstream_slot=self.slot)
            return other.node
        elif isinstance(other, NodeBase):
            self.node.set_upstream(other, downstream_slot=self.slot)
            return other
        elif isinstance(other, Pipeline):
            raise NotImplementedError()


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

        fit_wrapped.__func__ = func
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
        validate_fit_transform(name, attrs)
        return super().__new__(mcs, name, bases, attrs)


class NodeBase(VertexBase, BaseEstimator, TransformerMixin, metaclass=NodeBaseMeta):

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name

    def __getitem__(self, slot):
        available_slots = sorted({i.name for i in self.inputs}.union({i.name for i in self.outputs}))

        if slot not in available_slots:
            raise DaskPipesException(
                "{} Does not have input or output slot {}. Use one of {}".format(
                    self, slot, available_slots))
        return NodeSlot(self, slot)

    def __rshift__(self, other):
        """
        self >> other
        :param other:
        :return:
        """
        if isinstance(other, NodeSlot):
            self.set_downstream(other.node, downstream_slot=other.slot)
            return other.node
        elif isinstance(other, NodeBase):
            self.set_downstream(other)
            return other
        elif isinstance(other, Pipeline):
            raise NotImplementedError()

    def __lshift__(self, other):
        """
        self << other
        :param other:
        :return:
        """
        if isinstance(other, NodeSlot):
            self.set_upstream(other.node, upstream_slot=other.slot)
            return other.node
        elif isinstance(other, NodeBase):
            self.set_upstream(other)
            return other
        elif isinstance(other, Pipeline):
            raise NotImplementedError()

    @property
    def outputs(self) -> List[ReturnDescription]:
        """
        Outputs of an node (parses annotation of .transform function)

        Since this is defined in meta, we can get node outputs just based on it's class
        :return: List of named tuples 'ReturnDescription' with argument name, type and description
        """
        return get_return_description(getattr(self, 'transform'))

    @property
    def inputs(self) -> List[inspect.Parameter]:
        """
        Inputs of an node - parameters of self.fit function
        must be equal to parameters of self.transform

        Since this is defined in meta, we can get node outputs just based on it's class
        :return:
        """
        return get_arguments_description(getattr(self, 'fit'))

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
        >>> op1 = DummyNode('op1')
        >>> op2 = DummyNode('op2')
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
        >>> op1 = DummyNode('op1')
        >>> op2 = DummyNode('op2')
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
        >>> op1 = DummyNode('op1')
        >>> op2 = DummyNode('op2')
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
        >>> op1 = DummyNode('op1')
        >>> op2 = DummyNode('op2')
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

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def __repr__(self):
        if self.name is None:
            return '<Unnamed {} at {}>'.format(self.__class__.__name__, hex(id(self)))

        return '<{}: {}>'.format(self.__class__.__name__, self.name)


class DummyNode(NodeBase):

    def __init__(self, name=None):
        super().__init__(name=name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class PipelineMeta(type):
    def __new__(mcs, name, bases, attrs):
        """
        Run validations on pipeline's fit and transform signatures
        """
        validate_fit_transform(name, attrs, obligatory_variadic=True, allow_default=False)
        return super().__new__(mcs, name, bases, attrs)


class NodeCallable:
    def __call__(self, node: NodeBase, node_input: Tuple[Tuple[Any], Dict[str, Any]],
                 has_downstream: bool = True) -> Any: ...


class PipelineMixin:

    def fit(self,
            func: NodeCallable,
            node: NodeBase,
            node_input: Tuple[Tuple[Any], Dict[str, Any]],
            has_downstream=True):
        return func(node, node_input, has_downstream=has_downstream)

    def transform(self,
                  func: NodeCallable,
                  node: NodeBase,
                  node_input: Tuple[Tuple[Any], Dict[str, Any]],
                  has_downstream=True):
        return func(node, node_input, has_downstream=has_downstream)

    def wrap_fit(self, fit):
        def func(node, node_input, has_downstream=True):
            return self.fit(fit, node, node_input, has_downstream=has_downstream)

        return func

    def wrap_transform(self, transform):
        def func(node, node_input, has_downstream=True):
            return self.transform(transform, node, node_input, has_downstream=has_downstream)

        return func


NodeCache = namedtuple('NodeCache', ['node_input', 'node_output'])


class CacheMixin(PipelineMixin):

    def __init__(self, cache_dir):
        self.cache: Dict[str, NodeCache] = dict()
        self.cache_dir = cache_dir
        self.dump_cache = dict()
        self.load_cache = dict()

    def _record_cache(self, node, node_input, node_output):
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node.name,
            df_dump_cache=self.dump_cache,
            df_load_cache=self.load_cache,
        )
        node_input = yaml.dump(node_input, Dumper=dumper)
        node_output = yaml.dump(node_output, Dumper=dumper)
        self.cache[node.name] = NodeCache(
            node_input=node_input,
            node_output=node_output
        )

    def _load_result(self, node, node_input):
        if node.name not in self.cache:
            return None
        node_cache = self.cache[node.name]
        loader = dataframe_loader_factory(
            df_dump_cache=self.dump_cache,
            df_load_cache=self.load_cache,
        )
        dumper = dataframe_dumper_factory(
            self.cache_dir,
            node_name=node.name,
            df_dump_cache=self.dump_cache,
            df_load_cache=self.load_cache,
        )
        orig_input = yaml.dump(node_input, Dumper=dumper)
        if node_cache.node_input == orig_input:
            return yaml.load(node_cache.node_output, Loader=loader)
        return None

    def fit(self,
            func: NodeCallable,
            node: NodeBase,
            node_input: Tuple[Tuple[Any], Dict[str, Any]],
            has_downstream=True):
        node_output = self._load_result(node, node_input)
        if node_output is None:
            node_output = func(node, node_input, has_downstream=has_downstream)
            if has_downstream:
                self._record_cache(node, node_input, node_output)
                node_output = self._load_result(node, node_input)
                assert node_output is not None
        return node_output

    def transform(self,
                  func: NodeCallable,
                  node: NodeBase,
                  node_input: Tuple[Tuple[Any], Dict[str, Any]],
                  has_downstream=True):
        node_output = self._load_result(node, node_input)
        if node_output is None:
            node_output = func(node, node_input, has_downstream=has_downstream)
            self._record_cache(node, node_input, node_output)
            node_output = self._load_result(node, node_input)
            assert node_output is not None
        return node_output


class Pipeline(Graph, metaclass=PipelineMeta):
    """
    Pipeline is a graph structure, containing relationships between NodeBase (vertices)
    with NodeConnection as edges

    Defines fit and transform methods which iterate vertices in width-first order,
    calling fit() and transform() methods of nodes and
    piping outputs of upstream nodes to inputs of downstream nodes
    """

    def __init__(self, mixins=None):
        super().__init__()

        # Pipeline inputs. use set_input to add input
        self._inputs: List[PipelineInput] = list()

        self.node_dict = dict()

        # For naming unnamed nodes
        self.node_name_counter = 0

        self._params_downstream = None

        if mixins is None:
            self.mixins = list()
        else:
            self.mixins = mixins

    @staticmethod
    def _get_default_node_name(node, counter=None):
        if counter is None or counter == 0:
            return '{}'.format(node.__class__.__name__.lower())
        return '{}{}'.format(node.__class__.__name__.lower(), counter)

    def __rshift__(self, other):
        """
        self >> other
        :param other:
        :return:
        """
        if isinstance(other, NodeSlot):
            self.set_input(other.node, name=other.slot, downstream_slot=other.slot)
            return other.node
        elif issubclass(other.__class__, NodeBase):
            self.set_input(other)
            return other
        else:
            raise NotImplementedError(other.__class__.__name__)

    def __lshift__(self, other):
        """
        self << other
        :param other:
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, slot):
        """
        Pipeline['slot']
        :param slot:
        :return:
        """
        return NodeSlot(self, slot)

    def add_vertex(self, node: NodeBase, vertex_id=None):
        """
        Add node to current pipeline
        :param node: node to add
        :param vertex_id: node's id or None (if None - autoincrement) (used for deserialization from disk)
        :type vertex_id: int
        :return:
        """
        if node.name is None:
            while Pipeline._get_default_node_name(node, self.node_name_counter) in self.node_dict:
                self.node_name_counter += 1
            node.name = Pipeline._get_default_node_name(node, self.node_name_counter)
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
    def inputs(self) -> List[inspect.Parameter]:
        """
        Get human-friendly list of names of current pipeline inputs without duplicates
        :return: list of argument names for fit function
        """
        return get_input_signature(self)[0][1:]

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

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        """
        Set pipeline output
        :param other:
        :param upstream_slot:
        :param downstream_slot:
        :return:
        """
        raise NotImplementedError()

    def set_downstream(self, other,
                       upstream_slot: Optional[str] = None,
                       downstream_slot: Optional[str] = None, ):
        """
        Set pipeline input
        :param other:
        :param upstream_slot:
        :param downstream_slot:
        :return:
        """
        self.set_input(other, name=upstream_slot, downstream_slot=downstream_slot)

    def remove_input(self, name):
        len_before = len(self._inputs)
        if len_before == 0:
            raise DaskPipesException("{} Does not have any arguments".format(self))
        self._inputs = [i for i in self._inputs if i.name == name]
        if len(self._inputs) == len_before:
            raise DaskPipesException("{} Does not have argument {}".format(self, name))
        self._update_fit_transform_signatures()

    def remove_input_node(self, node: NodeBase):
        """
        Unset node as pipeline output
        :param node: node to unset as output
        :return:
        """
        self.validate_vertex(node)
        # Find inputs to remove
        self._inputs = [i for i in self._inputs if i.downstream_node is not node]
        self._update_fit_transform_signatures()

    def set_input(self, node: NodeBase, name=None, downstream_slot=None, suffix: Optional[str] = None):  # noqa: C901
        self.validate_vertex(node)
        # Assign node to current pipeline
        node.graph = self

        node_inputs = {i.name: i for i in node.inputs}
        if len(node_inputs) == 0:
            raise DaskPipesException("{} does not have any inputs")

        if downstream_slot is None:
            if name is not None and len(node_inputs) > 1:
                raise DaskPipesException(
                    "Node {} has multiple inputs, specific "
                    "pipeline argument name not supported. "
                    "Use suffix instead")
            for inp in node_inputs.keys():
                self.set_input(node, name, inp, suffix=suffix)
            return

        if downstream_slot not in node_inputs:
            raise DaskPipesException(
                "{} does not have input {}; available: {}".format(node, downstream_slot, list(node_inputs.keys())))

        if name is None:
            name = downstream_slot
            if suffix is None:
                suffix = '_{}'.format(node.name)

        if suffix is not None:
            name = '{}{}'.format(name, suffix)

        current_args = [i for i in self._inputs if i.name == name]
        if len(current_args) > 0:
            for existing_arg in current_args:
                if existing_arg.downstream_slot == downstream_slot and existing_arg.downstream_node is node:
                    return

        param = node_inputs[downstream_slot]

        self._inputs.append(PipelineInput(name=name,
                                          downstream_slot=downstream_slot,
                                          downstream_node=node,
                                          default=param.default,
                                          kind=param.kind,
                                          annotation=param.annotation))
        self._update_fit_transform_signatures()

    def _update_fit_transform_signatures(self):
        """
        Called when input node is added or removed.
        Infers and updates signatures of fit and transform methods for user-friendly hints
        :return:
        """
        if len(self._inputs) > 0:
            new_params, params_downstream = get_input_signature(self)
            self._params_downstream = params_downstream
            return_annotation = self.__class__.fit.__annotations__.get('return', inspect._empty)

            set_fit_signature(self, inspect.Signature(
                parameters=new_params,
                return_annotation=return_annotation))

            set_transform_signature(self, inspect.Signature(
                parameters=new_params,
                return_annotation=return_annotation))
        else:
            self._params_downstream = None
            reset_fit_signature(self)
            reset_transform_signature(self)

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
            slots_no_default = [i for i in downstream_inputs if i.default == inspect._empty]
            if len(slots_no_default) > 1:
                raise DaskPipesException(
                    "{} has multiple inputs, cannot infer downstream_slot. "
                    "Please, provide downstream_slot as one of {}".format(
                        downstream,
                        [i.name for i in downstream_inputs]))
            downstream_slot = slots_no_default[0].name

        # Create edge
        edge = NodeConnection(upstream=upstream,
                              downstream=downstream,
                              upstream_slot=upstream_slot,
                              downstream_slot=downstream_slot)
        return self.add_edge(edge)

    # Change signature of functions
    def add_edge(self, node_connection: NodeConnection, edge_id=None) -> NodeConnection:
        """
        Add NodeConnection to current pipeline
        node_connection must already have assigned upstream and downstream nodes
        :param node_connection: connection to add
        :param edge_id: integer id of edge, auto-increment if None (used for deserialization from disk)
        :return:
        """
        super().add_edge(node_connection, edge_id=edge_id)
        return node_connection

    def _parse_arguments(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Parse fit arguments and return dictionary of values.
        If argument has a default value and not provided,
        result dictionary will contain default value for that argument
        :param args: fit arguments
        :param kwargs: fit key-word arguments
        :return:
        """

        # Used for injection of custom signature before passing to getcallargs
        # We need to distill self.fit from parameters passed to pipeline against parameters passed to input nodes
        def fit(*args, **kwargs):
            pass

        fit_params = [param for param in self.inputs if param.name in self._params_downstream]
        fit.__signature__ = inspect.Signature(
            parameters=fit_params
        )

        var_pos = {param.name for param in fit_params if param.kind == inspect.Parameter.VAR_POSITIONAL}
        var_key = {param.name for param in fit_params if param.kind == inspect.Parameter.VAR_KEYWORD}
        rv = inspect.getcallargs(fit, *args, **kwargs)
        # Replace variadic with dummy asterisk
        # Doing this, because one variadic can have different names
        return {(k if k not in var_pos else '*') if k not in var_key else '**': v for k, v in rv.items()}

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

    def _check_arguements(self, node, edge, node_arguments, downstream_node, node_result):
        if downstream_node not in node_arguments:
            raise DaskPipesException("Pipeline does not contain {}".format(downstream_node))
        if edge.upstream_slot not in node_result:
            if len(node_result) == 0:
                raise DaskPipesException("Node {} did not return anything!".format(node))
            raise DaskPipesException(
                "Node {} did not return expected {}; "
                "recieved {}".format(node, edge.upstream_slot, list(node_result.keys())))

    def _iterate_graph(self, func, *args, **kwargs):  # noqa: C901
        """
        Helper function used in fit and transform methods
        :param func: function to apply during iteration
        :param args: pipeline positional input
        :param kwargs: pipeline key-word input
        :return:
        """
        args = self._parse_arguments(*args, **kwargs)
        node_arguments = {node: dict() for node in self.vertices}
        for inp in self._inputs:
            # We want to be extremely careful on each step
            # Even though, _parse_arguments must return full proper dictionary, filled with default values
            # Double-check it here
            if inp.kind == inspect.Parameter.VAR_KEYWORD:
                inp_name = '**'
            elif inp.kind == inspect.Parameter.VAR_POSITIONAL:
                inp_name = '*'
            else:
                inp_name = inp.name
            if inp_name in args:
                node_arguments[inp.downstream_node][inp.downstream_slot] = args[inp_name]
            else:
                raise DaskPipesException(
                    "Pipeline input {}->{}['{}'] not provided "
                    "and does not have default value".format(inp.name,
                                                             inp.downstream_node,
                                                             inp.downstream_slot, ))
        outputs = dict()

        for node in VertexWidthFirst(self):
            try:
                node: NodeBase
                node_callargs = node_arguments[node]

                if len(node_callargs) == 0:
                    raise DaskPipesException("No input for node {}".format(node))

                downstream_edges = self.get_downstream_edges(node)
                has_downstream = len(downstream_edges) > 0

                node_input = getcallargs_inverse(node.fit, **node_callargs)
                node_result = func(node, node_input, has_downstream=has_downstream)

                if node_result is None:
                    continue

                if not isinstance(node_result, dict):
                    raise DaskPipesException(
                        "Invalid return. Expected {}, received {}".format(dict.__name__,
                                                                          node_result.__class__.__name__))

                unused_output = set(node_result.keys())

                # Get downstream edges
                for edge in downstream_edges:
                    downstream_node: NodeBase = edge.downstream
                    edge: NodeConnection
                    self._check_arguements(node, edge, node_arguments, downstream_node, node_result)
                    if edge.upstream_slot in unused_output:
                        unused_output.remove(edge.upstream_slot)

                    # Populate arguments of downstream vertex
                    downstream_param = next((i for i in downstream_node.inputs if i.name == edge.downstream_slot))

                    # Properly handle variadic return arguments
                    if edge.upstream_slot[0] == '*' or edge.upstream_slot[:2] == '**':
                        upstream_val = node_result[edge.upstream_slot]
                    else:
                        if downstream_param.kind == inspect.Parameter.VAR_POSITIONAL:
                            upstream_val = (node_result[edge.upstream_slot],)
                        elif downstream_param.kind == inspect.Parameter.VAR_KEYWORD:
                            upstream_val = {edge.upstream_slot: node_result[edge.upstream_slot]}
                        else:
                            upstream_val = node_result[edge.upstream_slot]

                    downstream_args = node_arguments[downstream_node]
                    if downstream_param.kind == inspect.Parameter.VAR_POSITIONAL:
                        if edge.downstream_slot not in downstream_args:
                            downstream_args[edge.downstream_slot] = list()
                        try:
                            downstream_args[edge.downstream_slot].extend(upstream_val)
                        except TypeError:
                            raise DaskPipesException(
                                "{} returned non-iterable "
                                "as variadic '{}'. "
                                "Expected tuple, received {}".format(node, edge.upstream_slot,
                                                                     repr(upstream_val))) from None
                    elif downstream_param.kind == inspect.Parameter.VAR_KEYWORD:
                        if edge.downstream_slot not in downstream_args:
                            downstream_args[edge.downstream_slot] = dict()
                        downstream_dict = downstream_args[edge.downstream_slot]
                        try:
                            for k, v in upstream_val.items():
                                if k in downstream_dict:
                                    raise DaskPipesException(
                                        "Duplicate key-word argument "
                                        "'{}' for parameter '{}'".format(k, edge.downstream_slot))
                                downstream_dict[k] = v
                        except AttributeError:
                            raise DaskPipesException(
                                "{} returned non-mapping "
                                "as variadic '{}'. "
                                "Expected dict; received {}".format(node, edge.upstream_slot,
                                                                    repr(upstream_val))) from None
                    else:
                        if edge.downstream_slot in downstream_args:
                            raise DaskPipesException(
                                "Duplicate argument for parameter '{}'".format(edge.downstream_slot))
                        downstream_args[edge.downstream_slot] = upstream_val

                for upstream_slot in unused_output:
                    outputs[(node, upstream_slot)] = node_result[upstream_slot]
            except Exception as ex:
                raise DaskPipesException("Error occurred during {}".format(node)) from ex

        return outputs

    def pre_fit(self, node, node_input):
        pass

    def pre_transform(self, node, node_input):
        pass

    def post_fit(self, node, node_input):
        pass

    def post_transform(self, node, node_input, node_output):
        pass

    def _fit(self, node: NodeBase, node_input):
        """
        Helper function used for fit call and dumping params
        :param node:
        :param node_input:
        :return:
        """
        # ----------------------
        # Fit node
        # ----------------------
        # Make copy of arguments
        node_input = deepcopy(node_input)
        node.fit(*node_input[0], **node_input[1])

    def _transform(self, node: NodeBase, node_input):
        """
        Helper function, used for transform call and dumping outputs
        :param node:
        :param node_input:
        :return:
        """

        # ----------------------
        # Transform node
        # ----------------------
        # Make copy of arguments
        node_input = deepcopy(node_input)
        node_result = Pipeline._parse_node_output(node, node.transform(*node_input[0], **node_input[1]))
        return node_result

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: self
        """

        def func(node, node_input, has_downstream=True):
            self._fit(node, node_input)
            if has_downstream:
                rv = self._transform(node, node_input)
                return rv

        for mixin in self.mixins:
            func = mixin.wrap_fit(func)

        func.__name__ = 'fit'

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

        def func(node, node_input, has_downstream=True):
            return self._transform(node, node_input)

        for mixin in self.mixins:
            func = mixin.wrap_transform(func)

        func.__name__ = 'transform'

        outputs = self._iterate_graph(func, *args, **kwargs)
        rv = {'{}_{}'.format(k[0].name, k[1]): v for k, v in outputs.items()}
        # Return just dataframe if len 1
        if len(rv) == 1:
            return next(iter(rv.values()))
        return rv
