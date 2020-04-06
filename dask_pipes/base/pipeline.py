import inspect
from collections import defaultdict
from types import MethodType
from typing import List, Optional, Tuple, Any, Dict, TYPE_CHECKING, Union

from dask_pipes.base._pipeline_utils import (
    PipelineInput,
    PipelineOutput,
    get_input_signature,
    set_fit_signature,
    set_transform_signature,
    reset_transform_signature,
    reset_fit_signature,
    getcallargs_inverse,
    validate_fit_transform,
)
from dask_pipes.base.graph import Graph, VertexBase, EdgeBase
from dask_pipes.exceptions import DaskPipesException
from dask_pipes.utils import (
    get_arguments_description,
    get_return_description,
    ReturnDescription,
    assert_subclass,
)
from dask_pipes.utils import replace_signature
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    import graphviz  # noqa: F401

__all__ = [
    'PipelineMeta',
    'PipelineBase',
    'NodeBaseMeta',
    'NodeBase',
    'NodeSlot',
    'NodeConnection',
    'getcallargs_inverse',
    'PipelineMixin',
    'NodeCallable',
    'as_node',
]


def validate_estimator(obj):
    """
    Raise DaskPipesException if obj does not have 'fit' and 'transform' methods
    :param obj: class instance to check
    :return: None
    """
    if not hasattr(obj, 'fit'):
        raise DaskPipesException("{} must implement fit".format(obj))
    if not hasattr(obj, 'transform'):
        raise DaskPipesException("{} must implement transform".format(obj))


def is_estimator(obj):
    """
    Check if obj has 'fit' and 'transform' methods
    :param obj:
    :return:
    """
    return hasattr(obj, 'fit') and hasattr(obj, 'transform')


class NodeSlot:
    """
    Overloads byte shift to proxy piping to specific slot of parent node
    calls set_downstream with upstream_slot parameter
    """

    def __init__(self, node, slot: str):
        """
        :param node: Node to pipe into (or from)
        :param slot: Slot to pipe into (or from)
        """
        if not hasattr(node, 'set_downstream') or not hasattr(node, 'set_upstream'):
            raise DaskPipesException("node {} must implement set_downstream, set_upstream".format(node))

        self.node = node
        self.slot = slot

    def __rshift__(self, other):
        """
        self >> other

        :param other: can be one of NodeSlot, NodeBase, else as_node(other) is called.
        piping to instance of PipelineBase is not supported
        :return:
        """
        if isinstance(other, NodeSlot):
            # Seems like we're piping output of current node slot to another node slot
            if issubclass(other.node.__class__, NodeBase):
                self.node.set_downstream(other.node, upstream_slot=self.slot, downstream_slot=other.slot)
            elif issubclass(other.node.__class__, PipelineBase):
                other.node.set_output(other.slot, self.node, upstream_slot=self.slot)
                return None
            else:
                raise NotImplementedError("Unknown node class {}".format(other.node.__class__.__name__))

            self.node.set_downstream(other.node, upstream_slot=self.slot, downstream_slot=other.slot)
            return other.node
        elif isinstance(other, NodeBase):
            # Seems like we're piping output of current node slot to another node
            self.node.set_downstream(other, upstream_slot=self.slot)
            return other
        elif isinstance(other, PipelineBase):
            # Seems like we're piping output of current node slot to pipeline
            raise NotImplementedError()
        else:
            # Seems like we're piping output of current node slot to something callable or estimator
            other_wrapped = as_node(other)
            self.node.set_downstream(other_wrapped, upstream_slot=self.slot)
            return other_wrapped

    def __lshift__(self, other):
        """
        self << other

        :param other: can be one of NodeSlot, NodeBase, else as_node(other) is called.
        piping to instance of PipelineBase is not supported
        :return:
        """
        if isinstance(other, NodeSlot):
            # Seems like we're piping output of some node slot to current node slot
            self.node.set_upstream(other.node, upstream_slot=other.slot, downstream_slot=self.slot)
            return other.node
        elif isinstance(other, NodeBase):
            # Seems like we're piping output of some node to current node slot
            self.node.set_upstream(other, downstream_slot=self.slot)
            return other
        elif isinstance(other, PipelineBase):
            # Seems like we're piping output of pipeline to current node slot
            raise NotImplementedError()
        else:
            # Seems like we're piping output of some callable to current node slot
            other_wrapped = as_node(other)
            self.node.set_upstream(other_wrapped, downstream_slot=self.slot)
            return other_wrapped


class NodeConnection(EdgeBase):
    """
    Edge with additional info about upstream and downstream slot name
    """

    def __init__(self, upstream, downstream, upstream_slot: str, downstream_slot: str):
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

    # =======================================================
    # Graphviz representation
    # =======================================================

    def graphviz_edge_repr(self, g, path):
        """
        :type g: graphviz.Digraph
        """
        output_port = ''
        input_port = ''

        if issubclass(self.upstream.__class__, NodeBase):
            _output_port = self.upstream.graphviz_output_port_id(self.upstream_slot)
            if _output_port:
                output_port = ':' + self.upstream.graphviz_output_port_id(self.upstream_slot)

        if issubclass(self.downstream.__class__, NodeBase):
            _input_port = self.upstream.graphviz_input_port_id(self.downstream_slot)
            if _input_port:
                input_port = ':' + self.upstream.graphviz_input_port_id(self.downstream_slot)

        return g.edge(
            self.upstream.graphviz_output_node_id(self.upstream_slot, path) + output_port,
            self.downstream.graphviz_input_node_id(self.downstream_slot, path) + input_port,
        )


class NodeBaseMeta(type):
    """
    Meta-class for validating user-defined nodes
    User classes are derived from NodeBase
    """

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
    """
    Node baseclass, derive from it and overload fit and transform methods
    """

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name
        self._meta = defaultdict(dict)

    def get_default_name(self):
        """
        Get user-friendly name of current node if user does not bother specifying names themselves
        Used when adding node to pipeline and self.name is None, since each node assigned to pipeline must have name
        :return:
        """
        return self.__class__.__name__

    def __getitem__(self, slot):
        """
        Create child NodeSlot that will support piping to specific slots
        :param slot:
        :return:
        """
        available_slots = sorted({i.name for i in self.inputs}.union({i.name for i in self.outputs}))

        if slot not in available_slots:
            raise DaskPipesException(
                "{} Does not have input or output slot {}. Use one of {}".format(
                    self, slot, available_slots))
        return NodeSlot(self, slot)

    def __rshift__(self, other):
        """
        self >> other

        See also:
            NodeSlot.__rshift__
            PipelineBase.__lshift__

        :param other: can be one of NodeSlot, NodeBase, else as_node(other) is called.
        piping to instance of PipelineBase is not supported
        :return:
        """
        if isinstance(other, NodeSlot):
            # Seems like we're piping output of current node to another node slot
            if issubclass(other.node.__class__, NodeBase):
                self.set_downstream(other.node, downstream_slot=other.slot)
            elif issubclass(other.node.__class__, PipelineBase):
                other.node.set_output(other.slot, self)
                return None
            else:
                raise NotImplementedError("Unknown node class {}".format(other.node.__class__.__name__))
            return other.node
        elif isinstance(other, NodeBase):
            # Seems like we're piping output of current node to another node
            self.set_downstream(other)
            return other
        elif isinstance(other, PipelineBase):
            # Seems like we're piping output of current node to some pipeline
            raise NotImplementedError()
        else:
            # Seems like we may be piping come unknown class instance, function or anything else
            other_wrapped = as_node(other)
            self.set_downstream(other_wrapped)
            return other_wrapped

    def __lshift__(self, other):
        """
        self << other

        See also:
            NodeSlot.__lshift__
            PipelineBase.__lshift__

        :param other: can be one of NodeSlot, NodeBase, else as_node(other) is called.
        piping to instance of PipelineBase is not supported
        :return:
        """
        if isinstance(other, NodeSlot):
            # Seems like we're piping output of some node slot to current node
            self.set_upstream(other.node, upstream_slot=other.slot)
            return other.node
        elif isinstance(other, NodeBase):
            # Seems like we're piping output of some node to current node
            self.set_upstream(other)
            return other
        elif isinstance(other, PipelineBase):
            # Seems like we're piping output of some pipeline to current node
            raise NotImplementedError()
        else:
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

    def get_upstream(self):
        return self.graph.get_upstream_edges()

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        """
        Make other Node upstream of current
        Pipes output of specific slot of upstream node to specific slot of current node
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and nodes
        >>> p = PipelineBase()
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

    def get_downstream(self):
        return self.graph.get_downstream_edges(self)

    def is_leaf(self) -> bool:
        """
        Returns true if vertex has do downstream vertices, else false
        """
        return len(self.get_downstream()) == 0

    def set_downstream(self, other,
                       upstream_slot: Optional[str] = None,
                       downstream_slot: Optional[str] = None, ):
        """
        Make other Node upstream of current
        Pipes output of specific slot of upstream node to specific slot of current node
        Automatically assignes other or self to pipeline of one of them is not assigned

        Example:
        >>> # Construct pipeline and nodes
        >>> p = PipelineBase()
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
        >>> p = PipelineBase()
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
        >>> p = PipelineBase()
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

    # =======================================================
    # Methods for replaceing fit's and transforms signatures
    # =======================================================
    def _set_fit_signature(self, sign: inspect.Signature, doc=None):
        """
        Set fit signature of wrapped estimator
        :param sign:
        :param doc:
        :return:
        """
        self.fit = MethodType(replace_signature(self.__class__.fit, sign, doc=doc), self)

    def _set_transform_signature(self, sign: inspect.Signature, doc=None):
        """
        Set transform signature of  wrapped estimator
        :param sign:
        :param doc:
        :return:
        """
        self.transform = MethodType(replace_signature(self.__class__.transform, sign, doc=doc), self)

    def _reset_fit_signature(self):
        """
        When estimator is removed, reset fit signature and documentation string to original
        :return:
        """
        self.fit = MethodType(self.__class__.fit, self)
        self.fit.__doc__ = self.__class__.fit.__doc__

    def _reset_transform_signature(self):
        """
        When estimator is removed, reset transform signature and documentation string to original
        :return:
        """
        self.transform = MethodType(self.__class__.transform, self)
        self.transform.__doc__ = self.__class__.transform.__doc__

    # =================================================================
    # Graphviz node rendering
    # =================================================================

    @property
    def graphviz_show_input_ports(self):
        return len(self.inputs) > 1

    @property
    def graphviz_show_output_ports(self):
        return len(self.outputs) > 1

    def _graphviz_node_id(self, path):
        return '{}{}'.format(path, self.name)

    def _graphviz_node_name(self):
        return self.name

    def graphviz_input_node_id(self, input_slot, path):
        return self._graphviz_node_id(path)

    def graphviz_input_port_id(self, input_slot):
        if self.graphviz_show_input_ports:
            return 'inp_{}'.format(input_slot)

    def _graphviz_input_port_name(self, input_slot):
        return input_slot

    def graphviz_output_node_id(self, output_slot, path):
        return self._graphviz_node_id(path)

    def graphviz_output_port_id(self, output_slot):
        if self.graphviz_show_output_ports:
            return 'out_{}'.format(output_slot)

    def _graphviz_output_port_name(self, output_slot):
        return output_slot

    def graphviz_node_repr(self, g, path):
        """
        :type g: graphviz.Digraph
        """

        if self.graphviz_show_input_ports:
            input_defs = '<TR>' + ''.join([
                '<TD PORT="{port_id}">{port_name}</TD>'.format(
                    port_id=self.graphviz_input_port_id(inp.name),
                    port_name=self._graphviz_input_port_name(inp.name)
                )
                for inp in self.inputs if self.graphviz_input_port_id(inp.name)
            ]) + '</TR>'
        else:
            input_defs = ''

        if self.graphviz_show_output_ports:
            output_defs = '<TR>' + ''.join([
                '<TD PORT="{port_id}">{port_name}</TD>'.format(
                    port_id=self.graphviz_output_port_id(out.name),
                    port_name=self._graphviz_output_port_name(out.name)
                )
                for out in self.outputs if self.graphviz_output_port_id(out.name)
            ]) + "</TR>"
        else:
            output_defs = ''

        max_span = max(len(self.inputs), len(self.outputs))

        return g.node(self._graphviz_node_id(path), f'''<
        <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="2" STYLE="ROUNDED">
          {input_defs}
          <TR>
            <TD COLSPAN="{max_span}">{self._graphviz_node_name()}</TD>
          </TR>
          {output_defs}
        </TABLE>>''', shape='plaintext')


class FunctionNode(NodeBase):
    """
    Wraps custom function for use in pipeline
    Assumes embedded function is stateless and uses it as a transform method
    """

    def __init__(self, name=None, func=None):
        super().__init__(name)
        self._func = None
        self.func = func

    def get_default_name(self):
        try:
            return self.func.__name__
        except DaskPipesException:
            return self.__class__.__name__

    @property
    def func(self):
        if not self._func:
            raise DaskPipesException("{} does not have assigned function".format(self))
        return self._func

    @func.setter
    def func(self, func):
        if func is not None:
            self_parameter = list(inspect.signature(FunctionNode.fit).parameters.values())[0]
            sign = inspect.signature(func)
            fit_sign = inspect.Signature(
                parameters=[self_parameter] + list(sign.parameters.values()),
                return_annotation=FunctionNode
            )
            transform_sign = inspect.Signature(
                parameters=[self_parameter] + list(sign.parameters.values()),
                return_annotation=sign.return_annotation
            )
            self._set_fit_signature(fit_sign)
            self._set_transform_signature(transform_sign, doc=func.__doc__)
            self.__doc__ = func.__doc__
        else:
            self._reset_fit_signature()
            self._reset_transform_signature()
        self._func = func

    def fit(self, *args, **kwargs):
        """
        Callable is used only as transform without any persistent functions,
        so this method does not do anything
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def transform(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class EstimatorNode(NodeBase):
    """
    Wraps BaseEstimator for use in pipeline
    Supports fit and transform methods
    """

    def __init__(self, name=None, estimator=None):
        super().__init__(name)
        self._estimator = None
        self.estimator = estimator

    def get_default_name(self):
        try:
            return self.estimator.__class__.__name__
        except DaskPipesException:
            return self.__class__.__name__

    @property
    def estimator(self):
        if not self._estimator:
            raise DaskPipesException("{} does not have assigned estimator".format(self))
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        if estimator is not None:

            try:
                validate_estimator(estimator)
            except DaskPipesException as ex:
                raise DaskPipesException("Cannot wrap {}, reason: {}".format(estimator, str(ex)))

            fit_sign = inspect.signature(estimator.fit.__func__)
            fit_sign = inspect.Signature(
                parameters=list(fit_sign.parameters.values()),
                return_annotation=EstimatorNode
            )
            self._set_fit_signature(fit_sign, doc=estimator.fit.__doc__)
            self._set_transform_signature(inspect.signature(estimator.transform.__func__),
                                          doc=estimator.transform.__doc__)
            self.__doc__ = estimator.__doc__
        else:
            # noinspection PyTypeChecker
            self.__doc__ = self.__class__.__doc__
            self._reset_transform_signature()
            self._reset_fit_signature()
        self._estimator = estimator

    def __repr__(self):
        if self._estimator is None:
            return '<{}: {}>'.format(self.__class__.__name__, self.name)
        return '<{}({}): {}>'.format(self.__class__.__name__, self.estimator, self.name)

    def fit(self, *args, **kwargs):
        """
        Call fit of wrapped estimator with specific args
        Signature of this method is overwritten once estimator property is set
        :param args:
        :param kwargs:
        :return:
        """
        self.estimator.fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        """
        Call transform of wrapped estimator with specific args
        Signature of this method if overwritten once estimator property is set
        :param args:
        :param kwargs:
        :return:
        """
        return self.estimator.transform(*args, **kwargs)


class PipelineNode(NodeBase):

    def __init__(self, name=None, pipeline=None):
        super().__init__(name)
        self._pipeline = None
        self.pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline):
        if pipeline is not None:
            try:
                validate_estimator(pipeline)
            except DaskPipesException as ex:
                raise DaskPipesException("Cannot wrap {}, reason: {}".format(pipeline, str(ex)))

            try:
                self_parameter = [next(iter(inspect.signature(pipeline.fit.__func__).parameters.values()))]
            except StopIteration:
                self_parameter = list()
            seen_parameter_names = set()
            unique_additional_parameters = list()
            for param in pipeline.inputs:
                if param.name in seen_parameter_names:
                    continue
                seen_parameter_names.add(param.name)
                unique_additional_parameters.append(param)

            fit_sign = inspect.Signature(
                parameters=self_parameter + unique_additional_parameters,
                return_annotation=pipeline.__class__
            )
            self._set_fit_signature(fit_sign, doc=pipeline.fit.__doc__)

            transform_sign = inspect.Signature(
                parameters=self_parameter + unique_additional_parameters,
                return_annotation=tuple((i.name for i in pipeline.outputs))
            )

            self._set_transform_signature(transform_sign,
                                          doc=pipeline.transform.__doc__)
        else:
            # noinspection PyTypeChecker
            self.__doc__ = self.__class__.__doc__
            self._reset_transform_signature()
            self._reset_fit_signature()
        self._pipeline = pipeline

    def __repr__(self):
        if self._pipeline is None:
            return '<{}: {}>'.format(self.__class__.__name__, self.name)
        return '<{}({}): {}>'.format(self.__class__.__name__, self.pipeline, self.name)

    def fit(self, *args, **kwargs):
        self.pipeline.fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        res = self.pipeline.transform(*args, **kwargs)
        return tuple((res.outputs[output.name] for output in self.pipeline.outputs))

    # ===========================================================================
    # Graphviz stuff
    # ===========================================================================

    def _get_input_by_name(self, input_slot) -> PipelineInput:
        inp = [i for i in self.inputs if i.name == input_slot]
        if len(inp) == 0:
            raise DaskPipesException("{} does not have {} input".format(self, input_slot))
        assert len(inp) == 1, "Got multiple matching inputs somehow"
        return inp[0]

    def _get_output_by_name(self, output_slot) -> PipelineOutput:
        out = [i for i in self.outputs if i.name == output_slot]
        if len(out) == 0:
            raise DaskPipesException("{} does not have {} output".format(self, output_slot))
        assert len(out) == 1, "Got multiple matching output somehow"
        return out[0]

    def graphviz_input_node_id(self, input_slot, path):
        return self.pipeline.get_input_id(input_slot, path, self.name)

    def graphviz_input_port_id(self, input_slot):
        pass

    def graphviz_output_node_id(self, output_slot, path):
        return self.pipeline.get_output_id(output_slot, path, self.name)

    def graphviz_output_port_id(self, output_slot):
        pass

    def graphviz_node_repr(self, g, path):
        self.pipeline.graphviz_render(g, path, self.name)


class TransformNode(NodeBase):

    def __init__(self, name=None, node=None):
        super().__init__(name)
        self._node: Optional[NodeBase] = None
        self.node = node

    @property
    def node(self) -> Optional[NodeBase]:
        return self._node

    @node.setter
    def node(self, node):
        if node is not None:
            try:
                validate_estimator(node)
            except DaskPipesException as ex:
                raise DaskPipesException("Cannot wrap {}, reason: {}".format(node, str(ex)))

            fit_sign = inspect.signature(node.fit.__func__)
            fit_sign = inspect.Signature(
                parameters=list(fit_sign.parameters.values()),
                return_annotation=EstimatorNode
            )
            self._set_fit_signature(fit_sign, doc=node.fit.__doc__)
            self._set_transform_signature(inspect.signature(node.transform.__func__),
                                          doc=node.transform.__doc__)
        else:
            # noinspection PyTypeChecker
            self._reset_transform_signature()
            self._reset_fit_signature()
        self._node = node

    def fit(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        return self.node.fit(*args, **kwargs)


def as_node(obj: Any, name=None) -> Union[Union[FunctionNode, EstimatorNode], PipelineNode]:
    """
    Wrap callable or class instance with 'fit' and 'transform' methods to node, so it can be used in pipeline
    To add node to pipeline, see PipelineBase.set_input

    this function is called whenever non-node is piped to pipeline with byte-shift operators
    >>> p = PipelineBase()
    >>> def foo(X): ...
    >>> p >> foo  # as_node is called

    :param obj: Object to wrap
    :param name: Name of node to be used in pipeline
    :return: EstimatorNode or FunctionNode
    """
    if callable(obj):
        return FunctionNode(name=name, func=obj)
    elif issubclass(obj.__class__, PipelineBase):
        return PipelineNode(name=name, pipeline=obj)
    else:
        return EstimatorNode(name=name, estimator=obj)


class DummyNode(NodeBase):
    """
    Dummy node for examples
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X


class PipelineMeta(type):
    """
    Obsolete, deprecated, outdated, to be removed, discontinued
    """

    def __new__(mcs, name, bases, attrs):
        """
        Run validations on pipeline's fit and transform signatures
        """
        validate_fit_transform(name, attrs, obligatory_variadic=True, allow_default=False)
        return super().__new__(mcs, name, bases, attrs)


class NodeCallable:
    """
    Mock of callable func passed inside pipeline and wrapped
    """

    def __call__(self, run, node: NodeBase, node_input: Tuple[Tuple[Any], Dict[str, Any]]) -> Any: ...


class PipelineMixin:

    def __init__(self):
        pass

    def _fit(self,
             run,
             func: NodeCallable,
             node: NodeBase,
             node_input: Tuple[Tuple[Any], Dict[str, Any]]):
        return func(run, node, node_input)

    def _transform(self,
                   run,
                   func: NodeCallable,
                   node: NodeBase,
                   node_input: Tuple[Tuple[Any], Dict[str, Any]],
                   ):
        return func(run, node, node_input)

    def _wrap_fit(self, fit):
        def func(run, node, node_input):
            return self._fit(run, fit, node, node_input)

        return func

    def _wrap_transform(self, transform):
        def func(run, node, node_input):
            return self._transform(run, transform, node, node_input)

        return func

    def _start_run(self, run_id: str):
        pass

    def _end_run(self):
        pass


class PipelineBase(Graph, metaclass=PipelineMeta):

    def __init__(self, mixins: Optional[List[PipelineMixin]] = None):
        super().__init__()

        # Pipeline inputs. use set_input to add input
        self.inputs: List[PipelineInput] = list()
        self.outputs: List[PipelineOutput] = list()

        self.node_dict = dict()

        self._param_downstream_mapping: Optional[Dict[str, List[Tuple[Any, str]]]] = None

        if mixins is None:
            self.mixins: List[PipelineMixin] = list()
        else:
            self.mixins: List[PipelineMixin] = mixins

    @staticmethod
    def _get_default_node_name(node, counter=None):
        """
        Get default unique user-friendly name for node if user did not bother specifying it themselves

        Uses node.get_default_name + counter to assure uniqueness

        :param node: node to get name for
        :param counter:
        :return:
        """
        default_name = node.get_default_name()
        if counter is None or counter == 0:
            return '{}'.format(default_name)
        return '{}{}'.format(default_name, counter)

    def __rshift__(self, other):
        """
        self >> other
        :param other: can be one of NodeSlot, NodeBase, else as_node(other) is called.
        piping to instance of PipelineBase is not supported
        :return:
        """
        if isinstance(other, NodeSlot):
            # Seems like we're trying to pipe input of current pipeline to some node slot
            self.set_input(other.node, name=other.slot, downstream_slot=other.slot)
            return other.node
        elif issubclass(other.__class__, NodeBase):
            # Seems like we're trying to pipe input of current pipeline to some node
            self.set_input(other)
            return other
        elif isinstance(other, PipelineBase):
            # Seems like we're trying to pipe input of current pipeline to another pipeline
            raise NotImplementedError()
        else:
            # Seems like we're trying to pipe input of current pipeline to some callable or estimator
            other_wrapped = as_node(other)
            self.set_input(other_wrapped)
            return other_wrapped

    def __lshift__(self, other):
        """
        self << other

        piping to instance of PipelineBase is not supported.
        Raises NotImplementedError

        :param other: Any
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

    # TODO: rename to add_node
    def add_vertex(self, node: NodeBase, vertex_id=None):
        """
        Add vertex node to current pipeline
        To remove vertex (node), use remove_vertex
        :param node: node to add
        :param vertex_id: node's id or None (if None - autoincrement) (used for deserialization from disk)
        :type vertex_id: int
        :return:
        """
        if node.name is None:
            node_name_counter = 0
            while PipelineBase._get_default_node_name(node, node_name_counter) in self.node_dict:
                node_name_counter += 1
            node.name = PipelineBase._get_default_node_name(node, node_name_counter)
        if node.name in self.node_dict:
            if self.node_dict[node.name] is not node:
                raise DaskPipesException("Duplicate name for node {}".format(node))
        self.node_dict[node.name] = node
        super().add_vertex(node, vertex_id=vertex_id)

    # TODO: Rename to remove_node
    def remove_vertex(self, node: NodeBase) -> VertexBase:
        """
        Remove vertex node from pipeline
        To add vertex (node), use add_vertex
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
    def input_parameters(self) -> List[inspect.Parameter]:
        """
        Get list of inspect.parameter without duplicates (same value can be passed to multiple nodes)
        :return: list of inspect.Parameter for fit function, excluding self
        """
        return get_input_signature(self)[0][1:]

    @property
    def input_names(self):
        return [i.name for i in self.input_parameters]

    @property
    def output_names(self) -> List[str]:
        """
        Get human-friendly list of output names of .transform() function without detailed info about nodes
        :return: list of nodes's outputs that have no downstream nodes. Transform output
        """
        return [o.name for o in self.outputs]

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
        Method, used by NodeSlot to allow using Pipeline in byte-shift notation chains node1['X'] >> p['X']

        Not implemented

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
        Method, used by NodeSlot to allow using Pipeline in byte-shift notation chains p['X'] >> node1['X']

        :param other:
        :param upstream_slot:
        :param downstream_slot:
        :return:
        """
        self.set_input(other, name=upstream_slot, downstream_slot=downstream_slot)

    def remove_input(self, name):
        """
        Remove input by name from pipeline
        Removes all conections between pipeline input with name 'name' and any nodes

        To remove input piped to specific node, use remove_input_node
        To add input node, use set_input

        :param name:
        :return:
        """
        len_before = len(self.inputs)
        if len_before == 0:
            raise DaskPipesException("{} Does not have any arguments".format(self))
        self.inputs = [i for i in self.inputs if i.name == name]
        if len(self.inputs) == len_before:
            raise DaskPipesException("{} Does not have argument {}".format(self, name))
        self._update_fit_transform_signatures()

    def remove_input_node(self, node: NodeBase):
        """
        Unset node as pipeline output
        Removes all connections between node inputs and pipeline inputs

        To add input node, use set_input

        :param node: node to unset as output
        :return:
        """
        self.validate_vertex(node)
        # Find inputs to remove
        self.inputs = [i for i in self.inputs if i.downstream_node is not node]
        self._update_fit_transform_signatures()

    def remove_output(self, name):
        raise NotImplementedError()

    def set_output(self, name: str, upstream_node: NodeBase, upstream_slot: Optional[str] = None,
                   replace: bool = False):
        if not issubclass(upstream_node.__class__, NodeBase):
            raise ValueError("upstream_node must subclass {}".format(NodeBase.__name__))

            # Check if specific output exists
        for output in self.outputs:
            if output.name == name:
                if replace:
                    self.remove_output(name)
                else:
                    raise ValueError("Output {} already exists".format(name))

        # Check existance of upstream node output

        upstream_outputs = [i.name for i in upstream_node.outputs]
        if len(upstream_outputs) == 0:
            raise ValueError("Upstream node does not have any outputs")

        if not upstream_slot:
            if len(upstream_outputs) == 1:
                upstream_slot = upstream_outputs[0]
            else:
                raise ValueError("Upstream node {} has multiple outputs. Upstream node name must be one of {}. "
                                 "Example: upstream['node_name'] >> pipeline['output'] ".format(upstream_node,
                                                                                                upstream_outputs))

        if upstream_slot not in upstream_outputs:
            raise ValueError(
                "Upstream node '{}' does not have output named '{}'. Available outputs: {}".format(upstream_node,
                                                                                                   upstream_slot,
                                                                                                   upstream_outputs))

        self.outputs.append(
            PipelineOutput(name, upstream_node, upstream_slot)
        )

    def set_input(self, node: NodeBase, name=None, downstream_slot=None, suffix: Optional[str] = None):  # noqa: C901
        """
        Several actions are performed:

        1. Add 'node' to current pipeline
        2. if 'name' is None, infer it
        3. add 'name' to pipeline fit and transform signatures
        4. connect argument 'name' to 'downstream_slot' of 'node'

        How 'name' is inferred: 'downstream_slot'+'suffix'
        How 'suffix' is inferred: 'node.name'

        !Recursive when downstream_slot is not specified: applied to each input slot of node

        Raises DaskPipesException when name is specified,
        downstream_slot is not specified, and node has multiple input slots

        Relates:
        To see list of current pipeline inputs, use property .inputs
        To remove input, use remove_input or remove_input_node

        Outputs of node are not set. instead, you can get output of all leaf nodes from PipelineRun class

        :param node: Node to set as pipeline input
        :param name: Optional. Name of input (default: 'downstream_slot'+'suffix' )
        :param downstream_slot: Optional. Name of node's input slot. (default: applied recursive for each slot)
        :param suffix: Optional. used for inferring 'name' parameter
        :return: None
        """

        self.validate_vertex(node)
        # Assign node to current pipeline
        node.graph = self

        node_inputs = {i.name: i for i in node.inputs}
        if len(node_inputs) == 0:
            raise DaskPipesException("{} does not have any inputs".format(node))

        # Cannot pipe single pipeline input to multiple node inputs
        if downstream_slot is None:
            if name is not None and len(node_inputs) > 1:
                raise DaskPipesException(
                    "Node {} has multiple inputs, specific "
                    "pipeline argument name not supported. "
                    "Use suffix instead".format(node))
            for inp in node_inputs.keys():
                self.set_input(node, name, inp, suffix=suffix)
            return

        # Check existence of node input
        if downstream_slot not in node_inputs:
            raise DaskPipesException(
                "{} does not have input {}; available: {}".format(node, downstream_slot, list(node_inputs.keys())))

        if name is None:
            name = downstream_slot
            if suffix is None:
                suffix = '_{}'.format(node.name)

        if suffix is not None:
            name = '{}{}'.format(name, suffix)

        current_args = [i for i in self.inputs if i.name == name]
        if len(current_args) > 0:
            for existing_arg in current_args:
                if existing_arg.downstream_slot == downstream_slot and existing_arg.downstream_node is node:
                    return

        param = node_inputs[downstream_slot]

        self.inputs.append(PipelineInput(name=name,
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
        if len(self.inputs) > 0:
            new_params, param_downstream_mapping = get_input_signature(self)
            self._param_downstream_mapping = param_downstream_mapping
            return_annotation = self.__class__.fit.__annotations__.get('return', inspect._empty)

            set_fit_signature(self, inspect.Signature(
                parameters=new_params,
                return_annotation=return_annotation))

            set_transform_signature(self, inspect.Signature(
                parameters=new_params,
                return_annotation=return_annotation))
        else:
            self._param_downstream_mapping = None
            reset_fit_signature(self)
            reset_transform_signature(self)

    def connect(self,
                upstream: NodeBase,
                downstream: NodeBase,
                upstream_slot: Optional[str] = None,
                downstream_slot: Optional[str] = None) -> NodeConnection:
        """
        General method for connecting two nodes
        Creates directed connection with upstream and downstream slots

        if connection between slots was already made, raises DaskPipesException
        if downstream_slot already has piped input, raises DaskPipesException,
        since multiple inputs for one slot are not allowed

        used in _set_relationship and therefore in all
        set_upstream, set_downstream methods of all nodes

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
            elif len(slots_no_default) == 0:
                raise DaskPipesException(
                    "{} does not have inputs without default value to pipe into. "
                    "Cannot infer downstream_slot. Specify arguments for fit method".format(
                        downstream,
                    )
                )
            downstream_slot = slots_no_default[0].name

        # Create edge
        edge = NodeConnection(upstream=upstream,
                              downstream=downstream,
                              upstream_slot=upstream_slot,
                              downstream_slot=downstream_slot)
        return self.add_edge(edge)

    def _parse_arguments(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Parse fit arguments based on current pipeline inputs and return dictionary of node inputs.

        This function is needed, since pipeline's fit and transform signatures are dynamic
        and we need to mimic python behaviour (call inspect.getcallargs on function with specific arguments)

        If argument has a default value and not provided,
        result dictionary will contain default value for that argument
        :param fit_params:
        :param args: fit arguments
        :param kwargs: fit key-word arguments
        :return:
        """

        fit_params = [param for param in self.input_parameters if param.name in self._param_downstream_mapping]

        # Used for injection of custom signature before passing to getcallargs
        # We need to distill self.fit from parameters passed to pipeline against parameters passed to input nodes
        # This function is used to call inspect.getcallargs for getting function arguments without function itself
        def fit(*_, **__):
            pass

        fit.__signature__ = inspect.Signature(
            parameters=fit_params
        )

        var_pos = {param.name for param in fit_params if param.kind == inspect.Parameter.VAR_POSITIONAL}
        var_key = {param.name for param in fit_params if param.kind == inspect.Parameter.VAR_KEYWORD}
        rv = inspect.getcallargs(fit, *args, **kwargs)
        # Replace variadic with dummy asterisk
        # Doing this, because one variadic can have different names
        rv = {(k if k not in var_pos else '*') if k not in var_key else '**': v for k, v in rv.items()}

        node_arguments: Dict[str, Dict[str, Any]] = dict()
        for node in self.vertices:
            if not isinstance(node, NodeBase):
                raise DaskPipesException("Pipeline must contain only {}".format(NodeBase.__name__))
            node_arguments[node.name] = dict()

        # Convert pipeline arguments to node arguments
        for inp in self.inputs:
            if inp.kind == inspect.Parameter.VAR_KEYWORD:
                inp_name = '**'
            elif inp.kind == inspect.Parameter.VAR_POSITIONAL:
                inp_name = '*'
            else:
                inp_name = inp.name
            if inp_name in rv:
                node_arguments[inp.downstream_node.name][inp.downstream_slot] = rv[inp_name]
            else:
                # We want to be extremely careful on each step
                # Even though, _parse_arguments must return full proper dictionary, filled with default values
                # Double-check it here
                raise DaskPipesException(
                    "Pipeline input {}->{}['{}'] not provided "
                    "and does not have default value".format(inp.name,
                                                             inp.downstream_node,
                                                             inp.downstream_slot, ))

        return node_arguments

    def get_input_id(self, input_slot, path, name):
        return '{}{}/inp_{}'.format(path, name, input_slot)

    def get_output_id(self, output_slot, path, name):
        return '{}{}/out_{}'.format(path, name, output_slot)

    def graphviz_add_input(self, g, input: PipelineInput, path, name):
        # Create node
        g.node(self.get_input_id(input.name, path, name), input.name, shape='ellipse')

        # Connect node
        downstream_node_port = ''
        downstream_node_name = input.downstream_node.graphviz_input_node_id(input.downstream_slot,
                                                                            '{}{}/'.format(path, name))
        _downstream_node_port = input.downstream_node.graphviz_input_port_id(input.downstream_slot)
        if _downstream_node_port:
            downstream_node_port = ':' + _downstream_node_port
        g.edge(self.get_input_id(input.name, path, name), downstream_node_name + downstream_node_port)

    def graphviz_add_output(self, g, output: PipelineOutput, path, name):
        # Create node
        output_node_id = self.get_output_id(output.name, path, name)
        g.node(output_node_id, output.name, shape='ellipse')

        # # Connect node
        upstream_node_port = ''
        upstream_node_name = output.upstream_node.graphviz_output_node_id(output.upstream_slot,
                                                                          '{}{}/'.format(path, name))
        _upstream_node_port = output.upstream_node.graphviz_output_port_id(output.upstream_slot)
        if _upstream_node_port:
            upstream_node_port = ':' + _upstream_node_port
        g.edge(upstream_node_name + upstream_node_port, output_node_id)

    def graphviz_render(self, g, path, name):
        """
        Render pipeline to graphviz graph
        :type g: graphviz.Digraph
        """
        cluster_name = 'cluster_{}'.format(name)
        full_path = '{}{}/'.format(path, name) if path else '{}/'.format(name)
        with g.subgraph(name=cluster_name) as p:
            p.attr(label=name)
            p.attr(color='lightgray')
            # Render vertices and edges
            for vertex in self.vertices:
                vertex.graphviz_node_repr(p, path=full_path)

            for edge in self.edges:
                edge.graphviz_edge_repr(p, path=full_path)

            # Render inputs
            for pipeline_input in self.inputs:
                self.graphviz_add_input(p, pipeline_input, path, name)

            # Render outputs
            for pipeline_output in self.outputs:
                self.graphviz_add_output(p, pipeline_output, path, name)

    def show(self):
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("Graphviz not installed") from None

        g = Digraph('G')

        self.graphviz_render(g, '', 'main')

        return g

    # TODO: rename method (why? - we do not add edges. instead, we add connections)
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

    # TODO: Programmatically update parameters docstring
    def fit(self, *args, **kwargs):
        """
        Sequentially calls fit and transform in width-first order
        """
        raise NotImplementedError()

    # TODO: Programmatically update parameters docstring
    def transform(self, *args, **kwargs):
        """
        Sequentially calls transform in width-first order
        """
        raise NotImplementedError()
