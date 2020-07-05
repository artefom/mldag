import inspect
from copy import deepcopy
from typing import List, Optional, Dict, Any, Iterable
from uuid import uuid4

from dask_pipes.core import PipelineBase, NodeBase, NodeConnection, getcallargs_inverse
from dask_pipes.display import display
from dask_pipes.exceptions import DaskPipesException
from dask_pipes.utils import ReturnParameter

__all__ = ['Pipeline']


def _parse_node_output(node, output):
    """
    Convert function return value to dictionary using node output definition

    Parameters
    ----------
    node
        class of node to read transform return annotation from
    output
        return value of .transform() method

    Returns
    -------
    output : dict
        Dictionary of  output name : output value

    Raises
    -------
    DaskPipesException
        If function result length does not match expected number outputs defined in node.outputs
        If function result dictionary keys does not match keys defined in node node.outputs
    """
    expected_result: List[ReturnParameter] = node.outputs
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


def _check_arguements(node, edge, node_arguments, downstream_node, node_result):
    # if downstream_node.name not in node_arguments:
    #     raise DaskPipesException("Pipeline does not contain {}".format(downstream_node))
    if edge.upstream_slot not in node_result:
        if len(node_result) == 0:
            raise DaskPipesException("Node {} did not return anything!".format(node))
        raise DaskPipesException(
            "Node {} did not return expected {}; "
            "recieved {}".format(node, edge.upstream_slot, list(node_result.keys())))


class PipelineRun:
    """
    Manages pipeline computation.
    Derives proper node output to node parameter piping (including key-word and positional arguments)
    Stores node result and pipeline result as a whole after execution
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._computed = False

        self.node_inputs: Dict[str, Dict[str, Any]] = dict()
        self.node_outputs = dict()

        self.inputs = dict()
        self.outputs = dict()

    @staticmethod
    def _handle_var_pos(downstream_args: dict, downstream_slot: str, val: Iterable, node=None, upstream_slot=None):
        """
        Accumulates positional parameters to downstream_args dictionary with checks
        Called multiple times for same downstream_args
        Checks that val is iterable
        Extend downstream_args[downstream_slot] with iterable val and run checks

        Parameters
        ----------
        downstream_args
        downstream_slot
        val
        node
        upstream_slot

        Returns
        -------

        """
        # Create arguments list if it does not exist
        if downstream_slot not in downstream_args:
            downstream_args[downstream_slot] = list()
        try:
            downstream_args[downstream_slot].extend(val)
        except TypeError:
            raise DaskPipesException(
                "{} returned non-iterable "
                "as variadic '{}'. "
                "Expected tuple, received {}".format(node, upstream_slot,
                                                     repr(val))) from None

    @staticmethod
    def _handle_var_key(downstream_args: dict, downstream_slot: str, val: dict, node=None, upstream_slot=None):
        """
        Accumulates key-word parameters in downstream_args dictionary
        Checks for duplicate names, val type to be some kind of mapping
        Updates downstream_args[downstream_slot] with val and run checks

        Parameters
        ----------
        downstream_args
        downstream_slot
        val
        node
        upstream_slot

        Returns
        -------

        """
        # Create arguments dict if it does not exist
        if downstream_slot not in downstream_args:
            downstream_args[downstream_slot] = dict()
        downstream_dict = downstream_args[downstream_slot]
        try:
            for k, v in val.items():
                if k in downstream_dict:
                    raise DaskPipesException(
                        "Duplicate key-word argument "
                        "'{}' for parameter '{}'".format(k, downstream_slot))
                downstream_dict[k] = v
        except AttributeError:
            raise DaskPipesException(
                "{} returned non-mapping "
                "as variadic '{}'. "
                "Expected dict; received {}".format(node, upstream_slot,
                                                    repr(val))) from None

    @staticmethod
    def _handle_pos_or_key(downstream_args, downstream_slot, val: Any, node=None, upstream_slot=None):
        """
        Assigns downstream argument to specific value
        Checks that this parameter was not assigned before
        and check for absence of already existing value

        Parameters
        ----------
        downstream_args
        downstream_slot
        val
            any value
        node
            Used for exception message
        upstream_slot
            Used for exception message
        """
        if downstream_slot in downstream_args:
            raise DaskPipesException(
                "Duplicate argument for parameter '{}'".format(downstream_slot))
        downstream_args[downstream_slot] = val

    @staticmethod
    def _process_node_result_value(downstream_args,
                                   upstream_kind: inspect._ParameterKind,
                                   upstream_slot: str,
                                   downstream_kind: inspect._ParameterKind,
                                   downstream_slot: str,
                                   val: Any,
                                   node: NodeBase = None,
                                   ):
        """
        Add value to downstream_args based on upstream, downstream slot and kind

        If downstream slot is variadic
            if upstream slot is variadic
            - appends contents of variable to already existing dictionary or list
            if upstream slot is not variadic
            - appends value to already existing list or dictionary of values


        Parameters
        ----------
        downstream_args
            dictionary of values to add current value into
        upstream_kind
            upstream parameter kind
        upstream_slot
            upstream slot name - passed explicitly to allow neat exception message
        downstream_kind
            downstream parameter kind
        downstream_slot
            downstream slot name - passed explicitly as in case of variadic downstream parameter,
            list or dictionary will be created
        val : Any
            value to add
        node : optional, NodeBase
            Optional for exception printing
        """

        # Properly handle variadic return arguments
        if upstream_kind != inspect.Parameter.VAR_KEYWORD and upstream_kind != inspect.Parameter.VAR_POSITIONAL:
            if downstream_kind == inspect.Parameter.VAR_POSITIONAL:
                val = (val,)
            elif downstream_kind == inspect.Parameter.VAR_KEYWORD:
                val = {upstream_slot: val}

        {
            inspect.Parameter.VAR_POSITIONAL: PipelineRun._handle_var_pos,
            inspect.Parameter.VAR_KEYWORD: PipelineRun._handle_var_key,
        }.get(downstream_kind, PipelineRun._handle_pos_or_key)(
            downstream_args,
            downstream_slot,
            val,
            node=node,
            upstream_slot=upstream_slot
        )

    @staticmethod
    def _get_kind_by_name(param_name):
        if param_name == '*':
            return inspect.Parameter.VAR_POSITIONAL
        elif param_name == '**':
            return inspect.Parameter.VAR_KEYWORD
        else:
            return inspect.Parameter.POSITIONAL_OR_KEYWORD

    def _populate_node_inputs(self, node: NodeBase, node_result: Any):
        """
        Assign values to downstream nodes' inputs

        Parameters
        ----------
        node
            Node to get downstream from
        node_result
            fit or transform return value

        Returns
        -------

        """

        node_result_dict: Dict[str, Any] = _parse_node_output(node, node_result)

        if not isinstance(node_result_dict, dict):
            raise DaskPipesException(
                "Invalid return. Expected {}, received {}".format(dict.__name__,
                                                                  node_result_dict.__class__.__name__))

        # Propagate node result according to the downstream edges
        for edge in node.get_downstream():
            downstream_node: NodeBase = edge.downstream
            edge: NodeConnection
            _check_arguements(node, edge, self.node_inputs, downstream_node, node_result_dict)

            # Populate arguments of downstream vertex
            downstream_param = next((i for i in downstream_node.inputs if i.name == edge.downstream_slot))

            if downstream_node.name not in self.node_inputs:
                self.node_inputs[downstream_node.name] = dict()

            self._process_node_result_value(
                downstream_args=self.node_inputs[downstream_node.name],
                upstream_kind=self._get_kind_by_name(edge.upstream_slot),
                upstream_slot=edge.upstream_slot,
                downstream_kind=downstream_param.kind,
                downstream_slot=edge.downstream_slot,
                val=node_result_dict[edge.upstream_slot],
                node=node,
            )

        return node_result_dict

    def _set_inputs(self, node_inputs, graph_inputs):
        """
        Infer pipeline inputs from node inputs.
        Updates self.inputs
        """
        # Infer pipeline inputs
        for input in graph_inputs:
            if input.name not in self.inputs:
                self.inputs[input.name] = node_inputs[input.downstream_node.name][input.downstream_slot]

    def _set_outputs(self, graph_outputs, node, node_result_dict):
        """
        Infer pipeline outputs from node outputs
        Updates self.outputs
        """
        # Assign outputs
        for output in graph_outputs:
            if output.upstream_node == node:
                self.outputs[output.name] = node_result_dict[output.upstream_slot]

    def _compute_node(self, node, fit_func, transform_func, graph, compute_fit=False, compute_transform=False):
        """
        Compute node and distribute it's outputs
        self.node_inputs must contain input for 'node' beforehand.
        (self.node_inputs[node.name])
        """
        node: NodeBase
        node_callargs = self.node_inputs[node.name]

        if len(node_callargs) == 0:
            raise DaskPipesException("No input for node {}".format(node))

        # Convert node_callargs to match signature (properly handles *args and **kwargs)
        node_input = getcallargs_inverse(node.transform, **node_callargs)

        # Fit node
        if compute_fit:
            fit_func(self, node, node_input)

        # Transform node
        if compute_transform:
            # Compute results if node has something downstream
            # or we explicitly need to compute unused outputs
            node_result = transform_func(self, node, node_input)
            self.node_outputs[node.name] = node_result

            node_result_dict = self._populate_node_inputs(node, node_result)
            self._set_outputs(graph.outputs, node, node_result_dict)

    def _compute(self,
                 graph,
                 fit_func,
                 transform_func,
                 node_arguments,
                 compute_fit=False,
                 transform_leaf_nodes=False):
        """
        Run pipeline computation
        """
        if self._computed:
            raise DaskPipesException("Run already computed")
        self._computed = True

        # Populate initial node inputs from run arguments
        for k, v in node_arguments.items():
            self.node_inputs[k] = v

        self._set_inputs(self.node_inputs, graph.inputs)

        for node in PipelineWidthFirstIterator(graph):
            node: NodeBase
            try:
                self._compute_node(
                    node=node,
                    fit_func=fit_func,
                    transform_func=transform_func,
                    graph=graph,
                    compute_fit=compute_fit,
                    compute_transform=not node.is_leaf() or transform_leaf_nodes
                )
            except Exception as ex:
                raise DaskPipesException("Error occurred during {}".format(node)) from ex

        return self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'PipelineRun<{}>'.format(self.run_id[:5])


class PipelineWidthFirstIterator:
    """
    Iterator of pipeline nodes
    Iterates width-first by edge direction and node dependencies

    Vertex is visited only if all of its upstream vertices and dependencies are visited
    """

    def __init__(self,
                 graph,
                 starting_nodes=None):
        """
        Initialize iterator

        Parameters
        -----------------------------
        graph : Pipeline
            Pipeline to itearte
        starting_nodes : Optional[Iterable]
            Vertices to start iteration from
            if None (default) - starts iteration from vertices with no upstream dependencies

        Raises
        --------------------
        ValueError
            When graph does not root vertices

        """

        self.seen_nodes = set()
        self.all_nodes = set(graph.node_dict.values())

        # Initialise next nodes list
        if starting_nodes:
            self.next_nodes = [i for i in starting_nodes]
        else:
            self.next_nodes = list()
            for node in self.all_nodes:
                if self._node_ready(node):
                    self.next_nodes.append(node)
            if len(self.next_nodes) == 0:
                raise ValueError("Graph does not have root vertices")

    def _node_ready(self, node):
        for e in node.get_upstream():
            if e.upstream not in self.seen_nodes:
                return False
        for dependency_name, dependency_node in node.iter_valid_dependencies():
            if dependency_node not in self.seen_nodes:
                return False
        return True

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.next_nodes) == 0:
            raise StopIteration()
        next_vertex = self.next_nodes[0]
        del self.next_nodes[0]
        self.seen_nodes.add(next_vertex)

        for node in self.all_nodes.difference(self.seen_nodes).difference(self.next_nodes):
            if self._node_ready(node):
                self.next_nodes.append(node)
        return next_vertex


class Pipeline(PipelineBase):
    """
    Pipeline is a graph structure, containing relationships between NodeBase (vertices)
    with NodeConnection as edges

    Defines fit and transform methods which iterate vertices in width-first order,
    calling fit() and transform() methods of nodes and
    piping outputs of upstream nodes to inputs of downstream nodes
    """

    def __lshift__(self, other):
        raise NotImplementedError()

    def set_upstream(self, other,
                     upstream_slot: Optional[str] = None,
                     downstream_slot: Optional[str] = None):
        raise NotImplementedError()

    @staticmethod
    def _fit(node: NodeBase, node_input):
        """
        Helper function used for fit call and dumping params
        """
        node_input = deepcopy(node_input)
        return node.fit(*node_input[0], **node_input[1])

    @staticmethod
    def _transform(node: NodeBase, node_input):
        """
        Helper function, used for transform call and dumping outputs
        """
        node_input = deepcopy(node_input)
        return node.transform(*node_input[0], **node_input[1])

    def _mixins_initialize(self, run_id):
        """
        Initialize extensions and create fit, transform functions
        """

        def transform_func(run, node, node_input):
            return self._transform(node, node_input)

        def fit_func(run, node, node_input):
            return self._fit(node, node_input)

        for mixin in self.mixins:
            transform_func = mixin._wrap_transform(transform_func)
            fit_func = mixin._wrap_fit(fit_func)

        for mixin in self.mixins:
            mixin._start_run(run_id)

        return fit_func, transform_func

    def _mixins_finalize(self):
        for mixin in self.mixins:
            mixin._end_run()

    def _gen_run_id(self):
        return str(uuid4())

    def fit(self, *args, run_id=None, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order

        Parameters
        ----------
        args
            pipeline positional input to pass to input nodes
        run_id : optional, str
            run identifier string
        kwargs
            pipeline key-word input to  pass to input nodes

        Returns
        -------
        run : PipelineRun
            computed pipeline run
        """
        # Parse pipeline arguments to arguments of specific nodes
        node_args = self._parse_arguments(*args, **kwargs)

        run_id = run_id or self._gen_run_id()
        run = PipelineRun(run_id=run_id)

        fit_func, transform_func = self._mixins_initialize(run_id)
        try:
            return run._compute(
                graph=self,
                node_arguments=node_args,
                compute_fit=True,
                transform_leaf_nodes=False,
                fit_func=fit_func,
                transform_func=transform_func,
            )
        finally:
            self._mixins_finalize()

    def transform(self, *args, run_id=None, **kwargs):
        """
        Method for transforming based on previously fitted parameters


        Parameters
        ----------
        args
            pipeline positional input to pass to input nodes
        run_id : str, optional
            pipeline run identifier
        kwargs
            pipeline key-word input to  pass to input nodes

        Returns
        -------
        run : PipelineRun
            computed pipeline run containing all node outputs

        """
        # Parse pipeline arguments to arguments of specific nodes
        node_args = self._parse_arguments(*args, **kwargs)

        run_id = run_id or self._gen_run_id()
        run = PipelineRun(run_id=run_id)

        fit_func, transform_func = self._mixins_initialize(run_id)

        try:
            return run._compute(
                graph=self,
                node_arguments=node_args,
                compute_fit=False,
                transform_leaf_nodes=True,
                fit_func=fit_func,
                transform_func=transform_func,
            )
        finally:
            self._mixins_finalize()

    def show(self,
             renderer=None,
             show_ports=False,
             show_port_labels=True,
             port_labels_minimal=True,
             show_pipeline_io=True,
             show_class=True,
             class_max_len=0,
             cluster_pipeline_ports=True,
             max_pipeline_depth=-1):
        return display(self,
                       renderer=renderer,
                       show_ports=show_ports,
                       show_port_labels=show_port_labels,
                       port_labels_minimal=port_labels_minimal,
                       show_pipeline_io=show_pipeline_io,
                       show_class=show_class,
                       class_max_len=class_max_len,
                       cluster_pipeline_ports=cluster_pipeline_ports,
                       max_pipeline_depth=max_pipeline_depth)
