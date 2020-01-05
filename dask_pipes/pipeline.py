from dask_pipes.base.graph import VertexWidthFirst
from dask_pipes.exceptions import DaskPipesException
from typing import List, Optional, Dict, Any, Iterable
import inspect
from dask_pipes.utils import ReturnDescription
from dask_pipes.base import PipelineBase, NodeBase, NodeConnection, getcallargs_inverse
from uuid import uuid4

from copy import deepcopy

__all__ = ['Pipeline']


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

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._computed = False

        self.inputs: Dict[str, Dict[str, Any]] = dict()
        self.outputs = dict()
        self.leafs = dict()

    @staticmethod
    def _handle_var_pos(downstream_args, downstream_slot, val: Iterable, node=None, upstream_slot=None):
        """
        extend downstream_args[downstream_slot] with val and run checks
        :param downstream_args:
        :param downstream_slot:
        :param val: iterable
        :param node:
        :param upstream_slot:
        :return:
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
    def _handle_var_key(downstream_args, downstream_slot, val: Dict, node=None, upstream_slot=None):
        """
        Update downstream_args[downstream_slot] with val and run checks
        :param downstream_args:
        :param downstream_slot:
        :param val:
        :param node:
        :param upstream_slot:
        :return:
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
        assign
        downstream_args[downstream_slot] = val
        and check for absence of already existing value
        :param downstream_args:
        :param downstream_slot:
        :param val: any value
        :param node: Used for exception message
        :param upstream_slot: Used for exception message
        :return:
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

        :param downstream_args: dictionary of values to add current value into
        :param upstream_kind: upstream parameter kind
        :param upstream_slot: upstream slot name - passed explicitly to allow neat exception message
        :param downstream_kind: downstream parameter kind
        :param downstream_slot: downstream slot name - passed explicitly as in case of variadic downstream parameter,
        list or dictionary will be created
        :param val: value to add
        :param node: Optional for exception printing
        :return: None
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

    def _process_node_result(self, node: NodeBase, node_result: Any):
        """
        Assign values to downstream nodes' inputs
        :param node: Node to get downstream from
        :param node_result: fit or transform return value
        :return: None
        """

        node_result_dict: Dict[str, Any] = _parse_node_output(node, node_result)

        if not isinstance(node_result_dict, dict):
            raise DaskPipesException(
                "Invalid return. Expected {}, received {}".format(dict.__name__,
                                                                  node_result_dict.__class__.__name__))

        node.get_downstream()

        # Propagate node result according to the downstream edges
        for edge in node.get_downstream():
            downstream_node: NodeBase = edge.downstream
            edge: NodeConnection
            _check_arguements(node, edge, self.inputs, downstream_node, node_result_dict)

            # Populate arguments of downstream vertex
            downstream_param = next((i for i in downstream_node.inputs if i.name == edge.downstream_slot))

            if downstream_node.name in self.inputs:
                self.inputs[downstream_node.name] = dict()

            self._process_node_result_value(
                downstream_args=self.inputs[downstream_node.name],
                upstream_kind=self._get_kind_by_name(edge.upstream_slot),
                upstream_slot=edge.upstream_slot,
                downstream_kind=downstream_param.kind,
                downstream_slot=edge.downstream_slot,
                val=node_result_dict[edge.upstream_slot],
                node=node,
            )

    def _compute(self,
                 graph,
                 fit,
                 transform,
                 node_arguments,
                 compute_fit=False,
                 transform_leaf_nodes=False):
        """
        Run pipeline computation
        :return:
        """
        if self._computed:
            raise DaskPipesException("Run already computed")
        self._computed = True

        for k, v in node_arguments.items():
            self.inputs[k] = v

        for node in VertexWidthFirst(graph):
            try:
                node: NodeBase
                node_callargs = self.inputs[node.name]

                if len(node_callargs) == 0:
                    raise DaskPipesException("No input for node {}".format(node))

                node_input = getcallargs_inverse(node.fit, **node_callargs)

                leaf_node = len(node.get_downstream()) == 0

                # Fit, transform node
                if compute_fit:
                    fit(self, node, node_input)
                if leaf_node and not transform_leaf_nodes:
                    continue

                # Compute results if node has something downstream
                # or we explicitly need to compute unused outputs
                node_result = transform(self, node, node_input)
                self.outputs[node.name] = node_result
                self._process_node_result(node, node_result)

                if leaf_node:
                    self.leafs[node.name] = node_result

            except Exception as ex:
                raise DaskPipesException("Error occurred during {}".format(node)) from ex

        return self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'PipelineRun<{}>'.format(self.run_id[:5])


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
        :param node:
        :param node_input:
        :return:
        """
        node_input = deepcopy(node_input)
        return node.fit(*node_input[0], **node_input[1])

    @staticmethod
    def _transform(node: NodeBase, node_input):
        """
        Helper function, used for transform call and dumping outputs
        :param node:
        :param node_input:
        :return:
        """
        node_input = deepcopy(node_input)
        return node.transform(*node_input[0], **node_input[1])

    def fit(self, *args, run_id=None, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: self
        """
        node_args = self._parse_arguments(*args, **kwargs)

        run_id = run_id or str(uuid4())

        def transform_func(run, node, node_input):
            return self._transform(node, node_input)

        def fit_func(run, node, node_input):
            return self._fit(node, node_input)

        for mixin in self.mixins:
            transform_func = mixin._wrap_transform(transform_func)
            fit_func = mixin._wrap_fit(fit_func)

        for mixin in self.mixins:
            mixin._start_run(run_id)

        run = PipelineRun(run_id=run_id)

        try:
            return run._compute(
                graph=self,
                node_arguments=node_args,
                compute_fit=True,
                transform_leaf_nodes=False,
                fit=fit_func,
                transform=transform_func,
            )
        finally:
            for mixin in self.mixins:
                mixin._end_run()

    def transform(self, *args, run_id=None, **kwargs):
        """
        Method for transforming based on previously fitted parameters
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: dictionary of datasets, or single dataset.
        Output of nodes that was not piped anywhere
        """
        node_args = self._parse_arguments(*args, **kwargs)

        run_id = run_id or str(uuid4())

        def transform_func(run, node, node_input):
            return self._transform(node, node_input)

        def fit_func(run, node, node_input):
            return self._fit(node, node_input)

        for mixin in self.mixins:
            transform_func = mixin._wrap_transform(transform_func)
            fit_func = mixin._wrap_fit(fit_func)

        for mixin in self.mixins:
            mixin._start_run(run_id)

        import pandas as pd
        pd.DataFrame().__str__()

        run = PipelineRun(run_id=run_id)
        try:
            return run._compute(
                graph=self,
                node_arguments=node_args,
                compute_fit=False,
                transform_leaf_nodes=True,
                fit=fit_func,
                transform=transform_func,
            )
        finally:
            for mixin in self.mixins:
                mixin._end_run()
