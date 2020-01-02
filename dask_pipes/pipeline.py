from dask_pipes.base.graph import VertexWidthFirst
from dask_pipes.exceptions import DaskPipesException
from typing import List, Optional
import inspect
from dask_pipes.utils import ReturnDescription
from dask_pipes.base import PipelineBase, NodeBase, NodeConnection, getcallargs_inverse
from uuid import uuid4

from copy import deepcopy

__all__ = ['Pipeline']


class _Empty:
    pass


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
    if downstream_node not in node_arguments:
        raise DaskPipesException("Pipeline does not contain {}".format(downstream_node))
    if edge.upstream_slot not in node_result:
        if len(node_result) == 0:
            raise DaskPipesException("Node {} did not return anything!".format(node))
        raise DaskPipesException(
            "Node {} did not return expected {}; "
            "recieved {}".format(node, edge.upstream_slot, list(node_result.keys())))


class PipelineRun:

    def __init__(self, graph, node_arguments, compute_fit=False, compute_unused=False):
        self.run_id = str(uuid4())
        self.compute_fit = compute_fit
        self.compute_unused = compute_unused
        self.graph = graph
        self.node_arguments = node_arguments
        self.outputs = None

    @staticmethod
    def _fit(node: NodeBase, node_input):
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

    @staticmethod
    def _transform(node: NodeBase, node_input):
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

        node_result = node.transform(*node_input[0], **node_input[1])
        return node_result

    def compute(self):  # noqa: C901
        outputs = dict()

        for node in VertexWidthFirst(self.graph):
            try:
                node: NodeBase
                node_callargs = self.node_arguments[node]

                if len(node_callargs) == 0:
                    raise DaskPipesException("No input for node {}".format(node))

                downstream_edges = self.graph.get_downstream_edges(node)
                has_downstream = len(downstream_edges) > 0

                node_input = getcallargs_inverse(node.fit, **node_callargs)

                # Fit, transform node
                if self.compute_fit:
                    node.fit(*deepcopy(node_input[0]),
                             **deepcopy(node_input[1]),
                             )
                if has_downstream or self.compute_unused:
                    node_result = _parse_node_output(node,
                                                     node.transform(
                                                         *deepcopy(node_input[0]),
                                                         **deepcopy(node_input[1]),
                                                     ))
                else:
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
                    _check_arguements(node, edge, self.node_arguments, downstream_node, node_result)
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

                    downstream_args = self.node_arguments[downstream_node]
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

        outputs = {'{}_{}'.format(k[0].name, k[1]): v for k, v in outputs.items()}
        self.outputs = outputs
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

    def fit(self, *args, **kwargs):
        """
        Main method for fitting pipeline.
        Sequentially calls fit and transform in width-first order
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: self
        """
        node_args = self._parse_arguments(*args, **kwargs)
        run = PipelineRun(graph=self,
                          node_arguments=node_args,
                          compute_fit=True,
                          compute_unused=False)
        return run.compute()

    def transform(self, *args, **kwargs):
        """
        Method for transforming based on previously fitted parameters
        :param args: pipeline positional input to pass to input nodes
        :param kwargs: pipeline key-word input to  pass to input nodes
        :return: dictionary of datasets, or single dataset.
        Output of nodes that was not piped anywhere
        """
        node_args = self._parse_arguments(*args, **kwargs)
        run = PipelineRun(graph=self,
                          node_arguments=node_args,
                          compute_fit=False,
                          compute_unused=True)
        return run.compute()
