from dask_pipes.base.graph import VertexWidthFirst
from dask_pipes.exceptions import DaskPipesException
from typing import Any, Dict, List, Optional
import inspect
from dask_pipes.utils import ReturnDescription
from dask_pipes.base import PipelineBase, NodeBase, NodeConnection, getcallargs_inverse

from copy import deepcopy

__all__ = ['Pipeline']


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
        # This function is used to call inspect.getcallargs for getting function arguments without function itself
        def fit(*_, **__):
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

    @staticmethod
    def _check_arguements(node, edge, node_arguments, downstream_node, node_result):
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
