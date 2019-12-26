from typing import Tuple, Dict, Any
from dask_pipes.base import NodeBase


class NodeCallable:
    def __call__(self, node: NodeBase, node_input: Tuple[Tuple[Any], Dict[str, Any]],
                 has_downstream: bool = True) -> Any: ...


class PipelineMixin:

    def _fit(self,
            func: NodeCallable,
            node: NodeBase,
            node_input: Tuple[Tuple[Any], Dict[str, Any]],
            has_downstream=True):
        return func(node, node_input, has_downstream=has_downstream)

    def _transform(self,
                  func: NodeCallable,
                  node: NodeBase,
                  node_input: Tuple[Tuple[Any], Dict[str, Any]],
                  has_downstream=True):
        return func(node, node_input, has_downstream=has_downstream)

    def _wrap_fit(self, fit):
        def func(node, node_input, has_downstream=True):
            return self._fit(fit, node, node_input, has_downstream=has_downstream)

        return func

    def _wrap_transform(self, transform):
        def func(node, node_input, has_downstream=True):
            return self._transform(transform, node, node_input, has_downstream=has_downstream)

        return func
