from typing import Tuple, Dict, Any
from dask_pipes.base import NodeBase


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
