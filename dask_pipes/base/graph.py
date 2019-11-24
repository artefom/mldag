from ..exceptions import DaskPipesException
from typing import Optional, List
import importlib
from ..utils import *

__all__ = ['Graph', 'VertexBase', 'EdgeBase', 'VertexWidthFirst']


class VertexBase:
    """
    Base class for graph vertices
    holds pointer to parent graph and id (int)
    Can only be assigned to one graph at a time
    """

    def __init__(self):
        """
        :param graph: If passed, current vertex assigned to graph
        :type graph: Graph
        """
        self._graph = None  # type: Optional[Graph]
        self._id = None  # type: Optional[int]

    def __repr__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self._id)

    def __hash__(self):
        return hash(self._id)

    @staticmethod
    def validate(vertex):
        """
        :param vertex:
        :type vertex: VertexBase
        :return:
        """
        assert_subclass(vertex, VertexBase)

    def to_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, d):
        return cls()

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if graph is not None:
            Graph.validate(graph)
        if self._graph is not None:
            self._graph.remove_vertex(self)
        self._id = None
        self._graph = None
        if graph is not None:
            # Graph should handle attributes above
            graph.add_vertex(self)

    def _set_relationship(self, other, upstream: bool, *args, **kwargs):
        """
        General method  for connecting two vertices with an edge
        Uses parent's connect method, so two vertices cannot be connected if neither them assigned to a graph
        Also, two vertices from different graphs cannot be connected
        When one of two vertices subject to connection does not have graph assigned, it propagates from
        vertex with assigned graph
        :param other: Vertex to connect to
        :type other: VertexBase
        :param upstream: direction of connections upstream: other -> this; downstream: this -> other
        :return:
        """
        VertexBase.validate(other)
        if self.graph is not None and other.graph is not None:
            if self.graph is not other.graph:
                raise DaskPipesException(
                    "Tried to set relationship between vertices that are assigned to different graphs.")
        elif other.graph is not None:
            self.graph = other.graph
        elif self.graph is not None:
            other.graph = self.graph
        else:
            raise DaskPipesException(
                "Tried to set relationship between vertices that are not assigned to any graph.")

        if upstream:
            self.graph.connect(other, self, *args, **kwargs)
        else:
            self.graph.connect(self, other, *args, **kwargs)

    def _remove_relationship(self, other, upstream: bool):
        """
        General method for removing connections between two vertices of same graph
        :param other: vertex to remove connection with
        :type other: VertexBase
        :param upstream: kind of connection to remove
        :raises: DaskPipesException if two vertices are already disconnected
        :return:
        """
        VertexBase.validate(other)
        if upstream:
            self.graph.disconnect(other, self)
        else:
            self.graph.disconnect(self, other)

    def set_upstream(self, other, *args, **kwargs):
        """

        Create connection:
            THIS <<- OTHER

        :param other: Vertex to connect to
        :type other: VertexBase
        :return:
        """
        self._set_relationship(other, upstream=True, *args, **kwargs)

    def remove_upstream(self, other):
        """
        Remove connection:
            THIS <</- OTHER

        :param other: Vertex to remove connection with
        :type other: VertexBase
        :return:
        """
        self._remove_relationship(other, upstream=True)

    def set_downstream(self, other, *args, **kwargs):
        """

        Create connection:
            THIS ->> OTHER

        :param other: Vertex to connect to
        :type other: VertexBase
        :return:
        """
        self._set_relationship(other, upstream=False, *args, **kwargs)

    def remove_downstream(self, other):
        """

        Remove connection:
            THIS -/>> OTHER

        :param other: Vertex to remove connection with
        :type other: VertexBase
        :return:
        """
        self._remove_relationship(other, upstream=False)


class EdgeBase:
    """
    Directed graph edge,
    Base class for edges between vertices
    holds pointer to parent graph, vertices and id
    """

    def __init__(self, upstream=None, downstream=None, graph=None):
        self._graph = None  # type: Optional[Graph]
        self._id = None  # type: Optional[int]
        self._v1 = None  # type: Optional[VertexBase]
        self._v2 = None  # type: Optional[VertexBase]

        if upstream is not None:
            self.upstream = upstream
        if downstream is not None:
            self.downstream = downstream
        if graph is not None:
            self.graph = graph

    @staticmethod
    def validate(edge):
        """
        :param self:
        :param edge:
        :type edge: EdgeBase
        :return:
        """
        assert_subclass(edge, EdgeBase)

    def to_dict(self):
        return {'v1': self._v1._id, 'v2': self._v2._id}

    @classmethod
    def params_from_dict(cls, graph, d):
        v1 = graph._vertices[d['v1']]
        v2 = graph._vertices[d['v2']]
        return (v1, v2), {}

    @classmethod
    def from_dict(cls, graph, d):
        args, kwargs = cls.params_from_dict(graph, d)
        return cls(*args, **kwargs)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if graph is not None:
            Graph.validate(graph)

        if self._v1 is None or self._v2 is None:
            raise DaskPipesException(
                "Tried to assign {} with one or both NULL vertices to graph {}".format(self, graph))
        if self._graph is not None and graph is not None:
            if self._graph is graph:
                return
            raise DaskPipesException("Edge already assigned to another graph. "
                                     "Edge needs to be assigned with it's edges but "
                                     "implicit removal of vertices from graph not allowed")
        if self._graph is not None:
            self._graph.remove_edge(self)
        self._graph = None
        self._id = None
        if graph is not None:
            # Graph should handle attributes above
            if self._v1 is None or self._v2 is None:
                raise DaskPipesException(
                    "Tried to assign edge {} with one or both null vertices to graph".format(self))
            graph.add_edge(self)

    @property
    def upstream(self):
        return self._v1

    def _sanity_check(self):
        """
        Run sanity checks and assign graph to current edge if both vertices have graph assigned
        :return:
        """
        if self.graph is not None and (self._v1.graph is not None or self._v2.graph is not None):
            if self.graph is not self._v1.graph:
                raise DaskPipesException(
                    "Tried to set relationship between edge and vertex that are assigned to different graphs.")
            if self.graph is not self._v2.graph:
                raise DaskPipesException(
                    "Tried to set relationship between edge and vertex that are assigned to different graphs.")
        if (self._v1 is not None and self._v1.graph is not None) and \
                (self._v2 is not None and self._v2.graph is not None):
            if self._v1.graph is not self._v2.graph:
                raise DaskPipesException(
                    "Tried to set edge between vertices that are assigned to different graphs.")

    @upstream.setter
    def upstream(self, vertex):
        """
        Assign upstream vertex of this edge
        :param vertex: vertex to set upstream
        :type vertex: VertexBase
        :return:
        """
        VertexBase.validate(vertex)
        if self._graph is not None:
            if vertex is None:
                raise DaskPipesException("Cannot assign vertex of already added to graph edge to None. "
                                         "Either assign it to another vertex, or remove edge from graph")
            self._graph.remove_edge(self)
        self._v1 = vertex
        self._sanity_check()

    @property
    def downstream(self):
        return self._v2

    @downstream.setter
    def downstream(self, vertex):
        """
        Assign downstream vertex of this edge
        :param vertex: vertex to set downstream
        :type  vertex: VertexBase
        :return:
        """
        VertexBase.validate(vertex)
        if self._graph is not None:
            self._graph.remove_edge(self)
        self._v2 = vertex
        self._sanity_check()

    def __repr__(self):
        return '<{} ({},{}): {}>'.format(self.__class__.__name__, self._v1, self._v2, self._id)


class VertexWidthFirst:

    def __init__(self, graph, starting_vertices=None, how='width-first'):
        """
        Pipeline Iterator
        :param graph:
        :type graph: Graph
        :param starting_vertices:
        :type starting_vertices: Optional[ List[VertexBase] ]
        :param how: one of 'width-first', 'depth-first'
        """
        self.pipeline = graph
        self.how = how
        self.seen = set()
        self.next_vertices = starting_vertices

        if self.how == 'width-first':
            if self.next_vertices is None:
                self.next_vertices = list(self.pipeline.get_root_vertices())
                if len(self.next_vertices) == 0:
                    raise DaskPipesException("Graph has no root vertices, specify starting vertices explicitly: "
                                             "GraphIter(pipeline, next_vertices = [...])")
            self.seen.update(self.next_vertices)
        elif self.how == 'depth-first':
            raise NotImplementedError()
        else:
            raise DaskPipesException("Unknown iterator order: '{}'".format(self.how))

    def __iter__(self):
        return self

    def __next__(self):
        if self.how == 'width-first':
            if len(self.next_vertices) == 0:
                raise StopIteration()
            next_vertex = self.next_vertices[0]
            del self.next_vertices[0]
            downstream_vertices = set(self.pipeline.get_downstream_vertices(next_vertex)).difference(self.seen)
            for v in downstream_vertices:
                self.next_vertices.append(v)
            self.seen.update(downstream_vertices)
        elif self.how == 'depth-first':
            raise NotImplementedError()
        else:
            raise DaskPipesException("Unknown iterator order: '{}'".format(self.how))
        return next_vertex


class Graph:
    """
    Graph class, provides graph construction and methods for getting upstream,
    downstream relative vertices for each vertices
    """

    def __init__(self):
        self._downstream_edges = dict()
        self._upstream_edges = dict()
        self._vertices = dict()
        self._edges = dict()
        self._vertex_id_counter = 0
        self._edge_id_counter = 0

    @staticmethod
    def validate(graph):
        assert_subclass(graph, Graph)

    @staticmethod
    def validate_vertex(vertex: VertexBase):
        VertexBase.validate(vertex)

    @staticmethod
    def validate_edge(edge: EdgeBase):
        EdgeBase.validate(edge)

    def connect(self, upstream: VertexBase, downstream: VertexBase, *args, **kwargs) -> EdgeBase:
        """
        Connect two vertices together
        :param upstream: vertex to set upstream
        :param downstream: vertex to set downstream
        :param args: params passed to edge
        :param kwargs: params passed to edge
        :return: created edge
        """
        edge = EdgeBase(upstream=upstream, downstream=downstream)
        return self.add_edge(edge)

    def disconnect(self, v1: VertexBase, v2: VertexBase) -> EdgeBase:
        """
        Remove connection between vertex
        :param v1: upstream vertex of connection to remove
        :param v2: downstream vertex of connection to remove
        :return: removed edge
        """
        return self.remove_edge(self.get_edge(v1, v2))

    def add_edge(self, edge: EdgeBase, edge_id=None) -> EdgeBase:
        """
        Add edge to graph
        edge must already have vertices assigned
        :param edge: edge to add
        :return: added edge
        """
        self.validate_edge(edge)
        if edge._graph is not None:
            if edge._graph is not self:
                raise DaskPipesException(
                    "Tried to add edge ({}) that already belongs to another graph ({})".format(edge, edge._graph))
            else:
                # Edge already added, do nothing
                if edge._v1._graph is not self or edge._v2._graph is not self:
                    raise DaskPipesException(
                        "Edge already belongs to graph, but vertices ({}), ({}) do not".format(edge._v1, edge._v2))
                return edge
        self.add_vertex(edge._v1)
        self.add_vertex(edge._v2)
        if edge_id is not None:
            self._edge_id_counter = max(self._edge_id_counter, edge_id + 1)
            edge._id = edge_id
        else:
            edge._id = self._edge_id_counter
            self._edge_id_counter += 1
        edge._graph = self
        self._edges[edge._id] = edge
        self._downstream_edges[edge._v1._id] = self._downstream_edges[edge._v1._id] + (edge._id,)
        self._upstream_edges[edge._v2._id] = self._upstream_edges[edge._v2._id] + (edge._id,)
        return edge

    def get_edges(self, v1: VertexBase, v2: VertexBase) -> List[EdgeBase]:
        self.validate_vertex(v1)
        self.validate_vertex(v2)
        if v1._graph is not self or v2._graph is not self:
            raise DaskPipesException(
                "Tried to get edge between vertices ({}), ({}) "
                "that do not belong to current graph ({})".format(v1, v2, self))
        rv = [self._edges[i] for i in self._downstream_edges[v1._id]
              if self._edges[i].downstream._id == v2._id]
        if len(rv) == 0:
            raise DaskPipesException(
                "Edge ({}), ({}) does not exist".format(v1, v2))
        return rv

    def get_edge(self, v1: VertexBase, v2: VertexBase) -> EdgeBase:
        """
        Get edge by vertex
        :param v1: upstream vertex of edge
        :param v2: downstream vertex of edge
        :raises: DaskPipesException if edge not found
        :return: found edge
        """
        # TODO: REMOVE
        self.validate_vertex(v1)
        self.validate_vertex(v2)
        if v1._graph is not self or v2._graph is not self:
            raise DaskPipesException(
                "Tried to get edge between vertices ({}), ({}) "
                "that do not belong to current graph ({})".format(v1, v2, self))
        try:
            return self._edges[next((i for i in self._downstream_edges[v1._id]
                                     if self._edges[i].downstream._id == v2._id))]
        except StopIteration:
            raise DaskPipesException(
                "Edge ({}), ({}) does not exist".format(v1, v2)) from None

    def remove_edge(self, edge: EdgeBase) -> EdgeBase:
        """
        Remove specific edge from graph
        :param edge: edge to remove (can be obtained with get_edge)
        :return: removed edge
        """
        EdgeBase.validate(edge)
        if edge._graph is not self:
            raise DaskPipesException(
                "Tried to remove edge ({}) that does not belong to current graph ({})".format(edge, self))
        for vertex_id, edge_ids in self._downstream_edges.items():
            self._downstream_edges[vertex_id] = tuple((i for i in edge_ids if i != edge._id))
        for vertex_id, edge_ids in self._upstream_edges.items():
            self._upstream_edges[vertex_id] = tuple((i for i in edge_ids if i != edge._id))
        del self._edges[edge._id]
        edge._id = None
        edge._graph = None
        edge._v1 = None
        edge._v2 = None
        return edge

    def add_vertex(self, vertex: VertexBase, vertex_id=None):
        """
        Add vertex to graph
        :param vertex: vertex to add
        :return: added vertex
        """
        self.validate_vertex(vertex)
        if vertex is None:
            raise DaskPipesException("Expected {}, got None".format(VertexBase.__class__.__name__))
        if vertex._graph is not None:
            if vertex._graph is not self:
                raise DaskPipesException(
                    "Tried to add vertex ({}) that already belongs to another graph ({})".format(vertex, vertex._graph))
            else:
                # Vertex already added, do nothing
                if vertex._id is None or vertex._id not in self._vertices or not self._vertices[vertex._id] is vertex:
                    raise DaskPipesException("Vertex corrupted")
                return
        vertex._graph = self
        if vertex_id is not None:
            self._vertex_id_counter = max(self._vertex_id_counter, vertex_id + 1)
            vertex._id = vertex_id
        else:
            vertex._id = self._vertex_id_counter
            self._vertex_id_counter += 1
        self._vertices[vertex._id] = vertex
        self._downstream_edges[vertex._id] = tuple()
        self._upstream_edges[vertex._id] = tuple()
        return vertex

    def remove_vertex(self, vertex) -> VertexBase:
        """
        Remove vertex from graph
        :param vertex: vertex to remove
        :return: removed vertex
        """
        self.validate_vertex(vertex)
        for edge_id in self._downstream_edges[vertex._id]:
            self.remove_edge(self._edges[edge_id])
        for edge_id in self._upstream_edges[vertex._id]:
            self.remove_edge(self._edges[edge_id])
        del self._vertices[vertex._id]
        vertex._id = None
        vertex._graph = None
        return vertex

    def get_upstream_vertices(self, vertex) -> List[VertexBase]:
        """
        Get list of upstream vertices of specific vertex

        VERTEX <<- upstream vertices (List[VertexBase])

        :param vertex: vertex to get relatives of
        :return: List[VertexBase] - upstream vertices
        """
        self.validate_vertex(vertex)
        if vertex.graph is not self:
            raise DaskPipesException(
                "Tried to get upstream vertices of vertex ({}) "
                "that does not belong to current graph ({})".format(vertex, self))
        return [self._edges[edge_i].upstream for edge_i in self._upstream_edges[vertex._id]]

    def get_downstream_vertices(self, vertex) -> List[VertexBase]:
        """
        Get list of downstream vertices of specific vertex

        VERTEX ->> downstream vertices (List[VertexBase])

        :param vertex: vertex to get relatives of
        :return: List[VertexBase] - downstream vertices
        """
        self.validate_vertex(vertex)
        if vertex.graph is not self:
            raise DaskPipesException(
                "Tried to get downstream vertices of vertex ({}) "
                "that does not belong to current graph ({})".format(vertex, self))
        return [self._edges[edge_i].downstream for edge_i in self._downstream_edges[vertex._id]]

    def get_root_vertices(self):
        """
        Get list of vertices that do not have upstream (source, input) vertices

        :return:
        """
        rv = list()
        for v in self._vertices.values():
            if len(self.get_upstream_vertices(v)) == 0:
                rv.append(v)
        return rv

    def get_leaf_vertices(self):
        """
        Get list of vertices that do not have downstream (sink, output) vertices

        :return:
        """
        rv = list()
        for v in self._vertices.values():
            if len(self.get_downstream_vertices(v)) == 0:
                rv.append(v)
        return rv

    @property
    def vertices(self) -> List[VertexBase]:
        return sorted(self._vertices.values(), key=lambda x: x._id)

    @property
    def edges(self) -> List[EdgeBase]:
        return sorted(self._edges.values(), key=lambda x: x._id)

    def __repr__(self):
        return "<{}: V={},E={}>".format(self.__class__.__name__, len(self.vertices), len(self.edges))

    @staticmethod
    def vertex_to_dict(vertex):
        module_name = vertex.__module__
        class_name = vertex.__class__.__name__
        params = vertex.to_dict()
        return {
            'module': module_name,
            'class': class_name,
            'params': params
        }

    @staticmethod
    def vertex_from_dict(d):
        module_name = d['module']
        class_name = d['class']
        params = d['params']
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.from_dict(params)

    @staticmethod
    def edge_to_dict(edge):
        module_name = edge.__module__
        class_name = edge.__class__.__name__
        params = edge.to_dict()
        return {
            'module': module_name,
            'class': class_name,
            'params': params
        }

    @staticmethod
    def edge_from_dict(graph, d):
        module_name = d['module']
        class_name = d['class']
        params = d['params']
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.from_dict(graph, params)

    def to_dict(self):
        """
        Dump pipeline to dictionary
        :return:
        """
        vertices = {vertex._id: self.vertex_to_dict(vertex) for vertex in self.vertices}
        edges = {edge._id: self.edge_to_dict(edge) for edge in self.edges}
        module_name = self.__module__
        class_name = self.__class__.__name__
        return {
            'module': module_name,
            'class': class_name,
            'vertices': vertices,
            'edges': edges
        }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize pipeline from dictionary
        :param d:
        :return:
        """

        module_name = d['module']
        class_name = d['class']

        graph_cls = getattr(importlib.import_module(module_name), class_name)
        graph = graph_cls()

        for k, v in d['vertices'].items():
            vertex_id = int(k)
            vertex = Graph.vertex_from_dict(v)
            graph.add_vertex(vertex, vertex_id)

        for k, v in d['edges'].items():
            edge_id = int(k)
            e = Graph.edge_from_dict(graph, v)
            graph.add_edge(e, edge_id)

        return graph
