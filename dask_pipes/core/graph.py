import importlib
from typing import Optional, List

from dask_pipes.exceptions import DaskPipesException
from dask_pipes.utils import assert_subclass

__all__ = ['Graph', 'VertexBase', 'EdgeBase', 'VertexWidthFirst']


class VertexBase:
    """
    Base class for graph vertices
    holds pointer to parent graph and id (int)
    Can only be assigned to one graph at a time
    """

    def __init__(self):
        self._graph = None  # type: Optional[Graph]
        self._id = None  # type: Optional[int]

    def __repr__(self):
        return '<{}: {}>'.format(self.__class__.__name__, self._id)

    def __hash__(self):
        return hash(self._id)

    @staticmethod
    def validate(vertex):
        """
        Validates that vertex is of proper class

        Parameters
        ----------
        vertex
            Vertex to validate

        Returns
        -------

        Raises
        ------
        DaskPipesException
            If vertex is not subclass of VertexBase

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
            if self._graph == graph:
                return
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

        Parameters
        ----------
        other : VertexBase
            vertex tp connect to
        upstream : bool
            true if upstream direction
            direction of connections upstream: other -> this; downstream: this -> other
        args
            Additional connection arguments, passed to graph.connect
        kwargs
            Additional connection key-word arguments, passed to graph.connect

        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase

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

        Parameters
        ----------
        other : VertexBase
            vertex to remove connection with
        upstream : bool
            kind of connection to remove
            upstream - true
            downstream - false

        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase

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

        Parameters
        ----------
        other : VertexBase
            vertex to connect to
        args
            Additional connection arguments, passed to graph.connect
        kwargs
            Additional connection key-word arguments, passed to graph.connect

        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase

        """
        self._set_relationship(other, upstream=True, *args, **kwargs)

    def remove_upstream(self, other):
        """
        Remove connection:
        THIS <</- OTHER

        Parameters
        ----------
        other : VertexBase
            Vertex to remove connection with

        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase

        """
        self._remove_relationship(other, upstream=True)

    def set_downstream(self, other, *args, **kwargs):
        """
        Create connection:
        THIS ->> OTHER

        Parameters
        ----------
        other : VertexBase
            Vertex to connect to
        args
            Additional connection arguments, passed to graph.connect
        kwargs
            Additional connection key-word arguments, passed to graph.connect
        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase

        """
        self._set_relationship(other, upstream=False, *args, **kwargs)

    def remove_downstream(self, other):
        """
        Remove connection:
        THIS -/>> OTHER

        Parameters
        ----------
        other : VertexBase
            Vertex to remove connection with

        Returns
        -------

        Raises
        ------
        DaskPipesException
            if vertex already belongs to another graph
        DaskPipesException
            If vertex is not subclass of VertexBase
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

        Parameters
        ----------
        edge : EdgeBase
            edge to validate

        Returns
        -------

        Raises
        -------
        DaskPipesException
            if edge is not subclass of EdgeBase

        """

        assert_subclass(edge, EdgeBase)

    def to_dict(self):
        """
        Serialise current edge as dictionary
        Returns
        -------
        dict : dict
            {'v1': vertex 1 id, 'v2': vertex 2 id}

        """
        return {'v1': self._v1._id, 'v2': self._v2._id}

    @classmethod
    def params_from_dict(cls, graph, d):
        """
        Get vertex parameters from dictionary

        Parameters
        ----------
        graph : Graph
            graph to extract vertices by id from
        d : dict
            Dictionary serialized by to_dict()

        Returns
        -------

        """
        v1 = graph._vertices[d['v1']]
        v2 = graph._vertices[d['v2']]
        return (v1, v2), {}

    @classmethod
    def from_dict(cls, graph, d):
        """
        Create edge from dictionary

        Parameters
        ----------
        graph : Graph
            Graph to get vertices by id from
        d : dict
            Serialized dictionary

        Returns
        -------
        edge: EdgeBase
            EdgeBase instance added to graph
        """
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

        Returns
        -------

        DaskPipesException
            If current edge and it's vertices belong to different graphs
            If vertices belong to different graphs
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

        Parameters
        ----------
        vertex : VertexBase
             vertex to set upstream
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

        Parameters
        ----------
        vertex : VertexBase
            vertex to set downstream
        """
        VertexBase.validate(vertex)
        if self._graph is not None:
            self._graph.remove_edge(self)
        self._v2 = vertex
        self._sanity_check()

    def __repr__(self):
        return '<{} ({},{}): {}>'.format(self.__class__.__name__, self._v1, self._v2, self._id)


class VertexWidthFirst:
    """
    Graph width-first iterator
    """

    def __init__(self, graph, starting_vertices=None):
        """
        Graph Width-First iterator

        Parameters
        ----------
        graph : Graph
            graph to iterate
        starting_vertices : optional, Iterable
            starting vertices to begin iteration from
        """
        self.pipeline = graph
        self.seen = set()
        self.next_vertices = starting_vertices

        if self.next_vertices is None:
            self.next_vertices = list(self.pipeline.get_root_vertices())
            if len(self.next_vertices) == 0:
                raise DaskPipesException("Graph has no root vertices, specify starting vertices explicitly: "
                                         "GraphIter(pipeline, next_vertices = [...])")
        self.seen.update(self.next_vertices)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.next_vertices) == 0:
            raise StopIteration()
        next_vertex = self.next_vertices[0]
        del self.next_vertices[0]
        downstream_vertices = set(self.pipeline.get_downstream_vertices(next_vertex)).difference(self.seen)
        for v in downstream_vertices:
            self.next_vertices.append(v)
        self.seen.update(downstream_vertices)
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

        Parameters
        ----------
        upstream : VertexBase
            vertex to set upstream
        downstream : VertexBase
            vertex to set downstream
        args
            positional arguments passed to edge
        kwargs
            key-word arguments passed to edge

        Returns
        -------

        """
        edge = EdgeBase(upstream=upstream, downstream=downstream)
        return self.add_edge(edge)

    def disconnect(self, v1: VertexBase, v2: VertexBase) -> EdgeBase:
        """
        Remove connection between vertices

        Parameters
        ----------
        v1 : VertexBase
            upstream vertex of connection
        v2 : VertexBase
            downstream vertex of connection

        Returns
        -------
        edge : EdgeBase
            removed edge

        Raises
        -------
        DaskPipesException
            If edge does not belong to current graph
        """
        return self.remove_edge(self.get_edge(v1, v2))

    def get_downstream_edges(self, v: VertexBase) -> List[EdgeBase]:
        """
        Get list of downstream edges of vertex
        Parameters
        ----------
        v : VertexBase
            upstream vertex

        Returns
        -------
        edges: List[EdgeBase]
            list of downstream edges

        Raises
        -------
        DaskPipesException
            If v does not belong to current graph
        """
        self.validate_vertex(v)
        if v._id not in self._vertices or v.graph is not self or self._vertices[v._id] is not v:
            raise DaskPipesException("{} is not in graph".format(v))
        return [self._edges[i] for i in self._downstream_edges[v._id]]

    def get_upstream_edges(self, v: VertexBase) -> List[EdgeBase]:
        """
        Get list of upstream edges of vertex
        Parameters
        ----------
        v : VertexBase
            downstream vertex

        Returns
        -------
        edges : List[EdgeBase]
            list of upstream edges

        Raises
        -------
        DaskPipesException
            If v does not belong to current graph
        """
        self.validate_vertex(v)
        if v._id not in self._vertices or v.graph is not self or self._vertices[v._id] is not v:
            raise DaskPipesException("{} is not in graph".format(v))
        return [self._edges[i] for i in self._upstream_edges[v._id]]

    def add_edge(self, edge: EdgeBase, edge_id=None) -> EdgeBase:
        """
        Add edge to graph
        edge must already have vertices assigned

        Parameters
        -------
        edge : EdgeBase
            edge to add
        edge_id

        Returns
        -------
        edge : EdgeBase
            added edge

        Raises
        -------
        DaskPipesException
            If edge is not subclass of EdgeBase
            If edge already belongs to another graph
            If edge vertices do not belong to same graph as edge
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
        """
        Get list of all edges between v1 as upstream and v2 as downstream
        Parameters
        ----------
        v1 : VertexBase
            upstream vertex
        v2 : VertexBase
            downstream vertex

        Returns
        -------
        edges : List[EdgeBase]
            list of connecting edges

        Raises
        -------
        DaskPipesException
            If either v1 or v2 are not subclass of VertexBase
            If either v1 or v2 do not belong to current graph
        """
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
        Get single edge between v1 as upstream and v2 as downstream
        Parameters
        ----------
        v1 : VertexBase
            upstream vertex
        v2 : VertexBase
            downstream vertex

        Returns
        -------
        edge : EdgeBase
            connecting edge

        Raises
        -------
        DaskPipesException
            If either v1 or v2 are not subclass of VertexBase
            If either v1 or v2 do not belong to current graph
            If edge v1->v2 does not exist
            If there are multiple edges v1->v2
        """
        self.validate_vertex(v1)
        self.validate_vertex(v2)
        if v1._graph is not self or v2._graph is not self:
            raise DaskPipesException(
                "Tried to get edge between vertices ({}), ({}) "
                "that do not belong to current graph ({})".format(v1, v2, self))
        connecting_edges = [i for i in self._downstream_edges[v1._id]
                            if self._edges[i].downstream._id == v2._id]
        if len(connecting_edges) == 0:
            raise DaskPipesException(
                "Edge ({}), ({}) does not exist".format(v1, v2)) from None
        elif len(connecting_edges) > 1:
            raise DaskPipesException(
                "There are multiple edges connecting {} {}. use get_edges instead".format(v1, v2)) from None
        return self._edges[connecting_edges[0]]

    def remove_edge(self, edge: EdgeBase) -> EdgeBase:
        """
        Remove edge from graph

        Parameters
        ----------
        edge : EdgeBase
            edge to remove (can be obtained with get_edge)

        Returns
        -------
            removed edge

        Raises
        -------
        DaskPipesException
            If edge does not belong to current graph
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

        Parameters
        ----------
        vertex : VertexBase
            vertex to add
        vertex_id : optional, int
            vertex id
            if not None (default), updates internal vertex id counter

        Returns
        -------
        vertex : VertexBase
            Added vertex vertex

        Raises
        -------
        DaskPipesException
            If vertex already belongs to another graph
            If vertex is not subclass of VertexBase
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
                return vertex
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

        Parameters
        ----------
        vertex : VertexBase
            vertex to remove

        Returns
        -------
        vertex : VertexBase
            removed vertex

        Raises
        -------
        DaskPipesException
            If vertex is not subclass of VertexBase
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

        Parameters
        ----------
        vertex : VertexBase
            vertex to get relatives of

        Returns
        -------
        vertices : List[VertexBase]
            upstream vertices
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

        Parameters
        ----------
        vertex : VertexBase
            vertex to get relatives of

        Returns
        -------
        vertices : List[VertexBase]
            downstream vertices

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

        Returns
        -------
        vertices : List[VertexBase]
            list of root vertices
        """
        rv = list()
        for v in self._vertices.values():
            if len(self.get_upstream_vertices(v)) == 0:
                rv.append(v)
        return rv

    def get_leaf_vertices(self):
        """
        Get list of vertices that do not have downstream (sink, output) vertices

        Returns
        -------
        vertices : List[VertexBase]
            list of leaf vertices
        """
        rv = list()
        for v in self._vertices.values():
            if len(self.get_downstream_vertices(v)) == 0:
                rv.append(v)
        return rv

    @property
    def vertices(self) -> List[VertexBase]:
        # TODO: Remove sorting for better performance
        return sorted(self._vertices.values(), key=lambda x: x._id)

    @property
    def edges(self) -> List[EdgeBase]:
        # TODO: Remove sorting for better performance
        return sorted(self._edges.values(), key=lambda x: x._id)

    def __repr__(self):
        return "<{}: V={},E={}>".format(self.__class__.__name__, len(self.vertices), len(self.edges))

    @staticmethod
    def vertex_to_dict(vertex):
        """
        Serialize vertex as dictionary

        Parameters
        ----------
        vertex : VertexBase
            vertex to serialize

        Returns
        -------
        dict : dict
            Dictionary represeting vertex

        """
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
        """
        Create vertex from dictionary

        Parameters
        ----------
        d : dict
            Dictionary, created by vertex_to_dict

        Returns
        -------
        vertex : VertexBase
            Newly created vertex
        """
        module_name = d['module']
        class_name = d['class']
        params = d['params']
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.from_dict(params)

    @staticmethod
    def edge_to_dict(edge):
        """
        Serialize edge to dictionary

        Parameters
        ----------
        edge : EdgeBase edge to serialize

        Returns
        -------
        dict : dict
            Dictionary representation of edge

        """
        module_name = edge.__module__
        class_name = edge.__class__.__name__
        params = edge.to_dict()
        return {
            'module': module_name,
            'class': class_name,
            'params': params
        }

    # FIXME: Why this function needs graph parameter?
    @staticmethod
    def edge_from_dict(graph, d):
        """
        Deserialize edge from dictionary

        Parameters
        ----------
        graph : Graph which edge belongs to
        d : Dictionary, representing edge

        Returns
        -------
        edge: EdgeBase
            Newly created edge
        """
        module_name = d['module']
        class_name = d['class']
        params = d['params']
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.from_dict(graph, params)

    def to_dict(self):
        """
        Dump current graph to dictionary

        Returns
        -------
        dict : dict
            Dictionary, representing current pipeline
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

        Parameters
        ------------
        d : dict
            Dictionary representation of graph created by to_dict
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
