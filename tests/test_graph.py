import os

import pytest

import mldag
from mldag.exceptions import MldagException

tmp_folder = 'tmp'
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

params_folder = os.path.join(tmp_folder, 'meta')
persist_folder = os.path.join(tmp_folder, 'persist')

if not os.path.exists(params_folder):
    os.mkdir(params_folder)
if not os.path.exists(persist_folder):
    os.mkdir(persist_folder)

ds_name = 'test_ds'


def test_edges_3():
    f1 = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    v2 = mldag.core.VertexBase()
    v1.graph = f1
    v2.graph = f1

    e = mldag.core.EdgeBase(v1, v2)
    assert len(f1.edges) == 0

    f1.add_edge(e)

    assert f1.edges == [e]


def test_edges_negative_2():
    f1 = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    v1.graph = f1
    v2 = mldag.core.VertexBase()

    f2 = mldag.core.Graph()
    v3 = mldag.core.VertexBase()
    v3.graph = f2
    v4 = mldag.core.VertexBase()

    e0 = mldag.core.EdgeBase()
    with pytest.raises(MldagException):
        f1.add_edge(e0)

    with pytest.raises(MldagException):
        e0.graph = f1

    e1 = mldag.core.EdgeBase(v1, v4, graph=f1)

    with pytest.raises(MldagException):
        f2.add_edge(e1)

    f1.add_edge(e1)

    v1.set_downstream(v2)

    with pytest.raises(MldagException):
        v3.set_downstream(v4)


def test_edges_negative():
    f1 = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    v1.graph = f1
    v2 = mldag.core.VertexBase()

    f2 = mldag.core.Graph()
    v3 = mldag.core.VertexBase()
    f2.add_vertex(v3)
    v4 = mldag.core.VertexBase()

    with pytest.raises(MldagException):
        e1 = mldag.core.EdgeBase(v1, v3)

    e2 = mldag.core.EdgeBase(v2, v4)
    v2.graph = f1
    v4.graph = f2

    with pytest.raises(MldagException):
        f2.add_edge(e2)


def test_edges():
    g = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    g.add_vertex(v1)
    v2 = mldag.core.VertexBase()
    v3 = mldag.core.VertexBase()
    v4 = mldag.core.VertexBase()
    v5 = mldag.core.VertexBase()
    v6 = mldag.core.VertexBase()

    edge1 = mldag.core.EdgeBase(v1, v2)
    edge2 = mldag.core.EdgeBase(v3, v4)

    assert edge2._graph is None
    assert edge2._v1 == v3
    assert edge2._v2 == v4
    assert edge2._v1._graph is None
    assert edge2._v2._graph is None

    edge3 = mldag.core.EdgeBase(v2, v3)
    assert edge3._v1 == v2
    assert edge3._v2 == v3
    assert edge3._v1._graph is None
    assert edge3._v2._graph is None

    edge1.graph = g

    assert edge3._v1 == v2
    assert edge3._v2 == v3
    assert edge3._v1._graph == g
    assert edge3._v2._graph is None

    g.add_edge(edge2)

    assert g.get_edge(v3, v4) == edge2
    assert g.get_edge(v1, v2) == edge1
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v3, v4)]

    edge4 = mldag.core.EdgeBase(v5, v6, g)
    assert g.get_edge(v5, v6) == edge4
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v3, v4), g.get_edge(v5, v6)]

    with pytest.raises(MldagException):
        mldag.core.EdgeBase(graph=g)

    g2 = mldag.core.Graph()

    edge4.graph = g
    assert edge4._graph == g
    with pytest.raises(MldagException):
        edge4.graph = g2


def test_root_leaf():
    g = mldag.core.Graph()
    v1_0 = mldag.core.VertexBase()
    v1_1 = mldag.core.VertexBase()
    v1_2 = mldag.core.VertexBase()
    v2 = mldag.core.VertexBase()
    v3 = mldag.core.VertexBase()
    v4_0 = mldag.core.VertexBase()
    v4_1 = mldag.core.VertexBase()
    v4_2 = mldag.core.VertexBase()

    v1_0.graph = g
    v4_2.graph = g
    v1_0.set_downstream(v2)
    v1_1.set_downstream(v2)
    v1_2.set_downstream(v2)
    v2.set_downstream(v3)
    v2.set_downstream(v4_0)
    v3.set_downstream(v4_0)
    v1_0.set_downstream(v4_1)

    assert g.get_root_vertices() == [v1_0, v4_2, v1_1, v1_2]
    assert g.get_leaf_vertices() == [v4_2, v4_0, v4_1]


def test_vertex_iter():
    g = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    v2 = mldag.core.VertexBase()
    v3 = mldag.core.VertexBase()
    v4 = mldag.core.VertexBase()

    v1.graph = g
    v1.set_downstream(v2)
    v2.set_downstream(v3)
    v2.set_downstream(v4)
    v3.set_downstream(v4)

    assert g.get_root_vertices() == [v1]
    assert g.get_leaf_vertices() == [v4]

    assert list(mldag.core.VertexWidthFirst(g)) == [v1, v2, v3, v4]

    v1.set_upstream(v4)
    with pytest.raises(MldagException):
        list(mldag.core.VertexWidthFirst(g))


def test_graph_1():
    g = mldag.core.Graph()
    v1 = mldag.core.VertexBase()
    g.add_vertex(v1)
    v2 = mldag.core.VertexBase()
    v3 = mldag.core.VertexBase()
    v4 = mldag.core.VertexBase()

    assert g.edges == []
    assert g.vertices == [v1]

    # Assign graph
    v1.graph = g

    assert g.edges == []
    assert g.vertices == [v1]

    g.add_vertex(v4)

    assert g.edges == []
    assert g.vertices == [v1, v4]

    # Connect 1
    v2.set_upstream(v1)
    g.get_edge(v1, v2)
    with pytest.raises(MldagException):
        g.get_edge(v2, v1)
    assert g.get_downstream_vertices(v1) == [v2]
    assert g.get_upstream_vertices(v2) == [v1]
    assert g.edges == [g.get_edge(v1, v2)]
    assert g.vertices == [v1, v4, v2]

    # Connect 2
    g.connect(v2, v3)
    g.get_edge(v2, v3)
    with pytest.raises(MldagException):
        g.get_edge(v3, v2)
    with pytest.raises(MldagException):
        g.get_edge(v1, v3)
    assert g.get_downstream_vertices(v2) == [v3]
    assert g.get_upstream_vertices(v3) == [v2]
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v2, v3)]
    assert g.vertices == [v1, v4, v2, v3]

    # Connect 3
    g.add_edge(mldag.core.EdgeBase(v3, v4))
    g.get_edge(v3, v4)
    with pytest.raises(MldagException):
        g.get_edge(v4, v3)
    assert g.get_downstream_vertices(v3) == [v4]
    assert g.get_upstream_vertices(v4) == [v3]
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v2, v3), g.get_edge(v3, v4)]
    assert g.vertices == [v1, v4, v2, v3]

    g.disconnect(v2, v3)
    with pytest.raises(MldagException):
        g.get_edge(v2, v3)

    assert g.get_downstream_vertices(v2) == []
    assert g.get_upstream_vertices(v3) == []
    with pytest.raises(MldagException):
        g.get_edge(v2, v3)
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v3, v4)]
    assert g.vertices == [v1, v4, v2, v3]

    g.connect(v1, v4)
    v1.set_downstream(v3)

    assert g.get_upstream_vertices(v3) == [v1, ]
    assert g.get_upstream_vertices(v4) == [v3, v1]
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v3, v4),
                       g.get_edge(v1, v4), g.get_edge(v1, v3)]
    assert g.vertices == [v1, v4, v2, v3]

    # Equivalent to removing vertex
    v1.graph = None
    assert v1._id is None
    assert g.get_upstream_vertices(v2) == []
    with pytest.raises(MldagException):
        g.get_downstream_vertices(v1)
    with pytest.raises(MldagException):
        g.get_edge(v1, v2)
    assert g.vertices == [v4, v2, v3]
    assert g.edges == [g.get_edge(v3, v4)]  # 3->4

    # Remove edge
    g.get_edge(v3, v4).graph = None

    assert g.vertices == [v4, v2, v3]
    assert g.edges == []

    # Add some more edges
    v4.set_downstream(v2)
    v3.set_downstream(v2)
    with pytest.raises(MldagException):
        v3.remove_upstream(v2)

    assert g.edges == [g.get_edge(v4, v2), g.get_edge(v3, v2)]

    v2.remove_upstream(v3)
    v4.remove_downstream(v2)
    assert g.edges == []


def test_graph_2():
    pipeline1 = mldag.core.Graph()
    pipeline2 = mldag.core.Graph()
    op1 = mldag.core.VertexBase()
    op2 = mldag.core.VertexBase()

    with pytest.raises(MldagException):
        op1.set_downstream(op2)

    op1.graph = pipeline1
    op2.graph = pipeline2

    with pytest.raises(MldagException):
        op1.set_upstream(op2)

    op2.graph = pipeline1
    op1.set_downstream(op2)
