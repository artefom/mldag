import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytest
import os

import dask_pipes as dp
from dask_pipes.exceptions import DaskPipesException

ds1 = pd.DataFrame([['cat5', -0.08791349765766582, 1],
                    ['cat2', -0.45607955436914216, np.nan],
                    ['cat4', 1.0365671323315593, 0],
                    ['cat1', -0.024157518723391634, 1],
                    ['cat4', -1.0746881596620674, 1],
                    ['cat2', -1.3745769333109847, 1],
                    ['cat2', -0.8096348940348146, 1],
                    ['cat2', 0.9389351138213718, 1],
                    ['cat1', 0.0816240934021167, 0],
                    ['cat2', 0.23782656204987004, 1]],
                   columns=['cat', 'normal', 'normal2'])
ds2 = pd.DataFrame([['cat4', 0.0925574898889439, -1],
                    ['cat3', 0.5267352833224139, np.nan],
                    ['cat3', -0.6058660301330128, 1],
                    ['cat1', 0.8961509434493576, 1],
                    ['cat3', -0.0027012581656900036, 1],
                    ['cat3', 0.021680712905233424, np.nan],
                    ['cat3', -1.348967911605108, 1],
                    ['cat2', 1.6863322137777539, np.nan],
                    ['cat5', -0.5088200779053001, 1],
                    ['cat1', -0.16265239148925334, np.nan]],
                   columns=['cat', 'normal', 'normal2'])
ds1['normal'] += 2
ds2['normal'] += 5
test_ds = dd.concat([ds1, ds2])

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
    f1 = dp.Graph()
    v1 = dp.VertexBase()
    v2 = dp.VertexBase()
    v1.graph = f1
    v2.graph = f1

    e = dp.EdgeBase(v1, v2)
    assert len(f1.edges) == 0

    f1.add_edge(e)

    assert f1.edges == [e]


def test_edges_negative_2():
    f1 = dp.Graph()
    v1 = dp.VertexBase()
    v1.graph = f1
    v2 = dp.VertexBase()

    f2 = dp.Graph()
    v3 = dp.VertexBase()
    v3.graph = f2
    v4 = dp.VertexBase()

    e0 = dp.EdgeBase()
    with pytest.raises(DaskPipesException):
        f1.add_edge(e0)

    with pytest.raises(DaskPipesException):
        e0.graph = f1

    e1 = dp.EdgeBase(v1, v4, graph=f1)

    with pytest.raises(DaskPipesException):
        f2.add_edge(e1)

    f1.add_edge(e1)

    v1.set_downstream(v2)

    with pytest.raises(DaskPipesException):
        v3.set_downstream(v4)


def test_edges_negative():
    f1 = dp.Graph()
    v1 = dp.VertexBase()
    v1.graph = f1
    v2 = dp.VertexBase()

    f2 = dp.Graph()
    v3 = dp.VertexBase()
    f2.add_vertex(v3)
    v4 = dp.VertexBase()

    with pytest.raises(DaskPipesException):
        e1 = dp.EdgeBase(v1, v3)

    e2 = dp.EdgeBase(v2, v4)
    v2.graph = f1
    v4.graph = f2

    with pytest.raises(DaskPipesException):
        f2.add_edge(e2)


def test_edges():
    g = dp.Graph()
    v1 = dp.VertexBase()
    g.add_vertex(v1)
    v2 = dp.VertexBase()
    v3 = dp.VertexBase()
    v4 = dp.VertexBase()
    v5 = dp.VertexBase()
    v6 = dp.VertexBase()

    edge1 = dp.EdgeBase(v1, v2)
    edge2 = dp.EdgeBase(v3, v4)

    assert edge2._graph is None
    assert edge2._v1 == v3
    assert edge2._v2 == v4
    assert edge2._v1._graph is None
    assert edge2._v2._graph is None

    edge3 = dp.EdgeBase(v2, v3)
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

    edge4 = dp.EdgeBase(v5, v6, g)
    assert g.get_edge(v5, v6) == edge4
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v3, v4), g.get_edge(v5, v6)]

    with pytest.raises(DaskPipesException):
        dp.EdgeBase(graph=g)

    g2 = dp.Graph()

    edge4.graph = g
    assert edge4._graph == g
    with pytest.raises(DaskPipesException):
        edge4.graph = g2


def test_root_leaf():
    g = dp.Graph()
    v1_0 = dp.VertexBase()
    v1_1 = dp.VertexBase()
    v1_2 = dp.VertexBase()
    v2 = dp.VertexBase()
    v3 = dp.VertexBase()
    v4_0 = dp.VertexBase()
    v4_1 = dp.VertexBase()
    v4_2 = dp.VertexBase()

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
    g = dp.Graph()
    v1 = dp.VertexBase()
    v2 = dp.VertexBase()
    v3 = dp.VertexBase()
    v4 = dp.VertexBase()

    v1.graph = g
    v1.set_downstream(v2)
    v2.set_downstream(v3)
    v2.set_downstream(v4)
    v3.set_downstream(v4)

    assert g.get_root_vertices() == [v1]
    assert g.get_leaf_vertices() == [v4]

    assert list(dp.VertexWidthFirst(g)) == [v1, v2, v3, v4]

    v1.set_upstream(v4)
    with pytest.raises(DaskPipesException):
        list(dp.VertexWidthFirst(g))


def test_graph_1():
    g = dp.Graph()
    v1 = dp.VertexBase()
    g.add_vertex(v1)
    v2 = dp.VertexBase()
    v3 = dp.VertexBase()
    v4 = dp.VertexBase()

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
    with pytest.raises(DaskPipesException):
        g.get_edge(v2, v1)
    assert g.get_downstream_vertices(v1) == [v2]
    assert g.get_upstream_vertices(v2) == [v1]
    assert g.edges == [g.get_edge(v1, v2)]
    assert g.vertices == [v1, v4, v2]

    # Connect 2
    g.connect(v2, v3)
    g.get_edge(v2, v3)
    with pytest.raises(DaskPipesException):
        g.get_edge(v3, v2)
    with pytest.raises(DaskPipesException):
        g.get_edge(v1, v3)
    assert g.get_downstream_vertices(v2) == [v3]
    assert g.get_upstream_vertices(v3) == [v2]
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v2, v3)]
    assert g.vertices == [v1, v4, v2, v3]

    # Connect 3
    g.add_edge(dp.EdgeBase(v3, v4))
    g.get_edge(v3, v4)
    with pytest.raises(DaskPipesException):
        g.get_edge(v4, v3)
    assert g.get_downstream_vertices(v3) == [v4]
    assert g.get_upstream_vertices(v4) == [v3]
    assert g.edges == [g.get_edge(v1, v2), g.get_edge(v2, v3), g.get_edge(v3, v4)]
    assert g.vertices == [v1, v4, v2, v3]

    g.disconnect(v2, v3)
    with pytest.raises(DaskPipesException):
        g.get_edge(v2, v3)

    assert g.get_downstream_vertices(v2) == []
    assert g.get_upstream_vertices(v3) == []
    with pytest.raises(DaskPipesException):
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
    with pytest.raises(DaskPipesException):
        g.get_downstream_vertices(v1)
    with pytest.raises(DaskPipesException):
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
    with pytest.raises(DaskPipesException):
        v3.remove_upstream(v2)

    assert g.edges == [g.get_edge(v4, v2), g.get_edge(v3, v2)]

    v2.remove_upstream(v3)
    v4.remove_downstream(v2)
    assert g.edges == []


def test_graph_2():
    pipeline1 = dp.Graph()
    pipeline2 = dp.Graph()
    op1 = dp.VertexBase()
    op2 = dp.VertexBase()

    with pytest.raises(DaskPipesException):
        op1.set_downstream(op2)

    op1.graph = pipeline1
    op2.graph = pipeline2

    with pytest.raises(DaskPipesException):
        op1.set_upstream(op2)

    op2.graph = pipeline1
    op1.set_downstream(op2)
