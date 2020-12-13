import textwrap
from typing import Optional, Tuple

import pytest

import mldag


def test_doc1():
    # Test docstring propagation

    def foo(X: str, y: Optional[int] = None) -> Tuple[int, str]:
        """
        foo summary

        Parameters
        -----------
        y : int
            y description

        Returns
        -----------
        integer : int
            returns 1

        string : str type
            returns 'a'
        """
        return 1, 'a'

    p = mldag.MLDag()
    a = mldag.as_node(foo)

    p['i_X'] >> a['X']
    a['integer'] >> p['out_int']
    a['string'] >> p['out_str']

    p_node = mldag.as_node(p, 'pipeline_node')

    expected_p_fit_doc = textwrap.dedent("""
    Main method for fitting mldag.
    Sequentially calls fit and transform in width-first order

    Parameters
    ------------------
    i_X : str
        Downstream node - foo

    run_id : optional, str
        run identifier string

    Returns
    ------------------
    run : MLDagRun
        computed mldag run
    """).strip()

    expected_p_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters

    Parameters
    ------------------
    i_X : str
        Downstream node - foo

    run_id : str, optional
        mldag run identifier

    Returns
    ------------------
    run : MLDagRun
        computed mldag run containing all node outputs
    """).strip()

    assert p.fit.__doc__ == expected_p_fit_doc
    assert p.transform.__doc__ == expected_p_transform_doc

    expected_p_node_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters

    Parameters
    ------------------
    i_X : str
        Downstream node - foo
    
    Returns
    ------------------
    out_int : int
        Output of foo
    
    out_str : str type
        Output of foo
        """).strip()

    assert p_node.transform.__doc__ == expected_p_node_transform_doc


def test_doc2():
    @mldag.returns([('A', 'int'), ('B', 'str', 'test description')])
    def foo(X):
        return 1, 'a'

    p = mldag.MLDag()
    a = mldag.as_node(foo, name='a')

    p['i_X'] >> a['X']
    a['A'] >> p['out_a']
    a['B'] >> p['out_b']

    p_node = mldag.as_node(p)

    expected_p_node_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters
    
    Parameters
    ------------------
    i_X
        Downstream node - a
    
    Returns
    ------------------
    out_a : int
        Output of a
    
    out_b : str
        Output of a
    """).strip()

    assert p_node.transform.__doc__ == expected_p_node_transform_doc


def test_doc3():
    @mldag.returns([('A', 'int'), ('B', 'str', 'test description')])
    def foo(X):
        return 1, 'a'

    p1 = mldag.MLDag()
    a = mldag.as_node(foo, name='a')
    p1['_in_X'] >> a
    a['A'] >> p1['_out_a']
    a['B'] >> p1['_out_b']

    p1_node = mldag.as_node(p1, name='p1')

    p2 = mldag.MLDag()

    p2['in_X'] >> p1_node['_in_X']
    p1_node['_out_a'] >> p2['out_a']
    p1_node['_out_b'] >> p2['out_b']
    p2_node = mldag.as_node(p2, name='p2')

    expected_p2_node_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters

    Parameters
    ------------------
    in_X
        Downstream node - p1

    Returns
    ------------------
    out_a : int
        Output of p1

    out_b : str
        Output of p1
    """).strip()

    assert p2_node.transform.__doc__ == expected_p2_node_transform_doc


def test_doc4():
    def foo(X):
        return 1

    a = mldag.as_node(foo, name='a')

    assert a.transform.__doc__ is None

    class A:

        def fit(self, X):
            return 1

        def transform(self, X):
            return 1

    a = mldag.as_node(A(), name='a')

    assert a.fit.__doc__ is None
    assert a.transform.__doc__ is None


def test_doc5():
    class A:

        def fit(self, X):
            """

            Parameters
            ----------
            X : some other type

            Returns
            -------
            something
            """
            return 1

        @mldag.returns(['A', 'B'])
        def transform(self, X):
            """
            Description

            Parameters
            ----------
            X : some_type

            Returns
            -------
            A
                some a
            B
                some b
            """
            return 1

    p = mldag.MLDag()
    a = mldag.as_node(A(), name='a')

    p['in'] >> a
    a['A'] >> p['out_a']
    a['B'] >> p['out_b']

    p_node = mldag.as_node(p)

    expected_p_node_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters

    Parameters
    ------------------
    in : some_type
        Downstream node - a
    
    Returns
    ------------------
    out_a
        Output of a
    
    out_b
        Output of a
    """).strip()

    assert p_node.transform.__doc__ == expected_p_node_transform_doc


def test_doc6():
    def foo(a, a_default=None, *var_pos, b, b_default=None, **var_key):
        """
        Parameters
        -------------
        a : a_type
            a description
        b_default : b_type

        kwargs
            kwargs description

        Returns
        -------------
        some_result : int
            1
        """
        return 1

    p = mldag.MLDag()
    a = mldag.as_node(foo, name='a')

    p['in_a'] >> a['a']

    p['in_a_default'] >> a['a_default']

    p['args'] >> a['var_pos']
    p['args'] >> a['var_pos']
    p['in_b'] >> a['b']
    p['in_b_default'] >> a['b_default']
    p['kwargs'] >> a['var_key']
    p['kwargs'] >> a['var_key']

    a['some_result'] >> p['out']

    p_node = mldag.as_node(p)

    expected_p_node_transform_doc = textwrap.dedent("""
    Method for transforming based on previously fitted parameters
    
    Parameters
    ------------------
    in_a : a_type
        Downstream node - a
    
    in_a_default
        Downstream node - a
    
    args
        Downstream node - a
    
    in_b
        Downstream node - a
    
    in_b_default : b_type
        Downstream node - a
    
    kwargs
        Downstream node - a
    
    Returns
    ------------------
    out : int
        Output of a
    """).strip()

    assert p_node.transform.__doc__ == expected_p_node_transform_doc


def test_neg_1():
    @mldag.returns([('A', 'int'), ('B', 'str', 'test description')])
    def foo(X):
        return 1, 'a'

    p1 = mldag.MLDag()

    a = mldag.as_node(foo, name='a')
    p1['_in_X'] >> a

    with pytest.raises(ValueError):
        a.func = foo


def test_neg_2():
    @mldag.returns([('A', 'int'), ('B', 'str', 'test description')])
    def foo(X):
        return 1, 'a'

    p1 = mldag.MLDag()

    a = mldag.as_node(foo, name='a')
    p1['_in_X'] >> a

    with pytest.raises(mldag.exceptions.MldagException):
        a['result'] >> p1['result']


def test_neg_3():
    @mldag.returns([('A', 'int'), ('B', 'str', 'test description')])
    def foo(X):
        return 1, 'a'

    p1 = mldag.MLDag()

    a = mldag.as_node(foo, name='a')
    p1['_in_X'] >> a

    with pytest.raises(ValueError):
        a['A'] >> p1['']


if __name__ == '__main__':
    # Positive tests
    test_doc1()
    test_doc2()
    test_doc3()
    test_doc4()

    # Negative test (assert raises)
    test_neg_1()
    test_neg_2()
    test_neg_3()
