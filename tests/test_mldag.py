import mldag


def test_args():
    @mldag.returns(['some_result'])
    def foo(a, a_default=None, *var_pos, b, b_default=None, **var_key):
        return a, a_default, var_pos, b, b_default, var_key

    p = mldag.mldag.MLDag()
    a = mldag.as_node(foo, name='a')

    p['a'] >> a['a']
    p['a_default'] >> a['a_default']
    p['args'] >> a['var_pos']
    p['b'] >> a['b']
    p['kwargs'] >> a['var_key']
    a['some_result'] >> p['result']

    run = p.transform(1, 2, 3, 4, b=10, kwarg=10)
    assert run.outputs['result'] == (1, 2, (3, 4), 10, None, {'kwarg': 10})


def test_generic_1():
    def foo(arg1, arg2):
        """
        Parameters
        ----------
        arg1 : int
            some argument
        arg2 : some type

        Returns
        ----------
        res1 : int
        res2 : int
        """

        return arg1, arg2

    dag = mldag.MLDag()
    foo_node = mldag.as_node(foo, 'test_foo1')
    dag >> foo_node
    foo_node >> dag
    assert dag.input_names == ['arg1_test_foo1', 'arg2_test_foo1', 'run_id']
    assert [i.name for i in dag.outputs] == ['res1', 'res2']
    assert dag.output_names == ['res1', 'res2']
    assert dag.transform(1, 2).outputs == {'res1': 1, 'res2': 2}
    assert dag.transform(3, 4).outputs == {'res1': 3, 'res2': 4}


def test_generic_2():
    @mldag.returns(['res1', 'res2'])
    def foo(arg1, arg2):
        return arg1, arg2

    dag = mldag.MLDag()
    foo_node = mldag.as_node(foo, 'test_foo1')
    dag >> foo_node
    foo_node >> dag
    assert dag.input_names == ['arg1_test_foo1', 'arg2_test_foo1', 'run_id']
    assert [i.name for i in dag.outputs] == ['res1', 'res2']
    assert dag.output_names == ['res1', 'res2']
    assert dag.transform(1, 2).outputs == {'res1': 1, 'res2': 2}
    assert dag.transform(3, 4).outputs == {'res1': 3, 'res2': 4}


def test_generic_3():
    def foo(arg1, arg2):
        return arg1, arg2

    dag = mldag.MLDag()
    foo_node = mldag.as_node(foo, 'test_foo1')
    dag >> foo_node
    foo_node >> dag
    assert dag.input_names == ['arg1_test_foo1', 'arg2_test_foo1', 'run_id']
    assert [i.name for i in dag.outputs] == ['result']
    assert dag.transform(1, 2).outputs == {'result': (1, 2)}
    assert dag.transform(3, 4).outputs == {'result': (3, 4)}


def test_nested_1():
    def foo(arg1: int, arg2: int) -> tuple:
        return arg1, arg2

    subdag = mldag.MLDag()
    foo_node = mldag.as_node(foo, 'f1')
    subdag >> foo_node >> subdag
    dag = mldag.MLDag()
    subdag_node = mldag.as_node(subdag, 'subdag')
    dag >> subdag_node >> dag

    assert dag.input_names == ['arg1_f1_subdag', 'arg2_f1_subdag', 'run_id']
    assert [i.name for i in dag.outputs] == ['result']
    assert dag.transform(1, 2).outputs == {'result': (1, 2)}
    assert dag.transform(3, 4).outputs == {'result': (3, 4)}


if __name__ == '__main__':
    test_args()
