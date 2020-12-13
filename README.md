MLDAG
====================================
**Lightweight Directed Acyclic Graphs with fit\transform support**

![Test status](https://github.com/artefom/mldag/workflows/unit-test/badge.svg?branch=master)

Why useful
------------------------------------
Most ML engineering pipelines require fit\transform methods. It would be great to be able to organize classes with
fit\transform methods in graph structure and execute fit and transform separately.

MLDAG allows to do just that.

```python
import mldag


class Preprocess:
    def fit(self, dataset):
        # Estimate some parameters
        pass

    def transform(self, dataset):
        # Apply learned transformation to dataset
        return dataset


class Model:
    def fit(self, dataset):
        # Estimate some parameters
        pass

    def transform(self, dataset):
        # Apply learned transformation to dataset
        return dataset


# Create pipeline
dag = mldag.MLDag()

# Initialize nodes
nodes = {
    'preprocess': mldag.as_node(Preprocess()),
    'model': mldag.as_node(Model())
}

dag >> nodes['preprocess'] >> nodes['model'] >> dag

dag.show()
```

![Simple example](https://raw.githubusercontent.com/artefom/mldag/master/examples/simple_example.svg)
