from .slots import BaseOperatorSlot
from ..utils import is_int
from ..exceptions import DaskPipesException, DuplicateTaskIdFound
from typing import TYPE_CHECKING

import dask.dataframe as dd

if TYPE_CHECKING:
    from .operator import BaseOperator

__all__ = ['PipelineBase']


class PipelineBase:

    def __init__(self, name='pipeline'):
        self.name = name
        self.task_dict = dict()
        self.task_count = 0
        self._input_operator = None

    def __getitem__(self, item):
        return BaseOperatorSlot(self, item)

    def __rshift__(self, other, slot='*', other_slot='*'):
        other.__rrshift__(self, slot=other_slot, other_slot=slot)
        return other

    def get_params_storage(self, task_id):
        raise NotImplementedError()

    def get_persist_storage(self, task_id):
        raise NotImplementedError()

    def get_execution_plan(self):

        execution_plan = [self.input_operator.task_id]

        def get_ready_to_execute():
            rv = []
            for task_id, task in self.task_dict.items():
                if task_id in execution_plan:
                    continue
                upstream_tasks = task.upstream_task_ids
                ready = True
                for self_slot, other_slot, upstream_task_id in upstream_tasks:
                    if upstream_task_id not in execution_plan:
                        ready = False
                        break
                if ready:
                    rv.append(task_id)
            return sorted(rv)

        ready_to_execute = get_ready_to_execute()
        while len(ready_to_execute) > 0:
            execution_plan.extend(ready_to_execute)
            ready_to_execute = get_ready_to_execute()

        return execution_plan

    def _iterate_graph(self, transform, get_upstream_task_ids, *args, **kwargs):
        input_params = []
        for ds in args:
            if not isinstance(ds, dd.DataFrame):
                raise TypeError("Expected dd.DataFrame; received {}".format(ds.__class__.__name__))
            input_params.append(('*', ds))
        for k, ds in kwargs.items():
            if not isinstance(ds, dd.DataFrame):
                raise TypeError("Expected dd.DataFrame; received {}".format(ds.__class__.__name__))
            input_params.append((k, ds))

        exec_plan = self.get_execution_plan()
        operator_outputs = dict()
        output_nodes = set()
        for task_id in exec_plan:
            inp_datasets = dict()
            if task_id == 'input':
                for ds_id, ds in enumerate(args):
                    inp_datasets[str(ds_id)] = ds
                for ds_key, ds in kwargs.items():
                    inp_datasets[ds_key] = ds
            else:
                input_num = 0
                for this_slot, upstream_slot, upstream_task_id in get_upstream_task_ids(task_id):
                    if this_slot == '*':
                        if upstream_slot == '*':
                            output_nodes.remove(upstream_task_id)
                            for k, v in operator_outputs[upstream_task_id].items():
                                if is_int(k):
                                    inp_datasets[input_num] = v
                                    input_num += 1
                                else:
                                    if k in inp_datasets:
                                        raise DaskPipesException(
                                            "More than one dataset passed to slot {}, {}".format(k, self))
                                    inp_datasets[k] = v
                        else:
                            output_nodes.remove(upstream_task_id)
                            inp_datasets[upstream_slot] = operator_outputs[upstream_task_id][upstream_slot]
                    else:
                        if upstream_slot == '*':
                            if len(operator_outputs[upstream_task_id]) > 1:
                                raise DaskPipesException(
                                    "More than one dataset passed to slot {}, {}".format(this_slot, self))
                            elif len(operator_outputs[upstream_task_id]) == 0:
                                raise DaskPipesException("No outputs from {}".format(upstream_task_id))
                            else:
                                output_nodes.remove(upstream_task_id)
                                inp_datasets[this_slot] = next(iter(operator_outputs[upstream_task_id].values()))
                        else:
                            output_nodes.remove(upstream_task_id)
                            inp_datasets[this_slot] = operator_outputs[upstream_task_id][upstream_slot]

            operator_outputs[task_id] = dict()
            for dataset_name, dataset in inp_datasets.items():
                params_storage = self.get_params_storage(task_id)
                persist_storage = self.get_persist_storage(task_id)
                if task_id == 'input':
                    output_nodes.add(task_id)
                    operator_outputs[task_id][dataset_name] = dataset
                else:
                    output_nodes.add(task_id)
                    transformed = transform(task_id, params_storage, persist_storage, dataset.copy())
                    operator_outputs[task_id][dataset_name] = transformed

        rv = dict()
        for out_node_id in output_nodes:
            for k, v in operator_outputs[out_node_id].items():
                rv[(out_node_id, k)] = v

        return rv

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def has_input_operator(self):
        return getattr(self, '_input_operator', None) is not None

    @property
    def input_operator(self):
        if not self.has_input_operator():
            raise DaskPipesException("Input operator not defined, use Pipeline >> operator")
        return self._input_operator

    @input_operator.setter
    def input_operator(self, value):
        if self.has_input_operator() and not self._input_operator is value:
            raise DaskPipesException(
                "Input operator {} already assigned and cannot be changed".format(self._input_operator))
        self._input_operator = value  # pylint: disable=attribute-defined-outside-init

    @property
    def params_folder(self):
        return self._meta_folder

    @params_folder.setter
    def params_folder(self, value):
        self._meta_folder = value

    def add_task(self, task):
        """
        Add a task to the Pipeline
        :param task: the task you want to add
        :type task: BaseOperator
        """
        if task.task_id in self.task_dict and self.task_dict[task.task_id] is not task:
            raise DuplicateTaskIdFound(
                "Task id '{}' has already been added to the Pipeline".format(task.task_id))
        else:
            self.task_dict[task.task_id] = task
            task.pipeline = self
        self.task_count = len(self.task_dict)

    def get_task(self, task_id):
        try:
            return self.task_dict[task_id]
        except KeyError:
            raise DaskPipesException("Task {task_id} not found".format(task_id=task_id)) from None

    # Rendering methods
    def __repr__(self):
        return "<{self.__class__.__name__}: {self.name}>".format(self=self)
