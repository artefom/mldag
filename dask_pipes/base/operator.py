from typing import Set, Tuple
from ..exceptions import *
from .pipeline import PipelineBase
from .storage import StorageBase
from datetime import datetime
import logging
from .slots import BaseOperatorSlot

logger = logging.getLogger(__name__)

__all__ = ['BaseOperator']


def clear_task_instances(tis,
                         pipeline=None,
                         ):
    pass


class BaseOperatorFitWrap(type):
    """
    Meta-class for wrapping fit function
    """

    @staticmethod
    def wrap_fit(func):
        """Return a wrapped instance method"""

        def fit_wrapped(self, params_storage: StorageBase, persist_storage: StorageBase, *args, **kwargs):
            params_storage['module'] = self.__module__
            params_storage['class'] = self.__class__.__name__
            params_storage['modified'] = datetime.now().isoformat()

            start_d = datetime.now()
            rv = func(self, params_storage, persist_storage, *args, **kwargs)
            end_d = datetime.now()

            params_storage['last_run_elapsed_seconds'] = (end_d - start_d).total_seconds()

            return rv

        return fit_wrapped

    def __new__(mcs, name, bases, attrs):
        """If the class has a 'run' method, wrap it"""
        if 'fit' in attrs:
            attrs['fit'] = mcs.wrap_fit(attrs['fit'])
        return super().__new__(mcs, name, bases, attrs)


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class BaseOperator(metaclass=BaseOperatorFitWrap):

    # noinspection PyUnusedLocal
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
            self,
            task_id: str,
            *args,
            **kwargs
    ):
        if args or kwargs:
            raise DaskPipesException(
                "Invalid arguments were passed to {c} (task_id: {t}). Invalid "
                "arguments were:\n*args: {a}\n**kwargs: {k}".format(
                    c=self.__class__.__name__, a=args, k=kwargs, t=task_id),
            )

        self.task_id = task_id

        # Private attributes
        # (slot, other_slot, item)
        self._upstream_task_ids = set()  # type: Set[Tuple[str, str]]
        # (slot, other_slot, item)
        self._downstream_task_ids = set()  # type: Set[Tuple[str, str]]

        self._pipeline = None
        self._pipeline_slots = None

    # Composing Operators -----------------------------------------------

    def __getitem__(self, item):
        return BaseOperatorSlot(self, item)

    def __rshift__(self, other, slot='*', other_slot='*'):
        """
        Implements Self >> Other == self.set_downstream(other)

        If "Other" is a Pipeline, the Pipeline is assigned to the Operator.
        """

        other_orig = other
        if isinstance(other, BaseOperatorSlot):
            other_slot = other.slot
            other = other.parent
        if isinstance(other, PipelineBase):
            # if this pipeline is already assigned, do nothing
            # otherwise, do normal pipeline assignment
            if not (self.has_pipeline() and self.pipeline is other):
                self.pipeline = other

            # Set pipeline input as upstream
            if not self.pipeline.has_input_operator():
                self.pipeline.input_operator = BaseOperator('input')
            self.set_upstream(self.pipeline.input_operator, slot, other_slot)
        else:
            self.set_downstream(other, slot, other_slot)
        return other_orig

    def __lshift__(self, other, slot='*', other_slot='*'):
        """
        Implements Self << Other == self.set_upstream(other)

        If "Other" is a Pipeline, the Pipeline is assigned to the Operator.
        """
        other_orig = other
        if isinstance(other, BaseOperatorSlot):
            other_slot = other.slot
            other = other.parent
        if isinstance(other, PipelineBase):
            # if this pipeline is already assigned, do nothing
            # otherwise, do normal pipeline assignment
            if not (self.has_pipeline() and self.pipeline is other):
                self.pipeline = other

            # Set pipeline input as upstream
            if not self.pipeline.has_input_operator():
                self.pipeline.input_operator = BaseOperator('input')
            self.set_upstream(self.pipeline.input_operator, slot, other_slot)
        else:
            self.set_upstream(other, slot, other_slot)
        return other_orig

    def __rrshift__(self, other, slot='*', other_slot='*'):
        """
        Called for [Pipeline] >> [Operator] because Pipelines don't have
        __rshift__ operators.
        """

        self.__lshift__(other, slot=slot, other_slot=other_slot)
        return self

    def __rlshift__(self, other, slot='*', other_slot='*'):
        """
        Called for [Pipeline] << [Operator] because Pipelines don't have
        __lshift__ operators.
        """

        self.__rshift__(other, slot=slot, other_slot=other_slot)
        return self

    # /Composing Operators ---------------------------------------------

    @property
    def pipeline(self) -> PipelineBase:
        """
        Returns the Operator's Pipeline if set, otherwise raises an error
        """
        if self.has_pipeline():
            return self._pipeline
        else:
            raise DaskPipesException(
                'Operator {} has not been assigned to a Pipeline yet'.format(self))

    @pipeline.setter
    def pipeline(self, pipeline: PipelineBase):
        """
        Operators can be assigned to one Pipeline, one time. Repeat assignments to
        that same Pipeline are ok.
        """
        if not isinstance(pipeline, PipelineBase):
            raise TypeError(
                'Expected Pipeline; received {}'.format(pipeline.__class__.__name__))
        elif self.has_pipeline() and self.pipeline is not pipeline:
            raise DaskPipesException(
                "The Pipeline assigned to {} can not be changed.".format(self))
        elif self.task_id not in pipeline.task_dict:
            pipeline.add_task(self)
        elif self.task_id in pipeline.task_dict and pipeline.task_dict[self.task_id] != self:
            raise DuplicateTaskIdFound(
                "Task id '{}' has already been added to the Pipeline".format(self.task_id))
        self._pipeline = pipeline  # pylint: disable=attribute-defined-outside-init

    def has_pipeline(self):
        """
        Returns True if the Operator has been assigned to a Pipeline.
        """
        return getattr(self, '_pipeline', None) is not None

    def pre_execute(self, context):
        """
        This hook is triggered right before self.execute() is called.
        """

    def fit(self, params_storage, persist_storage, dataset) -> None:
        """
        Method to override, recieves DataFrames as arguments or kwargs
        :param dataset: Dataset to fit to
        :param params_storage:
        :param persist_storage:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def transform(cls, params_storage: StorageBase, persist_storage: StorageBase, dataset, run_name=None):
        """
        Method to transform dataframe
        :param dataset: Input dataset
        :param params_storage:
        :param persist_storage:
        :param run_name: Name of run for file naming
        :return:
        """
        raise NotImplementedError()

    def post_execute(self, context, result=None):
        """
        This hook is triggered right after self.execute() is called.
        It is passed the execution context and any results returned by the
        operator.
        """

    def on_kill(self):
        """
        Override this method to cleanup subprocesses when a task instance
        gets killed. Any use of the threading, subprocess or multiprocessing
        module within an operator needs to be cleaned up or it will leave
        ghost processes behind.
        """

    @property
    def upstream_list(self):
        """@property: list of tasks directly upstream"""
        return [(slot, other_slot, self.pipeline.get_task(tid)) for slot, other_slot, tid in self._upstream_task_ids]

    @property
    def upstream_task_ids(self):
        """@property: list of ids of tasks directly upstream"""
        return self._upstream_task_ids

    @property
    def downstream_list(self):
        """@property: list of tasks directly downstream"""
        return [(slot, other_slot, self.pipeline.get_task(tid)) for slot, other_slot, tid in self._downstream_task_ids]

    @property
    def downstream_task_ids(self):
        """@property: list of ids of tasks directly downstream"""
        return self._downstream_task_ids

    def clear(self,
              upstream=False,
              downstream=False,
              persist=True,
              params=False):
        """
        Clears the state of task instances associated with the task, following
        the parameters specified.
        """

        tasks = [self.task_id]

        if upstream:
            tasks += [
                t.task_id for t in self.get_flat_relatives(upstream=True)]

        if downstream:
            tasks += [
                t.task_id for t in self.get_flat_relatives(upstream=False)]

        clear_task_instances(tasks, pipeline=self.pipeline)

        return len(tasks)

    def get_flat_relative_ids(self, upstream=False, found_descendants=None):
        """
        Get a flat list of relatives' ids, either upstream or downstream.
        """

        if not found_descendants:
            found_descendants = set()
        relative_ids = self.get_direct_relative_ids(upstream)

        for relative_id in relative_ids:
            if relative_id not in found_descendants:
                found_descendants.add(relative_id)
                relative_task = self._pipeline.task_dict[relative_id]
                relative_task.get_flat_relative_ids(upstream,
                                                    found_descendants)

        return found_descendants

    def get_flat_relatives(self, upstream=False):
        """
        Get a flat list of relatives, either upstream or downstream.
        """
        return list(map(lambda task_id: self._pipeline.task_dict[task_id],
                        self.get_flat_relative_ids(upstream)))

    def get_direct_relative_ids(self, upstream=False):
        """
        Get the direct relative ids to the current task, upstream or
        downstream.
        """
        if upstream:
            return self._upstream_task_ids
        else:
            return self._downstream_task_ids

    def get_direct_relatives(self, upstream=False):
        """
        Get the direct relatives to the current task, upstream or
        downstream.
        """
        if upstream:
            return self.upstream_list
        else:
            return self.downstream_list

    def __repr__(self):
        return "<Task({self.__class__.__name__}): {self.task_id}>".format(
            self=self)

    @property
    def task_type(self):
        """@property: type of the task"""
        return self.__class__.__name__

    def add_only_new(self, item_set, slot, other_slot, item):
        """Adds only new items to item set"""
        if (slot, item) in item_set:
            logger.warning(
                'Dependency %s, %s already registered', self, item)
        else:
            item_set.add((slot, other_slot, item))

    def _set_relatives(self, task_or_task_list, slot1, slot2, upstream=False):
        """Sets relatives for the task."""
        if not isinstance(task_or_task_list, list):
            task_list = [task_or_task_list]
        elif isinstance(task_or_task_list, BaseOperatorSlot):
            task_list = task_or_task_list
        else:
            raise DaskPipesException(
                "Relationships can only be set between "
                "Operators; received {}".format(task_or_task_list.__class__.__name__))

        for task in task_list:
            if not isinstance(task, BaseOperator):
                raise DaskPipesException(
                    "Relationships can only be set between "
                    "Operators; received {}".format(task.__class__.__name__))

        # relationships can only be set if the tasks share a single Pipeline. Tasks
        # without a Pipeline are assigned to that Pipeline.
        pipelines = []
        for task in [self] + task_list:
            if not task.has_pipeline():
                continue
            cur_task_pipeline = task.pipeline
            already_has_pipeline = False
            for task_pipeline in pipelines:
                if task_pipeline is cur_task_pipeline:
                    already_has_pipeline = True
                    break
            if already_has_pipeline:
                continue
            pipelines.append(cur_task_pipeline)

        if len(pipelines) > 1:
            raise DaskPipesException(
                'Tried to set relationships between tasks in '
                'more than one Pipeline: {}'.format(pipelines))
        elif len(pipelines) == 1:
            pipeline = pipelines[0]
        else:
            raise DaskPipesException(
                "Tried to create relationships between tasks that don't have "
                "Pipelines yet. Set the Pipeline for at least one "
                "task  and try again: {}".format([self] + task_list))

        if pipeline and not self.has_pipeline():
            self.pipeline = pipeline

        for task in task_list:
            if pipeline and not task.has_pipeline():
                task.pipeline = pipeline
            if upstream:
                task.add_only_new(task.get_direct_relative_ids(upstream=False), slot2, slot1, self.task_id)
                self.add_only_new(self._upstream_task_ids, slot1, slot2, task.task_id)
            else:
                self.add_only_new(self._downstream_task_ids, slot1, slot2, task.task_id)
                task.add_only_new(task.get_direct_relative_ids(upstream=True), slot2, slot1, self.task_id)

    def set_downstream(self, task_or_task_list, slot='*', downstream_slot='*'):
        """
        Set a task or a task list to be directly downstream from the current
        task.
        """
        self._set_relatives(task_or_task_list, slot, downstream_slot, upstream=False)

    def set_upstream(self, task_or_task_list, slot='*', upstream_slot='*'):
        """
        Set a task or a task list to be directly upstream from the current
        task.
        """
        self._set_relatives(task_or_task_list, slot, upstream_slot, upstream=True)
