from .base import PipelineBase
from .utils import try_create_folder, import_class
import os
from .storage import DirectoryStorage

__all__ = ['FolderPipeline']


class FolderPipeline(PipelineBase):

    def __init__(self, params_folder, persist_folder):
        super().__init__()
        self.params_folder = params_folder
        self.persist_folder = persist_folder

    def fit(self, *args, **kwargs):
        def get_upstream_task_ids(task_id):
            task = self.task_dict[task_id]
            return task.upstream_task_ids

        def fit_transform(task_id, params_storage, persist_storage, dataset):
            task = self.task_dict[task_id]
            params_storage['upstream_task_ids'] = [{'this_slot': i[0], 'other_slot': i[1], 'task_id': i[2]}
                                                   for i in self.task_dict[task_id].upstream_task_ids
                                                   ]
            task.fit(params_storage, persist_storage, dataset.copy())
            params_storage.dump()
            persist_storage.dump()
            transformed = task.__class__.transform(params_storage, persist_storage, dataset.copy(), 'fitting')
            params_storage.dump()
            persist_storage.dump()
            return transformed

        rv = self._iterate_graph(fit_transform, get_upstream_task_ids, *args, **kwargs)
        if len(rv) == 0:
            return None
        if len(rv) == 1:
            return next(iter(rv.values()))
        return rv

    def transform(self, *args, **kwargs):
        # Load graph definition
        upstream_task_ids = dict()
        task_classes = dict()

        for task_id in os.listdir(self.params_folder):
            if task_id == 'input':  # Do not deserialize input
                continue
            task_id_folder = os.path.join(self.params_folder, task_id)
            params = DirectoryStorage(task_id_folder)
            upstream_task_ids[task_id] = [(d['this_slot'], d['other_slot'], d['task_id'])
                                          for d in params['upstream_task_ids']]
            task_classes[task_id] = import_class(params['module'], params['class'])

        def get_upstream_task_ids(task_id):
            return upstream_task_ids[task_id]

        def transform(task_id, params_storage, persist_storage, dataset):
            task_class = task_classes[task_id]
            transformed = task_class.transform(params_storage, persist_storage, dataset.copy(), 'transforming')
            params_storage.dump()
            persist_storage.dump()
            return transformed

        rv = self._iterate_graph(transform, get_upstream_task_ids, *args, **kwargs)
        if len(rv) == 0:
            return None
        if len(rv) == 1:
            return next(iter(rv.values()))
        return rv

    def get_params_storage(self, task_id):
        try_create_folder(self.params_folder)
        return DirectoryStorage(os.path.join(self.params_folder, task_id))

    def get_persist_storage(self, task_id):
        try_create_folder(self.persist_folder)
        return DirectoryStorage(os.path.join(self.persist_folder, task_id))
