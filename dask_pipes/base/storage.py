__all__ = ['StorageBase']


class StorageBase:

    def __getitem__(self, item):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()
