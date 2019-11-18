class DaskPipesException(Exception):
    pass


class DuplicateTaskIdFound(DaskPipesException):
    pass


class ProcessingException(DaskPipesException):
    pass
