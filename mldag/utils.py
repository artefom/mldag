import importlib
import inspect
import os
import shutil
from collections import namedtuple
from types import MethodType
from typing import Union, Dict, Any, Callable, List, Optional

import numpydoc.docscrape

from mldag.exceptions import MldagException

__all__ = ['cleanup_empty_dirs', 'try_create_dir', 'import_class',
           'assert_subclass', 'get_arguments_description', 'get_return_description', 'returns',
           'ReturnParameter', 'InputParameter', 'replace_signature', 'to_snake_case',
           'docstring_to_str',
           'set_function_return',
           'INSPECT_EMPTY_PARAMETER', ]

# noinspection PyProtectedMember
INSPECT_EMPTY_PARAMETER = inspect._empty

RETURNS_DECORATOR_ATTRIBUTE = '_dp_return_desc'

VALIDATE_SUBCLASSES = False

# Node input parameter class (Contains additional description field compared to inspect.Parameter)
InputParameter = namedtuple("InputParameter", ['name', 'kind', 'default', 'type', 'desc'])

# Node output parameter class
ReturnParameter = namedtuple("ReturnParameter", ['name', 'type', 'desc'])

RETURN_UNNAMED = 'result'


def to_snake_case(text):
    return ''.join(('_' + i.lower() if i.isupper() else i for i in text if i.isalnum())).lstrip('_')


def replace_signature(func, sign, doc):
    """
    Return wrapped function 'func' with specific signature 'sign' and doc string 'doc'

    Parameters
    ----------
    func
        Function to wrap
    sign
        Signature to use
    doc
        Doc string to use

    Returns
    -------
    func_w
        function wrapper
    """

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped.__func__ = func
    wrapped.__signature__ = sign
    wrapped.__doc__ = doc
    wrapped.__name__ = func.__name__
    return wrapped


def get_arguments_description(func: Callable) -> List[InputParameter]:
    rv = []

    doc = func.__doc__
    if doc:
        parameters_doc = numpydoc.docscrape.NumpyDocString(doc)
        parameters_doc = {p.name: p for p in parameters_doc['Parameters']}
    else:
        parameters_doc = dict()

    for param in inspect.signature(func).parameters.values():
        name = param.name
        kind = param.kind
        default = param.default
        if hasattr(param, 'type'):
            p_type = param.type
            desc = param.desc
        else:
            p_type = param.annotation
            parameter_doc = parameters_doc.get(name)
            if parameter_doc:
                desc = parameter_doc.desc
                if parameter_doc.type:
                    p_type = parameter_doc.type
            else:
                desc = ""
        rv.append(InputParameter(name, kind, default, p_type, desc))
    return rv


def docstring_to_str(np_docstring: numpydoc.docscrape.NumpyDocString):  # noqa: C901
    rv = []
    for k, v in np_docstring.items():
        if isinstance(v, dict):
            continue
        elif isinstance(v, list):
            _v = []
            for v_str in v:
                if v_str == inspect._empty:
                    continue
                elif isinstance(v_str, str) and len(v_str) == 0:
                    continue
                elif isinstance(v_str, numpydoc.docscrape.Parameter):
                    desc = v_str.desc
                    if isinstance(desc, list):
                        desc = '\n    '.join(desc)

                    p_type = v_str.type if v_str.type != inspect._empty else ""
                    p_name = v_str.name if v_str.name != inspect._empty else ""

                    if p_type and p_name:
                        _v.append("{} : {}\n    {}".format(p_name, p_type, desc))
                    else:
                        _v.append("{}\n    {}".format(p_name or p_type, desc))
                else:
                    _v.append(v_str)
            v = _v
        else:
            if v:
                v = [v]
            else:
                v = []

        if len(v) > 0:
            if k == "Summary":
                info = '\n'.join(v)
            else:
                info = "{}\n------------------\n{}".format(k, '\n\n'.join(v))
            rv.append(info)

    return '\n\n'.join(rv)


def _get_return_desc_annotation(func: Callable) -> Optional[List[ReturnParameter]]:
    """
    Get return description from func's annotation
    Parameters
    ----------
    func
        callable to get annotation from

    Returns
    -------
    return_description : List[ReturnDescription], optional
        Returns list if return annotation provided, else returns None
    """
    func_name = func.__qualname__

    return_type = inspect.signature(func).return_annotation

    if isinstance(return_type, tuple) or isinstance(return_type, list):
        if len(return_type) == 0:
            raise MldagException("return type '{}' of {} not understood".format(repr(return_type), func_name))
        rv = list()
        for v in return_type:
            if isinstance(v, tuple) or isinstance(v, list):
                if len(v) != 2:
                    raise MldagException(
                        "Return type annotations must be "
                        "((var_name, var_type1), (var_name2, var_type2))")
                var_name = v[0]
                var_type = v[1]
            elif isinstance(v, str):
                var_name = v
                var_type = INSPECT_EMPTY_PARAMETER
            else:
                raise NotImplementedError()
            if var_name in set((i[0] for i in rv)):
                raise MldagException("duplicate return name: {} of {}".format(var_name, func_name))
            rv.append(ReturnParameter(name=var_name, type=var_type, desc=None))
        return rv
    elif isinstance(return_type, dict):
        return [ReturnParameter(name=k, type=v, desc=None) for k, v in return_type.items()]
    elif isinstance(return_type, str):
        return [ReturnParameter(name=return_type, type=INSPECT_EMPTY_PARAMETER, desc=None)]
    return None


def _get_return_desc_docstring(func: Callable) -> Optional[List[ReturnParameter]]:
    """
    Get return description from docstring
    Numpy docstring format supported

    Parameters
    ----------
    func
        callable to get return description from

    Returns
    -------
    return_description : List[ReturnDescription], optional
        list of return descriptions or None if no parameters docstring provided

    Raises
    -------
    ValueError
        if docstrings contains \'Summary\n---------\'
    """
    doc = func.__doc__
    if not doc:
        return None

    doc = numpydoc.docscrape.NumpyDocString(doc)['Returns']

    if not doc:
        return None

    return [ReturnParameter(i.name if i.name else i.type, i.type if i.name else '', '\n'.join(i.desc)) for i in doc]


def set_function_return(func, i_return_desc):
    """
    Explicitly specifies return description of function

    Parameters
    ----------
    func
    i_return_desc : List of 2,3-tuples or ReturnDescriptions or strings
        In case tuple is passed, it is treated as follows:
        [('name','type'),...]
        [('name','type','description'),...]

    Examples
    ----------

    Set function return names
    >>> def foo(x):
    ...     return 1, 2
    ... set_function_return(foo, ['A','B']) # A = 1, B = 2

    Set function return with specific type and description
    >>> def foo(x):
    ...     return 1, 2
    ... set_function_return(foo, [('A','a-type','a-description'),('B','b-type','b-description')]) # A = 1, B = 2


    """
    if not isinstance(i_return_desc, list):
        raise ValueError("Only list of tuples or ReturnDescriptions or strings is supported")

    return_desc = list()
    for v in i_return_desc:
        if isinstance(v, str):
            return_desc.append((v, INSPECT_EMPTY_PARAMETER, None))
        elif isinstance(v, tuple):
            if len(v) == 2:
                r_name, r_type = v
                r_desc = None
            elif len(v) == 3:
                r_name, r_type, r_desc = v
            else:
                raise ValueError("Only 2 or 3-tuples supported")
            return_desc.append((r_name, r_type, r_desc))
        elif isinstance(v, ReturnParameter):
            return_desc.append((v.name, v.type, v.description))
        else:
            raise ValueError("Only list of tuples or ReturnDescriptions or strings is supported")

    if isinstance(func, MethodType):
        setattr(func.__func__, RETURNS_DECORATOR_ATTRIBUTE, return_desc)
    else:
        setattr(func, RETURNS_DECORATOR_ATTRIBUTE, return_desc)


def get_function_return(func):
    if isinstance(func, MethodType):
        func = func.__func__

    if hasattr(func, RETURNS_DECORATOR_ATTRIBUTE):
        return [ReturnParameter(r_name, r_type, r_desc)
                for r_name, r_type, r_desc in getattr(func, RETURNS_DECORATOR_ATTRIBUTE)]
    raise ValueError("Function {} does not have result assigned")


def returns(i_return_desc):
    """
    Explicitly specifies return description of function

    Parameters
    ----------
    i_return_desc : List of 2,3-tuples or ReturnDescriptions or strings

    Examples
    -------

    Set return of function as A,B
    >>> @returns(['A','B'])
    ... def foo(x):
    ...     return 1, 2

    Set return of function with type and description
    >>> @returns([('A','a-type','a-description'),('B','b-type','b-description')])
    ... def foo(x):
    ...     return 1, 2

    """

    def _returns(func):
        set_function_return(func, i_return_desc)
        return func

    return _returns


def _get_return_desc_wrapper(func: Callable) -> Optional[List[ReturnParameter]]:
    """
    Get return description if callable was decorated

    Parameters
    ----------
    func
        callable to get return description from

    Returns
    -------
    return_description : List[ReturnDescription], optional
        list of return descriptions or None
    """
    try:
        return get_function_return(func)
    except ValueError:
        return None


def _get_default_return_annotation(func: Callable) -> List[ReturnParameter]:
    return [ReturnParameter(name=RETURN_UNNAMED,
                            type=inspect.signature(func).return_annotation, desc=None)]


def get_return_description(func: Callable) -> List[ReturnParameter]:
    """
    Get return description from callable

    Parameters
    ----------
    func : callable
        function to extract return signature from

    Returns
    -------
    return_description

    """

    # Try getting annotation from function signature]

    rv = (_get_return_desc_wrapper(func)
          or _get_return_desc_docstring(func)
          or _get_return_desc_annotation(func)
          or _get_default_return_annotation(func))
    return rv


def assert_subclass(obj, cls):
    if VALIDATE_SUBCLASSES:
        if not isinstance(obj, cls):
            raise MldagException(
                "Expected subclass of {}; got {}".format(cls.__name__,
                                                         obj.__class__.__name__))


def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    cls = getattr(module, class_name)
    return cls


def try_create_dir(folder):
    if folder is not None and not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except FileNotFoundError:
            raise FileNotFoundError("Could not create {} folder because {} does not exist".format(
                folder,
                os.path.split(folder)[0])) from None


# Cleanup empty meta directories
def cleanup_empty_dirs(folder):
    for d in os.listdir(folder):
        d = os.path.join(folder, d)
        if os.path.isdir(d) and len(os.listdir(d)) == 0:
            shutil.rmtree(d)


def convert_to_list(x):
    if isinstance(x, set) or isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        return [convert_to_list(i) for i in x]
    if isinstance(x, dict):
        return {k: convert_to_list(v) for k, v in x.items()}
    return x
