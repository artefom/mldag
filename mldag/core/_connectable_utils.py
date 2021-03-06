import inspect
from collections import namedtuple
from types import MethodType
from typing import List, Tuple, Dict, Any, Union, Iterable

import numpydoc.docscrape

from ..exceptions import MldagException
from ..utils import replace_signature, INSPECT_EMPTY_PARAMETER, InputParameter

__all__ = ['MLDagInput', 'MLDagOutput', 'get_input_signature',
           'set_fit_signature', 'set_transform_signature',
           'reset_fit_signature', 'reset_transform_signature',
           'getcallargs_inverse', 'validate_fit_transform',
           'ARGS_PARAM_NAME', 'KWARGS_PARAM_NAME']

# Valid order of parameter types (kind+default flag)
ARG_ORDER = [
    'pos_only_no_default',
    'pos_only_w_default',
    'pos_or_key_no_default',
    'pos_or_key_w_default',
    'var_pos',
    'key_only_no_default',
    'key_only_w_default',
    'var_key'
]

ARGS_PARAM_NAME = 'args'
KWARGS_PARAM_NAME = 'kwargs'


def validate_fit_transform(name, attrs,  # noqa: C901
                           allow_default=True,
                           obligatory_variadic=False):
    if 'fit' in attrs:
        f_sign = inspect.signature(attrs['fit'])
    else:
        raise MldagException("Class {} does not have fit method".format(name))
    if 'transform' in attrs:
        if f_sign.parameters != inspect.signature(attrs['transform']).parameters:
            raise MldagException("Class {} fit parameters does not match transform parameters".format(name))
    else:
        raise MldagException("Class {} does not have transform method".format(name))

    if obligatory_variadic:
        var_pos = None
        var_kw = None
        for param in f_sign.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos = param
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw = param
        if var_pos is None or var_kw is None:
            raise MldagException(
                "{name}.fit and {name}.transform must "
                "have variadic positional and keyword arguments "
                "(*args and **kwargs)".format(name=name))
    if not allow_default:
        for param in list(f_sign.parameters.values())[1:]:  # Skip 'self'
            if (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                or param.kind == inspect.Parameter.POSITIONAL_ONLY) \
                    and param.default != INSPECT_EMPTY_PARAMETER:
                msg = ("{name}.fit and {name}.transform are not "
                       "allowed to have default arguments ({param.name} = {param.default}). "
                       "Remove default values".format(name=name, param=param))
                raise MldagException(msg)


class MLDagInput(namedtuple("MLDagInput", ['name',
                                           'downstream_slot',
                                           'downstream_node',
                                           'default',
                                           'kind',
                                           'type',
                                           'desc'])):
    """
    Analog of inspect.Parameter, but includes reference to downstream node and it's slot
    """

    def __str__(self):
        arg_param = inspect.Parameter(name=self.name, kind=self.kind, default=self.default,
                                      annotation=self.type)
        return str(arg_param)

    def __repr__(self):
        return '<{}>'.format(str(self))


MLDagOutput = namedtuple("MLDagOutput", ['name', 'upstream_node', 'upstream_slot', 'type', 'desc'])


def getcallargs_inverse(func, **callargs) -> Tuple[Iterable[Any], Dict[str, Any]]:  # noqa C901
    """
    Inverse function of inspect.getcallargs

    Transforms dictionary of values to (*args, **kwargs)

    Parameters
    ----------
    func
    callargs

    Returns
    -------

    """
    sign = inspect.signature(func)
    args = list()
    kwargs = dict()

    missing_positional = False

    for parameter in sign.parameters.values():
        try:
            values = callargs[parameter.name]
        except KeyError:
            if (not missing_positional and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or
                    parameter.kind == inspect.Parameter.POSITIONAL_ONLY):
                missing_positional = True
            continue

        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(values)
        elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(values)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            if missing_positional:
                raise MldagException("Cannot fill variadic positional {}, "
                                     "since preceding positional "
                                     "parameters are missing".format(parameter.name))
            try:
                args.extend(values)
            except TypeError:
                raise MldagException(
                    "Tried to pass non-terable parameter to variadic positional argument '{}'".format(
                        parameter.name)) from None
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            try:
                kwargs = {**kwargs, **values}
            except TypeError:
                raise MldagException(
                    "Tried to pass non-mapping parameter to variadic key-word argument '{}'".format(
                        parameter.name)) from None
        else:
            kwargs[parameter.name] = values
    return tuple(args), kwargs


def set_fit_signature(obj, sign: inspect.Signature, doc=None):
    obj.fit = MethodType(replace_signature(obj.__class__.fit, sign, doc=doc), obj)


def set_transform_signature(obj, sign: inspect.Signature, doc=None):
    obj.transform = MethodType(replace_signature(obj.__class__.transform, sign, doc=doc), obj)


def reset_fit_signature(obj):
    obj.fit = MethodType(obj.__class__.fit, obj)


def reset_transform_signature(obj):
    obj.transform = MethodType(obj.__class__.transform, obj)


def _split_signature_by_kind(parameters) -> Dict[str, List[inspect.Parameter]]:
    """
    Get lists of parameters splitted to dictionary by their kind
    Parameter kind names:
    pos_only_no_default
    pos_only_w_default
    pos_or_key_no_default
    var_pos
    pos_or_key_w_default
    key_only_no_default
    key_only_w_default
    var_key

    Parameters
    ----------
    parameters

    Returns
    -------
    signature_params : dict
        { kind_name: [parameter1, parameter2, ...], ...}
    """
    params_by_kind = dict()
    params_by_kind['pos_only_no_default'] = list()
    params_by_kind['pos_only_w_default'] = list()
    params_by_kind['pos_or_key_no_default'] = list()
    params_by_kind['var_pos'] = list()
    params_by_kind['pos_or_key_w_default'] = list()
    params_by_kind['key_only_no_default'] = list()
    params_by_kind['key_only_w_default'] = list()
    params_by_kind['var_key'] = list()
    for inp in parameters:
        inp: inspect.Parameter
        if inp.kind == inspect.Parameter.POSITIONAL_ONLY:
            if inp.default == INSPECT_EMPTY_PARAMETER:
                params_by_kind['pos_only_no_default'].append(inp)
            else:
                params_by_kind['pos_only_w_default'].append(inp)
        if inp.kind == inspect.Parameter.VAR_POSITIONAL:
            params_by_kind['var_pos'].append(inp)
        if inp.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if inp.default == INSPECT_EMPTY_PARAMETER:
                params_by_kind['pos_or_key_no_default'].append(inp)
            else:
                params_by_kind['pos_or_key_w_default'].append(inp)
        if inp.kind == inspect.Parameter.KEYWORD_ONLY:
            if inp.default == INSPECT_EMPTY_PARAMETER:
                params_by_kind['key_only_no_default'].append(inp)
            else:
                params_by_kind['key_only_w_default'].append(inp)
        if inp.kind == inspect.Parameter.VAR_KEYWORD:
            params_by_kind['var_key'].append(inp)
    return params_by_kind


def get_new_docstring(mldag, original_doc, new_parameters: List[inspect.Parameter]):  # noqa: C901
    # Handle numpy docstrings
    if original_doc:
        docstring = numpydoc.docscrape.NumpyDocString(original_doc)
    else:
        docstring = dict()

    params_docstrings_orig = {i.name: i for i in docstring['Parameters']}

    parameter_desc = dict()
    parameter_types = dict()

    for input in mldag.inputs:
        if input.desc and input.desc != inspect._empty:
            if input.name not in parameter_desc:
                parameter_desc[input.name] = list()
            if isinstance(input.desc, list):
                parameter_desc[input.name].extend(input.desc)
            elif isinstance(input.desc, str):
                parameter_desc[input.name].append(input.desc)
            else:
                raise ValueError("Invalid description: {}".format(input.desc))

        if input.type != inspect._empty:
            if input.name not in parameter_types:
                parameter_types[input.name] = list()
            parameter_types[input.name].append(input.type)

    new_docstring_parameters = list()
    for param in new_parameters:
        if param.name in parameter_types or param.name in parameter_desc:
            new_param_type = ', '.join({
                str(i.__name__ if isinstance(i, type) else i)
                for i in parameter_types.get(param.name, [])
            })
            new_param_desc = parameter_desc.get(param.name, [])

            new_docstring_parameters.append(numpydoc.docscrape.Parameter(param.name, new_param_type, new_param_desc))
        elif param.name in params_docstrings_orig:
            desc_orig = params_docstrings_orig[param.name].desc
            type_orig = params_docstrings_orig[param.name].type
            new_docstring_parameters.append(numpydoc.docscrape.Parameter(param.name, type_orig, desc_orig))

    docstring['Parameters'] = new_docstring_parameters

    return docstring


def get_input_signature(mldag):  # noqa: C901
    """
    Get list of parameters and their mapping to specific node inputs

    Parameters
    ----------
    mldag
        MLDag to get input from

    Returns
    -------
    signature_parameters
         [parameter1, parameter2, ...]
    params_downstream
        {parameter1.name: [(node,slot_name), ...] }
    fit_docstring
        Fit's docstring
    transform_docstring
        Transform's docstring

    """
    # Split input by kind
    original_signature = list(inspect.signature(mldag.__class__.transform).parameters.values())
    reserved_names = {param.name for param in original_signature}

    # MLDag may contain some custom parameters, take them into account by using original signature
    params_by_kind = _split_signature_by_kind(original_signature)

    if len(mldag.inputs) > 0:
        # Our mldag has some inputs, remove default *args, **kwargs parameters
        # and populate signature with new parameters
        params_by_kind['var_pos'] = list()  # Remove default 'args' parameter
        params_by_kind['var_key'] = list()  # Remove default 'kwargs' parameter
        for k, v in _split_signature_by_kind(mldag.inputs).items():
            # Validate param names
            for param in v:
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    if not param.name == KWARGS_PARAM_NAME:
                        raise ValueError("Variadic keyword arguments must have name '{}'".format(KWARGS_PARAM_NAME))
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    if not param.name == ARGS_PARAM_NAME:
                        raise ValueError("Variadic positional arguments must have name '{}'".format(ARGS_PARAM_NAME))
                if param.name in reserved_names \
                        and param.kind != inspect.Parameter.VAR_POSITIONAL \
                        and param.kind != inspect.Parameter.VAR_KEYWORD:
                    raise MldagException(
                        "Parameter name {} is reserved by mldag's fit signature".format(param.name))
            params_by_kind[k].extend(v)

    def _rename_params(new_name, params: List[Union[inspect.Parameter, MLDagInput]]):
        """
        Returns inspect.parameter or MLDagInput with new name
        Useful for renaming all variadic parameters of sub-nodes to same standard name i.e.: args, kwargs
        to allow them to merge

        Parameters
        ----------
        new_name : str
            New name to assign to parameters
        params : List of inspect.Parameter or MLDagInput
            List of parameters to rename

        Returns
        -------
        """
        new_param_list = list()
        for param in params:
            if isinstance(param, InputParameter):
                new_param_list.append(
                    InputParameter(
                        name=new_name,
                        kind=param.kind,
                        default=param.default,
                        type=param.type,
                        desc=param.desc,
                    )
                )
            if isinstance(param, inspect.Parameter):
                new_param_list.append(
                    inspect.Parameter(
                        name=new_name,
                        kind=param.kind,
                        default=param.default,
                        annotation=param.annotation,
                    )
                )
            if isinstance(param, MLDagInput):
                new_param_list.append(
                    MLDagInput(
                        name=new_name,
                        downstream_slot=param.downstream_slot,
                        downstream_node=param.downstream_node,
                        default=param.default,
                        kind=param.kind,
                        type=param.type,
                        desc=param.desc,
                    )
                )
        return new_param_list

    var_pos_names = set()
    if len(params_by_kind['var_pos']) > 1:
        # If mldag has multiple positional variadic arguments, name them all 'args', so they will merge
        # Since multiple positional variadic parameters are not allowed
        var_pos_names = {param.name for param in params_by_kind['var_pos']}
        if len(var_pos_names) > 1:
            params_by_kind['var_pos'] = _rename_params(ARGS_PARAM_NAME, params_by_kind['var_pos'])

    var_key_names = set()
    if len(params_by_kind['var_key']) > 1:
        # If mldag has multiple variadic keyword arguments, name them all 'kwargs', so they will merge
        # Since multiple keyword variadic parameters are not allowed
        var_key_names = {param.name for param in params_by_kind['var_key']}
        if len(var_key_names) > 1:
            params_by_kind['var_key'] = _rename_params(KWARGS_PARAM_NAME, params_by_kind['var_key'])

    # Set priority of parameter types to omit errors
    # Parameter categories are grouped to allow handling parameters with the same name but different categories
    # Parameters in one group are guaranteed to be unique and in correct order
    params_priority = [['pos_only_w_default',
                        'pos_only_no_default'],
                       ['pos_or_key_w_default',
                        'pos_or_key_no_default',
                        'key_only_no_default'],
                       ['pos_or_key_w_default',
                        'key_only_w_default',
                        'key_only_no_default'],
                       ['pos_or_key_w_default',
                        'key_only_no_default']]

    # Move params up on the hierarchy
    # If the same parameter name has and does not have the default value, just assume it has no default value
    for priority_hierarchy in params_priority:
        for prev_cat, next_cat in zip(priority_hierarchy[:-1], priority_hierarchy[1:]):
            prev_cat_names = {i.name for i in params_by_kind[prev_cat]}
            next_cat_names = {i.name for i in params_by_kind[next_cat]}
            name_inters = prev_cat_names.intersection(next_cat_names)
            params_by_kind[next_cat].extend([i for i in params_by_kind[prev_cat] if i.name in name_inters])
            params_by_kind[prev_cat] = [i for i in params_by_kind[prev_cat] if i.name not in name_inters]

    # Convert list of mixed parameters and MLDagInputs to two dictionaries
    params_downstream: Dict[str, List[Tuple[Any, str]]] = dict()  # Parameter name: Downstream node slot
    new_params_by_kind: Dict[str: List[inspect.Parameter]] = dict()  # Parameter kind: list   of parameters

    for cat, params in params_by_kind.items():
        new_params = list()
        for param in params:
            if isinstance(param, MLDagInput):
                param_name = param.name
                if param_name not in params_downstream:
                    params_downstream[param_name] = list()
                params_downstream[param_name].append((param.downstream_node, param.downstream_slot))
                new_params.append(inspect.Parameter(
                    name=param.name,
                    kind=param.kind,
                    default=param.default,
                    annotation=param.type
                ))
            else:
                new_params.append(param)
        new_params_by_kind[cat] = new_params
    params_by_kind = new_params_by_kind

    # Drop duplicates in each category, handling presence of default values correctly
    new_params_by_kind = dict()
    for cat, params in params_by_kind.items():
        seen_params = dict()
        seen_params_order = list()
        for param in params:
            if param.name in seen_params:
                old_param = seen_params[param.name]
                seen_params[param.name] = inspect.Parameter(
                    name=param.name,
                    kind=param.kind,
                    default=param.default
                    if param.default == old_param.default or param.default is old_param.default
                    else INSPECT_EMPTY_PARAMETER,
                    annotation=param.annotation
                    if param.annotation == old_param.annotation or param.annotation is old_param.annotation
                    else INSPECT_EMPTY_PARAMETER
                )
            else:
                seen_params_order.append(param.name)
                seen_params[param.name] = param
        new_params_by_kind[cat] = [seen_params[i] for i in seen_params_order]
    params_by_kind = new_params_by_kind

    # Check for duplicates in final result
    param_dups = dict()
    for cat, params in params_by_kind.items():
        for param in params:
            param_names = {param.name}
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                param_names.update(var_pos_names)
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                param_names.update(var_key_names)
            for param_name in param_names:
                if param_name in param_dups and param_dups[param_name] != param.kind:
                    raise MldagException("Parameter {} cannot be {} and {} at the same time".format(
                        param_name, param_dups[param_name].name, param.kind.name))
                else:
                    param_dups[param_name] = param.kind

    # Concatenate parameters
    rv: List[inspect.Parameter] = list()
    for cat in ARG_ORDER:
        rv.extend(params_by_kind[cat])

    fit_docstring = get_new_docstring(mldag, mldag.fit.__func__.__doc__, rv)
    transform_docstring = get_new_docstring(mldag, mldag.transform.__func__.__doc__, rv)

    return rv, params_downstream, fit_docstring, transform_docstring
