import yaml
from pkg_resources import resource_string, resource_listdir

__all__ = ['get', 'use', 'current', 'available']

DEFAULT_STYLE = 'default'
ACTIVE_STYLE = None

available = [i for i in resource_listdir(__name__, '') if i[0] != '_']


def get(name):
    return yaml.load(resource_string(__name__, '{}.yml'.format(name)), Loader=yaml.SafeLoader)['style']


def use(name):
    global ACTIVE_STYLE
    ACTIVE_STYLE = get(name)


def current():
    return ACTIVE_STYLE


use(DEFAULT_STYLE)
