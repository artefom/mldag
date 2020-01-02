from .nodes import *  # noqa F401
from .pipeline import *  # noqa F401
from . import mixins  # noqa F401
from . import base  # noqa F401
from . import column_selection  # noqa F401
from . import pipes  # noqa F401
from . import exceptions  # noqa F401

import opentracing

_tracer = opentracing.Tracer()


def init_jaeger(host='127.0.0.1', port='5775'):
    global _tracer
    from jaeger_client import Config
    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'local_agent': {
                'reporting_host': host,
                'reporting_port': port,
            },
            'logging': True,
        },
        service_name='pipeline',
        validate=True,
    )
    _tracer = config.initialize_tracer()
