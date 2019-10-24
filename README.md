Generic Neural Networks
============================

Generic Project for tabular data preprocessing and fitting using pytorch

Usage
============================

To install project in editable mode (can be edited after installation locally),
run cd `genn && pip install -e .`

After that, Generic Neural Networks can be imported inside any python code as
```python
import genn
# Generic Neural Networks creates configuration file genn.cfg from template (default_genn.cfg
# inside current working directory
# You can now edit the genn.cfg to change the configuration and rerun script

# Import main entrypoint
from genn.run import run

# Run
run()
```

Project configuration
============================

Project uses genn.cfg file for configuration, which can be overridden using environment variables.
The file is divied into several sections, grouping configuration parameters for convenience.

Each section can be acessed inside code as

```python
from genn.configuration import conf

# Get value from file or environment variable
conf.get('section_name', 'variable name')
```

Values in configuration must be specified without quotes

To override default variables in .cfg file one can specify environment variables as
GENN__SECTION__VALUE=value

On startup Generic Neural Networks will automatically parse those and use them instead of values from .cfg file 

See genn/config_templates/default_genn.cfg for detailed info about configuration

Docker execution
============================

Project comes with pre-configured docker file which will copy source code, install all libraries in cache-friendly fashion and run help message: `genn --help`
