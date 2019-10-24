import sys
from collections import OrderedDict
import copy
import os
import pathlib
from configparser import ConfigParser, _UNSET, NoOptionError, NoSectionError

# Configure logging
import logging

root_log = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(fmt)
root_log.addHandler(handler)
root_log.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)


class ConfigurationException(Exception):
    pass


def expand_env_var(env_var):
    """
    Expands (potentially nested) env vars by repeatedly applying
    `expandvars` and `expanduser` until interpolation stops having
    any effect.
    """
    if not env_var:
        return env_var
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated


def _read_default_config_file(file_name: str) -> str:
    templates_dir = os.path.join(os.path.dirname(__file__), 'config_templates')
    file_path = os.path.join(templates_dir, file_name)
    with open(file_path, encoding='utf-8') as file:
        return file.read()


class EnvConfigParser(ConfigParser):

    # This method transforms option names on every read, get, or set operation.
    # This changes from the default behaviour of ConfigParser from lowercasing
    # to instead be case-preserving
    def optionxform(self, optionstr: str) -> str:
        return optionstr

    def __init__(self, default_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config_defaults = ConfigParser(*args, **kwargs)
        if default_config is not None:
            self.config_defaults.read_string(default_config)

    @staticmethod
    def _env_var_name(section, key):
        return 'GENN__{S}__{K}'.format(S=section.upper(), K=key.upper())

    def _get_env_var_option(self, section, key):
        # must have format GENN__{SECTION}__{KEY} (note double underscore)
        env_var = self._env_var_name(section, key)
        if env_var in os.environ:
            return expand_env_var(os.environ[env_var])

    def get(self, section, key, **kwargs):
        section = str(section).lower()
        key = str(key).lower()

        # first check environment variables
        option = self._get_env_var_option(section, key)
        if option is not None:
            return option

        # ...then the config file
        if super().has_option(section, key):
            # Use the parent's methods to get the actual config here to be able to
            # separate the config from default config.
            return expand_env_var(
                super().get(section, key, **kwargs))

        # ...then the default config
        if self.config_defaults.has_option(section, key) or 'fallback' in kwargs:
            return expand_env_var(
                self.config_defaults.get(section, key, **kwargs))

        else:
            log.warning(
                "section/key [%s/%s] not found in config", section, key
            )

            raise ConfigurationException(
                "section/key [{section}/{key}] not found "
                "in config".format(section=section, key=key))

    def getboolean(self, section, key, **kwargs):
        val = str(self.get(section, key, **kwargs)).lower().strip()
        if '#' in val:
            val = val.split('#')[0].strip()
        if val in ('t', 'true', '1'):
            return True
        elif val in ('f', 'false', '0'):
            return False
        else:
            raise ValueError(
                'The value for configuration option "{}:{}" is not a '
                'boolean (received "{}").'.format(section, key, val))

    def getint(self, section, key, **kwargs):
        return int(self.get(section, key, **kwargs))

    def getfloat(self, section, key, **kwargs):
        return float(self.get(section, key, **kwargs))

    def has_option(self, section, option):
        try:
            # Using self.get() to avoid reimplementing the priority order
            # of config variables (env, config, cmd, defaults)
            # UNSET to avoid logging a warning about missing values
            self.get(section, option, fallback=_UNSET)
            return True
        except (NoOptionError, NoSectionError):
            return False

    def remove_option(self, section, option, remove_default=True):
        """
        Remove an option if it exists in config from a file or
        default config. If both of config have the same option, this removes
        the option in both configs unless remove_default=False.
        """
        if super().has_option(section, option):
            super().remove_option(section, option)

        if self.config_defaults.has_option(section, option) and remove_default:
            self.config_defaults.remove_option(section, option)

    def getsection(self, section):
        """
        Returns the section as a dict. Values are converted to int, float, bool
        as required.

        :param section: section from the config
        :rtype: dict
        """
        if (section not in self._sections and
                section not in self.config_defaults._sections):
            return None

        _section = copy.deepcopy(self.config_defaults._sections[section])

        if section in self._sections:
            _section.update(copy.deepcopy(self._sections[section]))

        section_prefix = 'GENN__{S}__'.format(S=section.upper())
        for env_var in sorted(os.environ.keys()):
            if env_var.startswith(section_prefix):
                key = env_var.replace(section_prefix, '').lower()
                _section[key] = self._get_env_var_option(section, key)

        for key, val in _section.items():
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() in ('t', 'true'):
                        val = True
                    elif val.lower() in ('f', 'false'):
                        val = False
            _section[key] = val
        return _section

    def as_dict(
            self, display_source=False, display_sensitive=False, raw=False,
            include_env=True, include_cmds=True):
        """
        Returns the current configuration as an OrderedDict of OrderedDicts.

        :param display_source: If False, the option value is returned. If True,
            a tuple of (option_value, source) is returned. Source is either
            'genn.cfg', 'default', 'env var', or 'cmd'.
        :type display_source: bool
        :param display_sensitive: If True, the values of options set by env
            vars and bash commands will be displayed. If False, those options
            are shown as '< hidden >'
        :type display_sensitive: bool
        :param raw: Should the values be output as interpolated values, or the
            "raw" form that can be fed back in to ConfigParser
        :type raw: bool
        :param include_env: Should the value of configuration from GENN__
            environment variables be included or not
        :type include_env: bool
        :param include_cmds: Should the result of calling any *_cmd config be
            set (True, default), or should the _cmd options be left as the
            command to run (False)
        :type include_cmds: bool
        """
        cfg = {}
        configs = [
            ('default', self.config_defaults),
            ('genn.cfg', self),
        ]

        for (source_name, config) in configs:
            for section in config.sections():
                sect = cfg.setdefault(section, OrderedDict())
                for (k, val) in config.items(section=section, raw=raw):
                    if display_source:
                        val = (val, source_name)
                    sect[k] = val

        # add env vars and overwrite because they have priority
        if include_env:
            for ev in [ev for ev in os.environ if ev.startswith('GENN__')]:
                try:
                    _, section, key = ev.split('__', 2)
                    opt = self._get_env_var_option(section, key)
                except ValueError:
                    continue
                if not display_sensitive and ev != 'GENN__CORE__UNIT_TEST_MODE':
                    opt = '< hidden >'
                elif raw:
                    opt = opt.replace('%', '%%')
                if display_source:
                    opt = (opt, 'env var')

                section = section.lower()
                # if we lower key for kubernetes_environment_variables section,
                # then we won't be able to set any Generic Neural Networks environment
                # variables. Generic Neural Networks only parse environment variables starts
                # with GENN_. Therefore, we need to make it a special case.
                if section != 'kubernetes_environment_variables':
                    key = key.lower()
                cfg.setdefault(section, OrderedDict()).update({key: opt})

        return cfg


def get_home_dir():
    rv = expand_env_var(os.environ.get('GENN_HOME', os.getcwd()))
    log.info("Using project home directory: {}".format(rv))
    return rv


def get_config(home_dir):
    if 'GENN_CONFIG' not in os.environ:
        return os.path.join(home_dir, 'genn.cfg')
    return expand_env_var(os.environ['GENN_CONFIG'])


def parameterized_config(template):
    all_vars = {k: v for d in [globals(), locals()] for k, v in d.items()}
    return template.format(**all_vars)


# Setting GENN_HOME and GENN_CONFIG from environment variables, using
# "~/genn" and "$GENN_HOME/genn.cfg"
# respectively as defaults.

HOME_DIR = get_home_dir()
CONFIG_PATH = get_config(HOME_DIR)
pathlib.Path(os.path.split(CONFIG_PATH)[0]).mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG = _read_default_config_file('default_genn.cfg')

if not os.path.isfile(CONFIG_PATH):
    log.info(
        'Creating new genn config file in: %s',
        CONFIG_PATH
    )
    with open(CONFIG_PATH, 'w') as file:
        cfg = parameterized_config(DEFAULT_CONFIG)
        file.write(cfg)

log.info("Reading the config from %s", CONFIG_PATH)

conf = EnvConfigParser(default_config=parameterized_config(DEFAULT_CONFIG))

conf.read(CONFIG_PATH)

# Set logging level from configuration

root_log.setLevel(getattr(logging, conf.get('logging', 'level').upper()))
