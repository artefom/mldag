import importlib
import os
import io
import logging
from setuptools import setup, find_packages

logger = logging.getLogger(__name__)

# Kept manually in sync with dask_pipes.__version__
# noinspection PyUnresolvedReferences
spec = importlib.util.spec_from_file_location("dask_pipes.version",
                                              os.path.join("dask_pipes", 'version.py'))
# noinspection PyUnresolvedReferences
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
version = mod.version

try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''


def do_setup():
    setup(
        name="dask_pipes",
        version=version,
        packages=find_packages(exclude=['tests', 'tests.*', 'home']),
        install_requires=[
            'begins',
        ],
        zip_safe=False,
        author="Artyom Fomenko",
        description="Generic Project for tabular data preprocessing and fitting using pytorch",
        long_description=long_description,
        license='MIT License',
    )


if __name__ == "__main__":
    do_setup()
