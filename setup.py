import importlib
import io
import logging
import os
import pathlib
from typing import List

from setuptools import setup, find_packages

PARENT = pathlib.Path(__file__).parent

logger = logging.getLogger(__name__)

# Kept manually in sync with mldag.__version__
# noinspection PyUnresolvedReferences
spec = importlib.util.spec_from_file_location("mldag.version",
                                              os.path.join("mldag", 'version.py'))
# noinspection PyUnresolvedReferences
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
version = mod.version

try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''


def read_requirements(path: str) -> List[str]:
    file_path = PARENT / path
    with open(file_path) as f:
        return f.read().split('\n')


def do_setup():
    setup(
        name="mldag",
        version=version,
        packages=find_packages(exclude=['tests', 'tests.*', 'home']),
        install_requires=read_requirements('requirements.txt'),
        zip_safe=False,
        author="Artyom Fomenko",
        description="Generic Project for tabular data preprocessing and fitting using pytorch",
        long_description=long_description,
        license='MIT License',
    )


if __name__ == "__main__":
    do_setup()
