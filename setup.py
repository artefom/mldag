import importlib
import os
import io
import logging
from setuptools import setup, find_packages

logger = logging.getLogger(__name__)

# Kept manually in sync with genn.__version__
# noinspection PyUnresolvedReferences
spec = importlib.util.spec_from_file_location("genn.version",
                                              os.path.join("genn", 'version.py'))
# noinspection PyUnresolvedReferences
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
version = mod.version

try:
    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''


def git_version(version_):
    """
    Return a version to identify the state of the underlying git repo. The version will
    indicate whether the head of the current git-backed working directory is tied to a
    release tag or not : it will indicate the former with a 'release:{version}' prefix
    and the latter with a 'dev0' prefix. Following the prefix will be a sha of the current
    branch head. Finally, a "dirty" suffix is appended to indicate that uncommitted
    changes are present.
    :param str version_: Semver version
    :return: Found genn version in Git repo
    :rtype: str
    """
    try:
        import git
        try:
            repo = git.Repo('.git')
        except git.NoSuchPathError:
            logger.warning('.git directory not found: Cannot compute the git version')
            return ''
    except ImportError:
        logger.warning('gitpython not found: Cannot compute the git version.')
        return ''
    if repo:
        sha = repo.head.commit.hexsha
        if repo.is_dirty():
            return '.dev0+{sha}.dirty'.format(sha=sha)
        # commit is clean
        return '.release:{version}+{sha}'.format(version=version_, sha=sha)
    else:
        return 'no_git_version'


def write_version(filename=os.path.join(*["genn", "git_version"])):
    """
    Write the Semver version + git hash to file, e.g. ".dev0+2f635dc265e78db6708f59f68e8009abb92c1e65".
    :param str filename: Destination file to write
    """
    text = "{}".format(git_version(version))
    with open(filename, 'w') as file:
        file.write(text)


def do_setup():
    write_version()
    setup(
        name="genn",
        version=version,
        packages=find_packages(exclude=['tests', 'tests.*', 'home']),
        install_requires=[
            'begins',
        ],
        scripts=['genn/bin/genn'],
        zip_safe=False,
        author="Artyom Fomenko",
        description="Generic Project for tabular data preprocessing and fitting using pytorch",
        long_description=long_description,
        license='MIT License',
    )


if __name__ == "__main__":
    do_setup()
