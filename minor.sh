#!/usr/bin/env bash

# Bump version number
_LAST_VERSION=$(git tag -l | awk '/./{line=$0} END{print line}')
IFS='.'
read -r -a LAST_VERSION <<< "${_LAST_VERSION}"
IFS=' '
NEW_VER=$((${LAST_VERSION[1]}+1))
NEW_VER=${LAST_VERSION[0]}.${NEW_VER}.${LAST_VERSION[2]}
PYVERSION="version = '$( echo "${NEW_VER}" | sed 's/v//g')'"

# Commit to github
echo ${PYVERSION} > dask_pipes/version.py

git add -A
git commit -m "$1"
git tag -a "${NEW_VER}" -m "$1"

# Setup package to update git_version
pip install -e .
