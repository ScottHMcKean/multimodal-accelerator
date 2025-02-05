"""
setup.py configuration script describing how to build and package this project.

This file is primarily used by the setuptools library and typically should not
be executed directly. See README.md for how to deploy, test, and run
the MAUD project.
"""
from setuptools import setup, find_packages

import sys
sys.path.append('./maud')

import datetime
import maud

setup(
    name="maud",
    # We use timestamp as Local version identifier (https://peps.python.org/pep-0440/#local-version-identifiers.)
    # to ensure that changes to wheel package are picked up when used on all-purpose clusters
    version=maud.__version__ + "+" + datetime.datetime.utcnow().strftime("%Y%m%d.%H%M%S"),
    url="https://databricks.com",
    author="scott.mckean@databricks.com",
    description="wheel file based on maud",
    packages=find_packages(where='./maud'),
    package_dir={'': 'maud'},
    entry_points={
        "packages": [
            "main=maud.main:main"
        ]
    },
    install_requires=[
        # Dependencies in case the output wheel file is used as a library dependency.
        # For defining dependencies, when this package is used in Databricks, see:
        # https://docs.databricks.com/dev-tools/bundles/library-dependencies.html
        "setuptools"
    ],
)