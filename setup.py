#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Setup script for the leakyIntegrator package related to the publication ...
It involves the use of a Makefile during the build phase, and it does so
by calling subprocess.Popen with shell=True. This means that you must
trust this source not to have been tampered with, or install without sudo.
For a more detailed explanation of why the build_ext class was overloaded
refer to https://stackoverflow.com/a/48641638/7253166

Author: Luciano Paz
Year: 2018

"""

# We need to import setuptools first to get numpy.distutils.setup to point to
# the setuptools implementation instead of distutils' implementation
import os
import codecs
import re
import setuptools
from numpy.distutils.misc_util import Configuration
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
SOURCE_FILE_PATH = os.path.join("sensory_integration_time", "src")

#extra_compile_args = [
#    "-I{}".format(d) for d in list(
#        set(
#            os.environ.get("C_INCLUDE_PATH", "").split(":") +
#            os.environ.get("CPLUS_INCLUDE_PATH", "").split(":")
#        )
#    ) if d
#]
#extra_link_args = [
#    "-L{}".format(d) for d in list(
#        set(
#            os.environ.get("LD_LIBRARY_PATH", "").split(":") +
#            os.environ.get("DYLD_LIBRARY_PATH", "").split(":")
#        )
#    ) if d
#]
include_dirs = [
    d for d in list(
        set(
            os.environ.get("C_INCLUDE_PATH", "").split(":") +
            os.environ.get("CPLUS_INCLUDE_PATH", "").split(":")
        )
    ) if d
]
library_dirs = [
    d for d in list(
        set(
            os.environ.get("LD_LIBRARY_PATH", "").split(":") +
            os.environ.get("DYLD_LIBRARY_PATH", "").split(":")
        )
    ) if d
]


description = "Leaky Integration underlying tactile perception interface for fitting and prediction"


def read(*parts):
    with codecs.open(os.path.join(PROJECT_ROOT, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def configuration():
    config = Configuration(
        package_name="sensory_integration_time",
        parent_name=None
    )
    config.add_include_dirs(*include_dirs)
    config.add_library(
        name="incgamNEG",
        sources=[
            os.path.join(SOURCE_FILE_PATH, "setprecision.f90"),
            os.path.join(SOURCE_FILE_PATH, "someconstants.f90"),
            os.path.join(SOURCE_FILE_PATH, "gammaError.f90"),
            os.path.join(SOURCE_FILE_PATH, "incgamNEG.f90")
        ],
        extra_f90_compile_args=["-O3", "-fPIC", "-Wc-binding-type"]
    )
    config.add_extension(
        name="leaky_integral_calculator",
        language="c",
        sources=["sensory_integration_time/src/leaky_integral_calculator.c"],
        libraries=[
            "m",
            "dl",
            "gsl",
            "gslcblas",
            "quadmath",
        ],
        library_dirs=library_dirs,
        extra_link_args=["-l:libincgamNEG.a"]
    )
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        configuration=configuration,
        version=find_version("sensory_integration_time", "__init__.py"),
        packages=setuptools.find_packages(),
        setup_requires=["numpy>=1.13"],
        install_requires=get_requirements(),
        # metadata for upload to PyPI
        author="Luciano Paz",
        author_email="lpaz@sissa.it",
        description=description,
        long_description=get_long_description(),
        license="MIT",
        keywords="Leaky integrator perceptual decision making model, "
        "Stochastic dynamics, "
        "Ornstein Ulhenbeck process, "
        "Parametric model fit",
        url="https://github.com/lucianopaz/sensory_integration_time",
    )