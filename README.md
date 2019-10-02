# `sensory_integration_time`
[![Build Status](https://dev.azure.com/lucianopazneuro/lucianopazneuro/_apis/build/status/lucianopaz.sensory_integration_time?branchName=master)](https://dev.azure.com/lucianopazneuro/lucianopazneuro/_build/latest?definitionId=3&branchName=master)

This package constains the source code used to fit the leaky integration model for tactile perception used in the Toso et al (submitted).

## Installation

This package contains a custom c extension that uses a fortran library (fortran source codes are included in this package). In order to install, your system must have
1. A fortran compiler
2. The GNU scientific library (GSL) must be installed and available in the build environment.

Once you have those, you should be able to simply install this package by downloading its source code and running

```bash
pip install path_to_downloaded_package
```

To try to install everything in a single go, you can try to use `conda` like this:

```
conda create -n env_name python=3.7 gsl some_fortran_compiler_or_toolchain
./scripts/add_env_vars.sh -n env_name
conda activate env_name
pip install -r path_to_downloaded_package/requirements.txt
pip install path_to_downloaded_package
```

Beware that `conda` can mess up pip on macos, the fortran compilers on windows are in non-default channels and once gsl is installed, its headers and lib directories are not added to environment variable. The bash script `add_env_vars.sh`, attempts to solve the last problem mentioned, the rest is up to you...

