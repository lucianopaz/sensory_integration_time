# Installation

This package requires some custom c extensions that require `gfortran` and some libraries. In particular it requires the GNU scientific library (GSL) be installed and available in the build environment. If you plan to install this package using conda, follow these instructions:

1. Create the environment where you want to install the package.
2. Create or edit the `env_vars.sh` script for the activation and deactivation of the envirnoment. If you don't know what this means, read [this section on conda environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux). What we want is to have conda include the `gsl` header files and shared libraries that it will install, so that our extension can be built correctly. You must do the following edits:

### In `activate.d/env_vars.sh` place
``` bash
#!/bin/sh

### Any pre-exiting content goes here ###

_OLD_CINCLUDE_PATH=$C_INCLUDE_PATH
_OLD_CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/lpaz/anaconda3/include/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/lpaz/anaconda3/include/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lpaz/anaconda3/lib
```

### In `deactivate.d/env_vars.sh` place
```bash
#!/bin/sh

### Any pre-exiting content goes here ###

export C_INCLUDE_PATH=$_OLD_CINCLUDE_PATH
export CPLUS_INCLUDE_PATH=$_OLD_CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH

unset _OLD_CINCLUDE_PATH
unset _OLD_CPLUS_INCLUDE_PATH
unset _OLD_LD_LIBRARY_PATH
```

3. Activate the environment you just built and configured
4. `conda install gfortran gsl`
5. Download the package into a folder and `cd` into it.
6. Run either `python setup.py install` or `pip install .` or `pip install -e .` to install the `leakyIntegrator` package.
