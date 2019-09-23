## Installation

This package contains some custom c extensions that require the `gfortran` compiler and some libraries. In particular it requires the GNU scientific library (GSL) be installed and available in the build environment.

There are two ways of installing this package. The first one relies on conda to get the required libraries and compilers. As this code is not provide as a conda build recipe we ship a custom `install.sh` bash script to simplify the installation. Simply use

```bash
./script/install.sh
```

If you want to install the package within some conda environment use

```bash
./script/install.sh -n conda_environment_name
```

If you want to install the package in development mode (changes made to the python source code will have an immediate impact without having to reinstall) use

```bash
./script/install.sh -d
```

The second way of installing this package is by doing all of the work manually.

1. Install `make` and `gfortran`.
2. Install the gsl libraries and make sure they are available in the `$LD_LIBRAY_PATH` or `$DYLD_LIBRARY_PATH`, and also that the respective header files are includable.
3. Download the package into a folder and `cd` into it.
4. Install the python requirements with `pip install -r requirements.txt`
5. Run either `python setup.py install` or `pip install .` or `pip install -e .` to install the `leakyIntegrator` package.
