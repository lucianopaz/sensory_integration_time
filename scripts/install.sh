#!/usr/bin/env bash

set -ex

# Parse input
usage() { echo "Usage: $0 [-n conda environment onto which install] [-d flag to indicate whether to install the package in pip development mode]" 1>&2; exit 1; }
TARGET_ENV=$CONDA_DEFAULT_ENV

while getopts ":n:d" o; do
  case "${o}" in
    n)
      TARGET_ENV=${OPTARG}
      ;;
    d)
      DEVELOPMENT=1
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

# Get the scripts directory and the short uname
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
UNAME=$(uname -s)
if [[ $UNAME == Darwin ]] || [[ $UNAME == Linux ]]
then
  PATHSEP=/
else
  PATHSEP=\\
fi

# Get the conda environment path
conda_env_root_dir=$(dirname $CONDA_PREFIX | xargs basename)
if [[ $conda_env_root_dir == envs ]]
then
  CONDA_ENVS_DIR=$(dirname $CONDA_PREFIX)
  NOT_IN_CONDA_ROOT_ENV=1
else
  CONDA_ENVS_DIR="$CONDA_PREFIX${PATHSEP}envs"
fi
if [[ $TARGET_ENV != $CONDA_DEFAULT_ENV ]]
then
  if [[ $TARGET_ENV != "base" ]]
  then
    TARGET_ENV_DIR=$CONDA_ENVS_DIR${PATHSEP}$TARGET_ENV
  else
    TARGET_ENV_DIR=$(dirname $CONDA_ENVS_DIR)
  fi
else
  TARGET_ENV_DIR=$CONDA_PREFIX
fi

# Create target conda environment and install requirements
if [[ ! -z $OS ]]
then
  # On windows we need to install gnu make and gfortran using chocolatey and cygwin
  choco install make
  CONDA_INSTALLS="python=3.7 gsl pip"
else
  CONDA_INSTALLS="python=3.7 gsl pip make"
fi

if [[ $(conda env list | grep -c "^$TARGET_ENV[[:space:]]") ]]
then
  echo "Will create a conda environment $TARGET_ENV"
  conda create -n $TARGET_ENV -y $CONDA_INSTALLS
else
  echo "Will install required packages to $TARGET_ENV"
  conda install -n $TARGET_ENV -y $CONDA_INSTALLS
fi

# We activate the environment immediately
source activate $TARGET_ENV

# Platform dependent installs
# Ugly hack because we are not using conda build
if [[ $UNAME == Darwin ]]
then
  conda install -y -n $TARGET_ENV -c conda-forge gfortran_osx-64
elif [[ $UNAME == Linux ]]
then
  conda install -y -n $TARGET_ENV gfortran_linux-64  # This adds the FC environment variable to point to gfortran
fi

# Make the fortran shared library and copy it to the environment's lib
cd $DIR${PATHSEP}..${PATHSEP}leakyIntegrator${PATHSEP}src
make
mkdir -p $TARGET_ENV_DIR${PATHSEP}lib
cp libincgamNEG.so $TARGET_ENV_DIR${PATHSEP}lib
cp libincgamNEG.a $TARGET_ENV_DIR${PATHSEP}lib

# Install requirements and the package
cd $DIR${PATHSEP}..
pip install --upgrade pip
pip install -r requirements.txt
if [[ -z $DEVELOPMENT ]]
then
  pip install -v .
else
  pip install -r requirements-dev.txt
  pip install -v -e .
fi

echo "Successfully installed package!"