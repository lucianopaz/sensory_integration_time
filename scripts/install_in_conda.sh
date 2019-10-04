#!/usr/bin/env bash

set -ex

# Parse input
usage() { echo "Usage: $0 [-n conda environment onto which install] [-d flag to indicate whether to install the package in pip development mode]" 1>&2; exit 1; }
TARGET_ENV=$CONDA_DEFAULT_ENV
DEVELOPMENT_MODE=1

while getopts ":n:d" o; do
  case "${o}" in
    n)
      TARGET_ENV=${OPTARG}
      ;;
    d)
      DEVELOPMENT_MODE=0
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

# Test if target environment exits
if [[ -z $TARGET_ENV ]]
then
  MSG="\n
       Your shell is not in an active conda environment\n
       and no conda target environment was specificied through option -n.\n
       Please either activate an environment\n
       or specify the target environment explicitly.";
  echo -e $MSG;
  exit 1;
fi


if [[ -z $OS ]]
then
  if [[ $(uname -s) == Linux ]]
  then
    COMPILER=gfortran_linux-64
    CHANNEL=anaconda
  else
    COMPILER=gfortran_osx-64
    CHANNEL=anaconda
  fi
else
  COMPILER=m2w64-toolchain
  CHANNEL=msys2
fi

if [[ $(conda env list | grep -c "^$TARGET_ENV[[:space:]]") ]]
then
  conda create --yes -n $TARGET_ENV python=3.7 gsl
  conda install --yes -n $TARGET_ENV -c $CHANNEL $COMPILER
else
  conda install --yes -n $TARGET_ENV gsl
  conda install --yes -n $TARGET_ENV -c $CHANNEL $COMPILER
fi

# Get the scripts directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ -z $OS ]]
then
  PATHSEP=/
  ENV_VARS=env_vars.sh
else
  PATHSEP=\\
  ENV_VARS=env_vars.bat
fi

# Update the environment's activation env_vars scripts
eval $DIR${PATHSEP}add_env_vars.sh

# If $TARGET_ENV is not active, attempt to activate it
if [[ $TARGET_ENV != $CONDA_DEFAULT_ENV ]]
then
  eval $(conda shell.bash hook)
  source activate $TARGET_ENV
fi

TARGET_ENV_DIR=$(conda env list | $DIR${PATHSEP}get_env_prefix.py $TARGET_ENV)
export GSL_HEADER_DIRECTORY="$TARGET_ENV_DIR${PATHSEP}include"
export GSL_LIBRARY_DIRECTORY="$TARGET_ENV_DIR${PATHSEP}lib"

# Install requirements and package
pip install -r $DIR${PATHSEP}..${PATHSEP}requirements.txt
if [[ $DEVELOPMENT_MODE ]]
then
  pip install -r $DIR${PATHSEP}..${PATHSEP}requirements-dev.txt
  pip install -e $DIR${PATHSEP}..
else
  pip install $DIR${PATHSEP}..
fi