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
echo $TARGET_ENV

# Get the scripts directory and the short uname
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OS=$(uname -s)

# Get the conda environment path
conda_env_root_dir=$(dirname $CONDA_PREFIX | xargs basename)
if [[ $conda_env_root_dir == envs ]]
then
  CONDA_ENVS_DIR=$(dirname $CONDA_PREFIX)
  NOT_IN_CONDA_ROOT_ENV=1
else
  CONDA_ENVS_DIR="$CONDA_PREFIX/envs"
fi

# Activate target conda environment
if [[ $TARGET_ENV != $CONDA_DEFAULT_ENV ]] && [[ ! -d "$CONDA_ENVS_DIR/$TARGET_ENV" ]] && [[ $TARGET_ENV != "base" ]]
then
  echo "Will create a conda environment $TARGET_ENV"
  conda create -n $TARGET_ENV -y
fi

if [[ $TARGET_ENV != $CONDA_DEFAULT_ENV ]]
then
  if [[ $TARGET_ENV != "base" ]]
  then
    TARGET_ENV_DIR=$CONDA_ENVS_DIR/$TARGET_ENV
  else
    TARGET_ENV_DIR=$(dirname $CONDA_ENVS_DIR)
  fi
else
  TARGET_ENV_DIR=$CONDA_PREFIX
fi

# Update the environment's activation env_vars scripts
mkdir -p $TARGET_ENV_DIR/etc/conda/activate.d
mkdir -p $TARGET_ENV_DIR/etc/conda/deactivate.d

# Check if env_vars exist
if [[ ! -f $TARGET_ENV_DIR/etc/conda/activate.d/env_vars.sh ]]
then
  echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR/etc/conda/activate.d/env_vars.sh
fi
if [[ ! -f $TARGET_ENV_DIR/etc/conda/deactivate.d/env_vars.sh ]]
then
  echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR/etc/conda/deactivate.d/env_vars.sh
fi

# Init the activation commands
HEADER="####### LINES ADDED BY THE INSTALLATION OF THE LEAKYINTEGRATION PACKAGE ######"
ACTIVATION="
_OLD_CINCLUDE_PATH=\$C_INCLUDE_PATH
_OLD_CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH
_OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
_OLD_DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH

export C_INCLUDE_PATH=$TARGET_ENV_DIR/include/:\$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$TARGET_ENV_DIR/include/:\$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$TARGET_ENV_DIR/lib/:\$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$TARGET_ENV_DIR/lib/:\$DYLD_LIBRARY_PATH"
DEACTIVATION="
export C_INCLUDE_PATH=\$_OLD_CINCLUDE_PATH
export CPLUS_INCLUDE_PATH=\$_OLD_CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=\$_OLD_LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=\$_OLD_DYLD_LIBRARY_PATH

unset _OLD_CINCLUDE_PATH
unset _OLD_CPLUS_INCLUDE_PATH
unset _OLD_LD_LIBRARY_PATH
unset _OLD_DYLD_LIBRARY_PATH"

# Check if env_vars was already edited, and if not, edit
if [[ ! $(grep -q "$HEADER" $TARGET_ENV_DIR/etc/conda/activate.d/env_vars.sh) ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR/etc/conda/activate.d/env_vars.sh
  echo -e "$ACTIVATION" >> $TARGET_ENV_DIR/etc/conda/activate.d/env_vars.sh
fi
if [[ ! $(grep -q "$HEADER" $TARGET_ENV_DIR/etc/conda/deactivate.d/env_vars.sh) ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR/etc/conda/deactivate.d/env_vars.sh
  echo -e "$DEACTIVATION" >> $TARGET_ENV_DIR/etc/conda/deactivate.d/env_vars.sh
fi

# conda install requirements
conda install -y -n $TARGET_ENV make gsl pip

# Platform dependent installs
# Ugly hack because we are not using conda build
if [[ $OS == Linux ]]
then
  conda install -y -n $TARGET_ENV -c anaconda gfortran_linux-64
elif [[ $OS == Darwin ]]
then
   conda install -y -n $TARGET_ENV -c conda-forge gfortran_osx-64
fi

# Activate environment
echo "Activating target environment $TARGET_ENV"
set +x
eval "$(conda shell.bash hook)"
conda activate $TARGET_ENV
set -x

# Make the fortran shared library
cd $DIR/../leakyIntegrator/src
make shared

# pip install the package
cd $DIR/..
pip install --upgrade pip
pip install -r requirements.txt
if [[ -z $DEVELOPMENT ]]
then
  pip install .
else
  pip install -e .
fi

echo "Successfully installed package!"