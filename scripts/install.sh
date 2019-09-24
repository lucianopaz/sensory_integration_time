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

# Activate target conda environment
if [[ $TARGET_ENV != $CONDA_DEFAULT_ENV ]] && [[ ! -d "$CONDA_ENVS_DIR${PATHSEP}$TARGET_ENV" ]] && [[ $TARGET_ENV != "base" ]]
then
  echo "Will create a conda environment $TARGET_ENV"
  conda create -n $TARGET_ENV -y
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

# Update the environment's activation env_vars scripts
mkdir -p $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d
mkdir -p $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d

# Check if env_vars exist
if [[ ! -f $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}env_vars.sh ]]
then
  echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}env_vars.sh
fi
if [[ ! -f $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}env_vars.sh ]]
then
  echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}env_vars.sh
fi

# Init the activation commands
_HEADER="LINES ADDED BY THE INSTALLATION OF THE LEAKYINTEGRATION PACKAGE"
HEADER="####### $_HEADER ######"
ACTIVATION="
_OLD_CINCLUDE_PATH=\$C_INCLUDE_PATH
_OLD_CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH
_OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
_OLD_DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH

export C_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include${PATHSEP}:\$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include${PATHSEP}:\$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib${PATHSEP}:\$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib${PATHSEP}:\$DYLD_LIBRARY_PATH"
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
if [[ ! $(grep -q "$_HEADER" $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}env_vars.sh) ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}env_vars.sh
  echo -e "$ACTIVATION" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}env_vars.sh
fi
if [[ ! $(grep -q "$_HEADER" $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}env_vars.sh) ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}env_vars.sh
  echo -e "$DEACTIVATION" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}env_vars.sh
fi

# conda install requirements
conda install -y -n $TARGET_ENV gsl pip
if [[ ! -z $OS ]]
then
  # On windows we need to install gnu make and gfortran using chocolatey and cygwin
  choco install make
else
  conda install -y -n $TARGET_ENV make
fi

# Platform dependent installs
# Ugly hack because we are not using conda build
if [[ $UNAME == Darwin ]]
then
  conda install -y -n $TARGET_ENV -c conda-forge gfortran_osx-64
elif [[ $UNAME == Linux ]]
then
  conda install -y -n $TARGET_ENV gfortran_linux-64
  # Horrible hack to get CI to work properly
  if [[ ! -f $TARGET_ENV_DIR${PATHSEP}bin${PATHSEP}gfortran ]]
  then
    ln $TARGET_ENV_DIR${PATHSEP}bin${PATHSEP}x86_64-conda_*-linux-gnu-gfortran $TARGET_ENV_DIR${PATHSEP}bin${PATHSEP}gfortran;
  fi
fi

# We cant activate the environment properly in a bash script so we do some exports to get things almost right
export PATH=$TARGET_ENV_DIR${PATHSEP}bin:$PATH
export C_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib:$DYLD_LIBRARY_PATH

# Make the fortran shared library and copy it to the environment's lib
cd $DIR${PATHSEP}..${PATHSEP}leakyIntegrator${PATHSEP}src
make static shared
mkdir -p $TARGET_ENV_DIR${PATHSEP}lib
cp libincgamNEG.so $TARGET_ENV_DIR${PATHSEP}lib
cp libincgamNEG.a $TARGET_ENV_DIR${PATHSEP}lib

# pip install the package
cd $DIR${PATHSEP}..
if [[ -z $OS ]]
then
  PIP=$TARGET_ENV_DIR${PATHSEP}bin${PATHSEP}pip
else
  PIP=$TARGET_ENV_DIR${PATHSEP}bin${PATHSEP}pip.exe
fi
$PIP install --upgrade pip
$PIP install -r requirements.txt

if [[ -z $DEVELOPMENT ]]
then
  $PIP install -v .
else
  $PIP install -v -e .
fi

echo "Successfully installed package!"