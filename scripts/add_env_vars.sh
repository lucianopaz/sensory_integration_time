#!/usr/bin/env bash

set -ex

# Parse input
usage() { echo "Usage: $0 [-n conda environment that must be modified]" 1>&2; exit 1; }
TARGET_ENV=$CONDA_DEFAULT_ENV

while getopts ":n:" o; do
  case "${o}" in
    n)
      TARGET_ENV=${OPTARG}
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

if [[ ! $(conda env list | grep -c "^$TARGET_ENV[[:space:]]") ]]
then
  MSG="\n
       The desired target envirornment $TARGET_ENV does not exist.\n
       Please create it and rerun this script.";
  echo -e $MSG;
  exit 2;
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

TARGET_ENV_DIR=$(conda env list | $DIR${PATHSEP}get_env_prefix.py $TARGET_ENV)

# Update the environment's activation env_vars scripts
mkdir -p $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d
mkdir -p $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d

# Check if env_vars exist
if [[ ! -f $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS ]]
then
  if [[ -z $OS ]]
  then
    echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS
  else
    type NUL > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS
  fi
fi
if [[ ! -f $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS ]]
then
  if [[ -z $OS ]]
  then
    echo -e "#!/usr/bin/env bash\n" > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS
  else
    type NUL > $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS
  fi
fi

# Init the activation commands
_HEADER="LINES ADDED BY THE INSTALLATION OF THE SENSORY_INTEGRATION_TIME PACKAGE"
HEADER="####### $_HEADER ######"

if [[ -z $OS ]]
then
  # UNIX settings
  ACTIVATION="
  export _OLD_CINCLUDE_PATH=\$C_INCLUDE_PATH
  export _OLD_CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH
  export _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
  export _OLD_DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH
  
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
else
  # WINDOWS settings
  ACTIVATION="
  set _OLD_CINCLUDE_PATH=\$C_INCLUDE_PATH
  set _OLD_CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH
  set _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
  set _OLD_DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH
  
  set C_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include${PATHSEP};\$C_INCLUDE_PATH
  set CPLUS_INCLUDE_PATH=$TARGET_ENV_DIR${PATHSEP}include${PATHSEP};\$CPLUS_INCLUDE_PATH
  set LD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib${PATHSEP};\$LD_LIBRARY_PATH
  set DYLD_LIBRARY_PATH=$TARGET_ENV_DIR${PATHSEP}lib${PATHSEP};\$DYLD_LIBRARY_PATH"
  DEACTIVATION="
  set C_INCLUDE_PATH=\$_OLD_CINCLUDE_PATH
  set CPLUS_INCLUDE_PATH=\$_OLD_CPLUS_INCLUDE_PATH
  set LD_LIBRARY_PATH=\$_OLD_LD_LIBRARY_PATH
  set DYLD_LIBRARY_PATH=\$_OLD_DYLD_LIBRARY_PATH
  
  set _OLD_CINCLUDE_PATH=
  set _OLD_CPLUS_INCLUDE_PATH=
  set _OLD_LD_LIBRARY_PATH=
  set _OLD_DYLD_LIBRARY_PATH="
fi

# Check if env_vars was already edited, and if not, edit
if [[ $(grep -c "$_HEADER" $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS) -eq 0 ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS
  echo -e "$ACTIVATION" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}activate.d${PATHSEP}$ENV_VARS
fi
if [[ $(grep -c "$_HEADER" $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS) -eq 0 ]]
then
  echo "$HEADER" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS
  echo -e "$DEACTIVATION" >> $TARGET_ENV_DIR${PATHSEP}etc${PATHSEP}conda${PATHSEP}deactivate.d${PATHSEP}$ENV_VARS
fi