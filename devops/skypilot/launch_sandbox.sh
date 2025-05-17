#!/bin/bash
set -e

# Ensure script is run with bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script must be run with bash" >&2
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 RUN_ID"
  exit 1
fi

SANDBOX_ID=$1
shift 1

# Defaults
export METTA_GIT_REF=$(git rev-parse HEAD)
gpus=1
nodes=1
cpus=8

for arg in "$@"; do
  case $arg in
    --git-branch=*)
      value="${arg#*=}"
      export METTA_GIT_REF=$(git rev-parse "$value")
      echo "METTA_GIT_REF: $METTA_GIT_REF"
      CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--git-branch=[^ ]* ?//g")
      ;;
    --gpus=*)
      gpus="${arg#*=}"
      echo "gpus: $gpus"
      CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--gpus=[^ ]* ?//g")
      ;;
    --nodes=*)
      nodes="${arg#*=}"
      echo "nodes: $nodes"
      CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--nodes=[^ ]* ?//g")
      ;;
    --cpus=*)
      cpus="${arg#*=}"
      echo "cpus: $cpus"
      CMD_ARGS=$(echo "$CMD_ARGS" | sed -E "s/--cpus=[^ ]* ?//g")
      ;;
  esac
done
sky launch \
  --gpus A100:8 \
  --num-nodes $nodes \
  --cpus $cpus\+ \
  --cluster $SANDBOX_ID \
  ./devops/skypilot/config/sk_sanbox.yaml \
  --env METTA_RUN_ID=$RUN_ID \
  --env METTA_GIT_REF \
  --detach-run \
  --async \
  --yes
