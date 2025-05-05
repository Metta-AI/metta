#!/bin/bash
set -e

# Ensure script is run with bash
if [ -z "$BASH_VERSION" ]; then
  echo "This script must be run with bash" >&2
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 CMD RUN_ID"
  exit 1
fi

CMD=$1
RUN_ID=$2
shift 2
CMD_ARGS="$@"

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

source .venv/skypilot/bin/activate

export SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-1)

AWS_PROFILE=softmax-db-admin sky jobs launch \
  --gpus L4:$gpus \
  --num-nodes $nodes \
  --cpus $cpus\+ \
  --name $RUN_ID \
  ./devops/skypilot/config/train.yaml \
  --env SKYPILOT_DOCKER_PASSWORD \
  --env METTA_RUN_ID=$RUN_ID \
  --env METTA_CMD=$CMD \
  --env METTA_CMD_ARGS="$CMD_ARGS" \
  --env METTA_GIT_REF \
  --detach-run \
  --async \
  --yes
