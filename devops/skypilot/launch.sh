#!/bin/bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 CMD RUN_ID"
  exit 1
fi

CMD=$1
RUN_ID=$2
shift 2
CMD_ARGS="$@"


SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-1)

AWS_PROFILE=stem-db-admin sky jobs launch \
  --name $RUN_ID \
  ./devops/skypilot/config/train.yaml \
  --env SKYPILOT_DOCKER_PASSWORD=$SKYPILOT_DOCKER_PASSWORD \
  --env METTA_RUN_ID=$RUN_ID \
  --env METTA_CMD=$CMD \
  --env METTA_CMD_ARGS="$CMD_ARGS" \
  --env METTA_GIT_REF=$(git rev-parse HEAD) \
  --detach-run \
  --async \
  --yes
