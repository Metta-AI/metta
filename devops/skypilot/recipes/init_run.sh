#!/usr/bin/env bash
# Common initialization for SkyPilot jobs

set -e
cd /workspace/metta

# Handle virtual environment activation
# Note that the docker image may start with its own venv - switch to metta venv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi
. .venv/bin/activate

# Configure environment based on job type
if [ "${SANDBOX_MODE:-false}" = "true" ]; then
    ./devops/skypilot/utils/configure_environment.py --sandbox
else
    ./devops/skypilot/utils/configure_environment.py
fi

# Source environment file
METTA_ENV_FILE="$(uv run ./common/src/metta/common/util/constants.py METTA_ENV_FILE)"
source "$METTA_ENV_FILE"
