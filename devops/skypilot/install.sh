#! /bin/bash -e

<<<<<<< HEAD
# Create a new virtual environment using uv
uv venv .venv/skypilot --python=3.11 --no-project
source .venv/skypilot/bin/activate

# Install SkyPilot with all cloud providers
uv pip install skypilot==0.9.2 --prerelease=allow
uv pip install "skypilot[aws]"
uv pip install "skypilot[vast]"

SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM"
=======
# Install SkyPilot directly into the .venv
# This is not in requirements_pinned.txt because want to make sure that users are using our remote API server
uv pip install skypilot==0.9.2 --prerelease=allow

# Obtain the API server URL with credentials
SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text || true)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM. Have you ran ./devops/aws/setup_aws_profiles.sh?"
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
  exit 1
fi

sky api login -e "$SERVER"
