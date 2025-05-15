#! /bin/bash -e

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
  exit 1
fi

sky api login -e "$SERVER"
