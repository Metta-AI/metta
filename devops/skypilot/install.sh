#! /bin/bash -e

# Install SkyPilot with all cloud providers directly into the .venv
uv tool install skypilot==0.9.2 --from 'skypilot[aws,vast]'

SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM"
  exit 1
fi

sky api login -e "$SERVER"
