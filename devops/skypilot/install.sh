#! /bin/bash -e

# Install SkyPilot with all cloud providers directly into the .venv
uv tool install skypilot==0.9.2 --from 'skypilot[aws,vast]'

# Obtain the API server URL with credentials
SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text || true)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM. Have you ran ./devops/aws/setup_aws_profiles.sh?"
  exit 1
fi

sky api login -e "$SERVER"
