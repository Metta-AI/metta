#! /bin/bash -e

# Obtain the API server URL with credentials
SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text || true)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM. Have you ran ./devops/aws/setup_aws_profiles.sh?"
  exit 1
fi

# Install SkyPilot directly into the .venv
# This is not in requirements_pinned.txt because want to make sure that users are using our remote API server
uv pip install skypilot==0.9.2 --prerelease=allow

sky api login -e "$SERVER"
