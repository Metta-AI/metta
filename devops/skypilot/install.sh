#! /bin/bash -e

# Obtain the API server URL with credentials
SERVER=$(AWS_PROFILE=softmax aws ssm get-parameter --name /skypilot/api_url --query Parameter.Value --output text || true)

if [ -z "$SERVER" ]; then
  echo "Failed to get Skypilot API server URL from SSM. Have you ran ./devops/aws/setup_aws_profiles.sh?"
  exit 1
fi

uv run sky api login -e "$SERVER"
