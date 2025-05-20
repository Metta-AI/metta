<<<<<<< HEAD
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

=======
#!/bin/bash

REGION=us-east-1
ACCOUNT_ID="$1"
DOCKER_PASSWORD="$2"

if [ -z "$ACCOUNT_ID" ]; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
if [ -z "$ACCOUNT_ID" ]; then
  echo "Failed to get ACCOUNT_ID"
  exit 1
fi

HOST="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Uploading metta image to $HOST"

<<<<<<< HEAD
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $HOST
=======
if [ -z "$DOCKER_PASSWORD" ]; then
  DOCKER_PASSWORD=$(aws ecr get-login-password --region $REGION)
fi

echo "$DOCKER_PASSWORD" | docker login --username AWS --password-stdin $HOST
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
docker tag mettaai/metta:latest $HOST/metta:latest
docker push $HOST/metta:latest
