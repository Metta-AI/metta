#!/bin/bash

REGION=us-east-1
ACCOUNT_ID="$1"
DOCKER_PASSWORD="$2"

if [ -z "$ACCOUNT_ID" ]; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

if [ -z "$ACCOUNT_ID" ]; then
  echo "Failed to get ACCOUNT_ID"
  exit 1
fi

HOST="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Uploading metta image to $HOST"

if [ -z "$DOCKER_PASSWORD" ]; then
  DOCKER_PASSWORD=$(aws ecr get-login-password --region $REGION)
fi

echo "$DOCKER_PASSWORD" | docker login --username AWS --password-stdin $HOST
docker tag mettaai/metta:latest $HOST/metta:latest
docker push $HOST/metta:latest
