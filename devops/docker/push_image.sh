REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$ACCOUNT_ID" ]; then
  echo "Failed to get ACCOUNT_ID"
  exit 1
fi

HOST="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Uploading metta image to $HOST"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $HOST
docker tag mettaai/metta:latest $HOST/metta:latest
docker push $HOST/metta:latest
