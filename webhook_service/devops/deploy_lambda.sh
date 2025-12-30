#!/bin/bash
set -e

echo "=== Packaging Lambda Function ==="
cd "$(dirname "$0")/.."

# Clean previous build
rm -f devops/webhook_service.zip

# Create a temporary directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Installing dependencies..."
# Install dependencies to temp directory
uv pip install --system \
    --target "$TEMP_DIR" \
    asana boto3 fastapi httpx mangum pydantic pydantic-settings requests uvicorn[standard]

# Copy source code
echo "Copying source code..."
cp -r src "$TEMP_DIR/"

# Copy lambda handler
cp lambda_function.py "$TEMP_DIR/"

# Create zip file
echo "Creating zip package..."
cd "$TEMP_DIR"
ZIP_PATH="$(dirname "$0")/webhook_service.zip"
mkdir -p "$(dirname "$ZIP_PATH")"
zip -r "$ZIP_PATH" . -q

cd "$(dirname "$0")"
echo "✅ Package created: webhook_service.zip"
echo ""
echo "=== Deploying with Terraform ==="
terraform init
terraform plan
echo ""
read -p "Apply changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    terraform output webhook_service_url
fi

