#!/bin/bash
set -e

echo "=== Deploying Lambda with AWS CLI ==="

FUNCTION_NAME="github-webhook-service"
ZIP_FILE="webhook_service.zip"
ROLE_NAME="github-webhook-lambda-role"
REGION="us-east-1"

# Check if zip exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: $ZIP_FILE not found. Run deploy_lambda.sh first to create the package."
    exit 1
fi

echo "📦 Using package: $ZIP_FILE ($(ls -lh $ZIP_FILE | awk '{print $5}'))"
echo ""

# Check if function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$ZIP_FILE" \
        --region "$REGION"

    echo "✅ Function updated!"
else
    echo "⚠️  Function doesn't exist. You need to create it first with Terraform or AWS Console."
    echo ""
    echo "To create the function, you need:"
    echo "1. IAM role with Lambda execution permissions"
    echo "2. Secrets Manager read permissions"
    echo ""
    echo "Run: terraform apply (if terraform is installed)"
    echo "Or ask Nishad to deploy it via Terraform"
    exit 1
fi

# Get Function URL
echo ""
echo "📋 Function URL:"
aws lambda get-function-url-config \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" \
    --query 'FunctionUrl' \
    --output text 2>/dev/null || echo "⚠️  Function URL not configured. Create it via Terraform or AWS Console."


