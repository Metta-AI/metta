#!/bin/bash
set -e

echo "=== Testing GitHub Webhook Service with Real PR ==="
echo ""
echo "This script will help you test the webhook service with a real GitHub PR."
echo ""

# Check if using AWS Secrets Manager or environment variables
if [ "$USE_AWS_SECRETS" = "true" ]; then
    echo "✅ Using AWS Secrets Manager for configuration"
    export AWS_REGION=${AWS_REGION:-us-east-1}
else
    # Check if required env vars are set
    if [ -z "$ASANA_PAT" ] || [ -z "$ASANA_WORKSPACE_GID" ] || [ -z "$ASANA_PROJECT_GID" ]; then
        echo "❌ Error: Required environment variables not set:"
        echo "   - ASANA_PAT"
        echo "   - ASANA_WORKSPACE_GID"
        echo "   - ASANA_PROJECT_GID"
        echo ""
        echo "For local testing, set them in your shell:"
        echo "   export ASANA_PAT='your-token-here'"
        echo "   export ASANA_WORKSPACE_GID='your-workspace-gid'"
        echo "   export ASANA_PROJECT_GID='your-project-gid'"
        echo ""
        echo "Or use AWS Secrets Manager: export USE_AWS_SECRETS=true"
        exit 1
    fi
    echo "✅ Using local environment variables for testing"
fi
echo ""

# Start the webhook service
echo "Starting webhook service on http://localhost:8000..."
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
uv run python -m github_webhook.app

