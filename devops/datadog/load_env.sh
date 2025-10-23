#!/usr/bin/env bash
# Source this file to load Datadog credentials into your environment
#
# Usage:
#   source ./load_env.sh
#   # or
#   . ./load_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE" >&2
    echo "Copy .env.sample to .env and fill in your credentials" >&2
    return 1 2>/dev/null || exit 1
fi

# Load environment variables
set -a
source "$ENV_FILE"
set +a

# Also export Terraform variables
export TF_VAR_datadog_api_key="$DD_API_KEY"
export TF_VAR_datadog_app_key="$DD_APP_KEY"

# Verify credentials are set
if [ "$DD_API_KEY" = "your_api_key_here" ] || [ -z "$DD_API_KEY" ]; then
    echo "⚠️  Warning: DD_API_KEY is not set or is using placeholder value" >&2
    echo "Edit $ENV_FILE and add your actual Datadog API key" >&2
    return 1 2>/dev/null || exit 1
fi

if [ "$DD_APP_KEY" = "your_app_key_here" ] || [ -z "$DD_APP_KEY" ]; then
    echo "⚠️  Warning: DD_APP_KEY is not set or is using placeholder value" >&2
    echo "Edit $ENV_FILE and add your actual Datadog Application key" >&2
    return 1 2>/dev/null || exit 1
fi

echo "✓ Datadog credentials loaded successfully"
echo "  DD_API_KEY: ${DD_API_KEY:0:8}... (${#DD_API_KEY} chars)"
echo "  DD_APP_KEY: ${DD_APP_KEY:0:8}... (${#DD_APP_KEY} chars)"
echo "  DD_SITE: ${DD_SITE:-datadoghq.com}"
echo ""
echo "Environment variables are now set. You can run:"
echo "  ./fetch_dashboards.py --format=summary"
echo "  ./export_dashboard.py <dashboard-id>"
echo "  ./batch_export.py --limit=5"
