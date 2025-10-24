# Datadog Collectors - Secrets Setup Guide

This guide explains how to configure secrets for the Datadog collectors using AWS Secrets Manager.

## Overview

All sensitive credentials (API keys, tokens) are stored in AWS Secrets Manager for security. The collectors
automatically fetch these secrets at runtime.

## Prerequisites

- AWS CLI installed and configured
- AWS credentials with `secretsmanager:CreateSecret` and `secretsmanager:GetSecretValue` permissions
- Appropriate AWS profile configured (e.g., `softmax-admin`)

## Quick Start

### 1. Set Your AWS Profile

```bash
export AWS_PROFILE=softmax-admin
```

### 2. Store All Secrets

Run these commands with your actual credential values:

```bash
# Datadog API Key
# Get from: https://app.datadoghq.com/organization-settings/api-keys
aws secretsmanager create-secret \
  --name datadog/api-key \
  --secret-string "YOUR_DD_API_KEY" \
  --region us-east-1

# Datadog Application Key
# Get from: https://app.datadoghq.com/organization-settings/application-keys
# Required permissions: dashboards_read, dashboards_public_share, metrics_write
aws secretsmanager create-secret \
  --name datadog/app-key \
  --secret-string "YOUR_DD_APP_KEY" \
  --region us-east-1

# GitHub Personal Access Token
# Get from: https://github.com/settings/tokens
# Required scopes: repo, public_repo, workflow
aws secretsmanager create-secret \
  --name github/dashboard-token \
  --secret-string "YOUR_GITHUB_TOKEN" \
  --region us-east-1

# Asana Credentials
# Get access token from: https://app.asana.com (Profile > My Settings > Apps > Personal access tokens)
aws secretsmanager create-secret \
  --name asana/access-token \
  --secret-string "YOUR_ASANA_TOKEN" \
  --region us-east-1

# Get workspace GID from your Asana workspace URL: https://app.asana.com/0/WORKSPACE_GID/...
aws secretsmanager create-secret \
  --name asana/workspace-gid \
  --secret-string "YOUR_WORKSPACE_GID" \
  --region us-east-1

# Optional: Get bugs project GID from project URL: .../project/PROJECT_GID/...
aws secretsmanager create-secret \
  --name asana/bugs-project-gid \
  --secret-string "YOUR_BUGS_PROJECT_GID" \
  --region us-east-1
```

### 3. Configure Environment Variables (Optional)

All configuration values can now be fetched from AWS Secrets Manager.

For local development with custom values, create a `.env` file in `devops/datadog/`:

```bash
# Datadog Site (optional, defaults to datadoghq.com)
DD_SITE=datadoghq.com

# GitHub Configuration (required)
GITHUB_ORG=PufferAI
GITHUB_REPO=metta

# Asana Configuration (optional if stored in AWS Secrets Manager)
# Only needed if you want to override the AWS values
#ASANA_WORKSPACE_GID=your_workspace_gid
#ASANA_BUGS_PROJECT_GID=your_bugs_project_gid
```

**Note:** Since workspace and project GIDs are now in AWS Secrets Manager, you don't need to set them locally unless you
want to override the AWS values.

### 4. Validate Setup

Run the validation script to ensure all secrets are configured:

```bash
uv run python devops/datadog/scripts/validate_secrets.py
```

This will check:

- ✓ All required secrets exist in AWS Secrets Manager
- ✓ All required environment variables are set
- ✓ Secrets can be retrieved successfully

### 5. Test Collectors

```bash
# Test each collector (dry-run, no push to Datadog)
uv run python devops/datadog/run_collector.py github --verbose
uv run python devops/datadog/run_collector.py skypilot --verbose
uv run python devops/datadog/run_collector.py asana --verbose

# Test with actual push to Datadog
uv run python devops/datadog/run_collector.py github --push --verbose
```

## Secret Management

### View Existing Secrets

```bash
# List all secrets
aws secretsmanager list-secrets --region us-east-1 \
  | jq '.SecretList[] | select(.Name | test("datadog|github|asana")) | {Name, CreatedDate}'

# Verify a specific secret exists (doesn't show value)
aws secretsmanager describe-secret \
  --secret-id datadog/api-key \
  --region us-east-1
```

### Update Existing Secrets

```bash
# Update Datadog API key
aws secretsmanager update-secret \
  --secret-id datadog/api-key \
  --secret-string "NEW_DD_API_KEY" \
  --region us-east-1

# Update Datadog App key
aws secretsmanager update-secret \
  --secret-id datadog/app-key \
  --secret-string "NEW_DD_APP_KEY" \
  --region us-east-1

# Update GitHub token
aws secretsmanager update-secret \
  --secret-id github/dashboard-token \
  --secret-string "NEW_GITHUB_TOKEN" \
  --region us-east-1

# Update Asana token
aws secretsmanager update-secret \
  --secret-id asana/access-token \
  --secret-string "NEW_ASANA_TOKEN" \
  --region us-east-1
```

### Delete Secrets

```bash
# Schedule deletion (30-day recovery window by default)
aws secretsmanager delete-secret \
  --secret-id datadog/api-key \
  --region us-east-1

# Force immediate deletion (use with caution!)
aws secretsmanager delete-secret \
  --secret-id datadog/api-key \
  --force-delete-without-recovery \
  --region us-east-1
```

## Required Secrets Summary

| Secret Name              | Description                      | How to Get                                                       |
| ------------------------ | -------------------------------- | ---------------------------------------------------------------- |
| `datadog/api-key`        | Datadog API Key                  | https://app.datadoghq.com/organization-settings/api-keys         |
| `datadog/app-key`        | Datadog Application Key          | https://app.datadoghq.com/organization-settings/application-keys |
| `github/dashboard-token` | GitHub Personal Access Token     | https://github.com/settings/tokens                               |
| `asana/access-token`     | Asana Personal Access Token      | Asana → My Settings → Apps → Personal access tokens              |
| `asana/workspace-gid`    | Asana Workspace ID               | From Asana workspace URL                                         |
| `asana/bugs-project-gid` | Asana Bugs Project ID (optional) | From Asana project URL                                           |

## Required Environment Variables

| Variable                 | Description           | Example            | Required                       |
| ------------------------ | --------------------- | ------------------ | ------------------------------ |
| `DD_SITE`                | Datadog site          | `datadoghq.com`    | No (defaults to datadoghq.com) |
| `GITHUB_ORG`             | GitHub organization   | `PufferAI`         | Yes                            |
| `GITHUB_REPO`            | GitHub repository     | `metta`            | Yes                            |
| `ASANA_WORKSPACE_GID`    | Asana workspace ID    | `1234567890123456` | No (from AWS if not set)       |
| `ASANA_BUGS_PROJECT_GID` | Asana bugs project ID | `1234567890123457` | No (from AWS if not set)       |

## Fallback Behavior

The collectors use this priority order:

1. **Environment variables** (from `.env` file or shell)
2. **AWS Secrets Manager** (if env var not set)
3. **Error** (if neither found)

This allows:

- **Local development**: Override with `.env` file (faster, no AWS calls)
- **Production/K8s**: Use AWS Secrets Manager (secure, centralized)

## Local Development Override

For local development, you can bypass AWS Secrets Manager by setting environment variables:

```bash
# Add these to .env for local development only
DD_API_KEY=your_local_dev_key
DD_APP_KEY=your_local_dev_app_key
GITHUB_TOKEN=your_local_github_token
ASANA_ACCESS_TOKEN=your_local_asana_token
```

**Warning**: Never commit `.env` with real credentials to version control!

## Troubleshooting

### Secret Not Found Error

```
Error: DD_API_KEY not found in environment or AWS Secrets Manager
```

**Solutions:**

1. Check AWS profile is correct: `echo $AWS_PROFILE`
2. Verify secret exists: `aws secretsmanager describe-secret --secret-id datadog/api-key --region us-east-1`
3. Check AWS permissions: Ensure your role has `secretsmanager:GetSecretValue`
4. Verify region: Secrets are stored in `us-east-1`

### AWS Credentials Error

```
Unable to locate credentials
```

**Solutions:**

1. Run `aws configure` to set up credentials
2. Set AWS_PROFILE: `export AWS_PROFILE=softmax-admin`
3. Check `~/.aws/credentials` file exists

### Asana Workspace GID Not Set

```
Error: ASANA_WORKSPACE_GID environment variable not set
```

**Solution:** Add to `.env` file:

```bash
ASANA_WORKSPACE_GID=your_workspace_gid_here
ASANA_BUGS_PROJECT_GID=your_bugs_project_gid_here
```

## Security Best Practices

1. **Never commit secrets to git** - Use `.gitignore` for `.env` files
2. **Rotate secrets regularly** - Update secrets in AWS Secrets Manager every 90 days
3. **Use least privilege** - Only grant necessary AWS permissions
4. **Enable secret rotation** - Configure automatic rotation for production
5. **Audit access** - Review AWS CloudTrail logs for secret access

## Production Deployment (Kubernetes)

For Kubernetes deployments, secrets are injected via environment variables:

```yaml
# In Helm chart values
env:
  - name: ASANA_WORKSPACE_GID
    value: '1209016784099267'
  - name: ASANA_BUGS_PROJECT_GID
    value: '1210062854657778'
```

The collectors automatically fetch credentials from AWS Secrets Manager when running in the cluster.

## Support

For issues or questions:

- Check the validation script: `uv run python devops/datadog/scripts/validate_secrets.py`
- Review collector logs with `--verbose` flag
- Verify AWS access with `aws sts get-caller-identity`
