# Lambda Deployment

## Prerequisites

1. Package the webhook service:
```bash
cd webhook_service
zip -r webhook_service.zip . -x "*.pyc" "__pycache__/*" ".venv/*" "*.git/*"
```

2. Ensure Terraform is configured with AWS credentials

## Deploy

```bash
cd devops
terraform init
terraform plan
terraform apply
```

## Configuration

The Lambda function will automatically read secrets from AWS Secrets Manager:
- `github/webhook-secret`
- `asana/access-token`
- `asana/workspace-gid`
- `asana/bugs-project-gid`

Set `USE_AWS_SECRETS=true` in Lambda environment variables (already configured in Terraform).

## Update GitHub Webhook

After deployment, update the GitHub webhook URL to point to the Lambda Function URL (output from Terraform).


