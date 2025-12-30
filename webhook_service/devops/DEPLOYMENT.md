# Lambda Deployment Instructions

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform installed
3. All required secrets in AWS Secrets Manager:
   - `github/webhook-secret`
   - `asana/access-token`
   - `asana/workspace-gid`
   - `asana/bugs-project-gid`
   - `github/token`

## Step 1: Package Lambda Function

Run the deployment script to create the zip package:

```bash
cd webhook_service
./devops/deploy_lambda.sh
```

This will:
- Install all Python dependencies
- Copy source code
- Create `devops/webhook_service.zip`

## Step 2: Deploy with Terraform

```bash
cd webhook_service/devops
terraform init
terraform plan
terraform apply
```

## Step 3: Get Lambda URL

After deployment, get the Function URL:

```bash
terraform output webhook_service_url
```

Or from AWS Console:
- Go to Lambda → Functions → `github-webhook-service`
- Click "Configuration" → "Function URL"
- Copy the URL

## Step 4: Configure GitHub Webhook

1. Go to GitHub repo → Settings → Webhooks
2. Add webhook with:
   - **Payload URL**: Lambda Function URL from Step 3
   - **Content type**: `application/json`
   - **Secret**: Value from `github/webhook-secret` in AWS Secrets Manager
   - **Events**: Select "Pull requests" (or individual events: opened, assigned, unassigned, edited, closed, reopened, synchronize)

## Testing

Test the webhook by:
1. Creating a PR
2. Changing assignees
3. Closing/reopening the PR
4. Check Asana for task updates


