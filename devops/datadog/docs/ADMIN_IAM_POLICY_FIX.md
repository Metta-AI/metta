# IAM Policy Update - Admin Instructions

## Quick Summary

The Kubernetes CronJob `dashboard-cronjob` needs permission to access additional AWS Secrets Manager secrets for WandB and Asana collectors.

## Required Action

Update IAM policy `dashboard-secrets-access` to add 4 new secrets.

## Command to Run

```bash
# 1. Create updated policy document
cat > /tmp/dashboard-secrets-policy.json <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["secretsmanager:GetSecretValue"],
            "Resource": [
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:github/dashboard-token-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/api-key-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/app-key-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/access-token-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/workspace-gid-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/bugs-project-gid-*",
                "arn:aws:secretsmanager:us-east-1:751442549699:secret:wandb/api-key-*"
            ]
        }
    ]
}
EOF

# 2. Update the policy (creates new version and sets as default)
aws iam create-policy-version \
  --policy-arn arn:aws:iam::751442549699:policy/dashboard-secrets-access \
  --policy-document file:///tmp/dashboard-secrets-policy.json \
  --set-as-default
```

## What Changed

**Added 4 new secret ARNs:**
- `asana/access-token-*`
- `asana/workspace-gid-*`
- `asana/bugs-project-gid-*`
- `wandb/api-key-*`

**Existing secrets (unchanged):**
- `github/dashboard-token-*`
- `datadog/api-key-*`
- `datadog/app-key-*`

## Verification

```bash
# Verify policy was updated
aws iam get-policy \
  --policy-arn arn:aws:iam::751442549699:policy/dashboard-secrets-access \
  --query 'Policy.{DefaultVersionId:DefaultVersionId,UpdateDate:UpdateDate}' \
  --output table
```

The collectors should now run successfully in the K8s cronjob.
