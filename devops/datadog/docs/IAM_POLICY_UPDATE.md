# IAM Policy Update Required

## Issue

The K8s test jobs are failing because the service account's IAM role (`dashboard-cronjob`) doesn't have permission to access WandB and Asana secrets in AWS Secrets Manager.

## Root Cause

The IAM policy `dashboard-secrets-access` only grants access to:
- `github/dashboard-token-*`
- `datadog/api-key-*`
- `datadog/app-key-*`

But the collectors also need:
- `asana/access-token-*`
- `asana/workspace-gid-*`
- `asana/bugs-project-gid-*`
- `wandb/api-key-*`

## Error Message

```
Error: WANDB_API_KEY not found in environment or AWS Secrets Manager.
An error occurred (ResourceNotFoundException) when calling the GetSecretValue operation:
Secrets Manager can't find the specified secret.
```

## Solution

Update the IAM policy to include the missing secrets. Run this command with admin credentials:

```bash
# Create the updated policy document
cat > /tmp/updated-policy.json <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
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

# Update the IAM policy (requires admin permissions)
aws iam create-policy-version \
  --policy-arn arn:aws:iam::751442549699:policy/dashboard-secrets-access \
  --policy-document file:///tmp/updated-policy.json \
  --set-as-default

# Verify the update
aws iam get-policy-version \
  --policy-arn arn:aws:iam::751442549699:policy/dashboard-secrets-access \
  --version-id $(aws iam get-policy --policy-arn arn:aws:iam::751442549699:policy/dashboard-secrets-access --query 'Policy.DefaultVersionId' --output text) \
  --query 'PolicyVersion.Document' \
  --output json
```

## Alternative: AWS Console

1. Go to IAM → Policies → `dashboard-secrets-access`
2. Click "Edit policy"
3. Add the following resources to the existing statement:
   - `arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/access-token-*`
   - `arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/workspace-gid-*`
   - `arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/bugs-project-gid-*`
   - `arn:aws:secretsmanager:us-east-1:751442549699:secret:wandb/api-key-*`
4. Click "Review policy" → "Save changes"

## Verification

After updating the policy, test with:

```bash
# Create a test job
kubectl create job --from=cronjob/dashboard-cronjob-dashboard-cronjob manual-test-iam-fix-$(date +%s) -n monitoring

# Watch the job
kubectl get jobs -n monitoring -w

# Check logs
kubectl logs -n monitoring -l job-name=manual-test-iam-fix-* --tail=100
```

All 6 collectors (github, kubernetes, ec2, skypilot, wandb, asana) should now run successfully.
