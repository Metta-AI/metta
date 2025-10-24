# IRSA (IAM Roles for Service Accounts) Fix Plan

## Problem
Kubernetes pods can't assume IAM role to access AWS Secrets Manager.

**Error**: `Not authorized to perform sts:AssumeRoleWithWebIdentity`

**Impact**: CronJob failing since ~40 minutes ago (started around 17:00 UTC 2025-10-24)

## Root Cause
The IAM role `dashboard-cronjob` trust policy uses `StringEquals` with a wildcard pattern (`dashboard-cronjob-*`), but **`StringEquals` does not support wildcards in AWS IAM**. Wildcards only work with `StringLike`.

This causes the OIDC authentication to fail because the wildcard pattern in the `:sub` condition doesn't match when using `StringEquals`.

**Note**: This is the same issue we've encountered before. The fix is to change `StringEquals` to `StringLike` for the service account pattern.

## Current Configuration

**Cluster**: main (us-east-1)
**Account**: 751442549699
**Service Account**: monitoring/dashboard-cronjob-dashboard-cronjob
**IAM Role**: arn:aws:iam::751442549699:role/dashboard-cronjob

## Diagnostic Steps

### 1. Get EKS Cluster OIDC Information

```bash
# Get OIDC provider URL
AWS_PROFILE=softmax-admin aws eks describe-cluster --name main \
  --query "cluster.identity.oidc.issuer" --output text

# Example output: https://oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE
# Extract the ID from the URL (everything after /id/)
```

### 2. Check Current IAM Role Trust Policy

```bash
# View current trust policy
AWS_PROFILE=softmax-admin aws iam get-role \
  --role-name dashboard-cronjob \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json

# Save backup before making changes
AWS_PROFILE=softmax-admin aws iam get-role \
  --role-name dashboard-cronjob \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json > /tmp/trust-policy-backup.json
```

### 3. Verify OIDC Provider Exists

```bash
# List OIDC providers
AWS_PROFILE=softmax-admin aws iam list-open-id-connect-providers

# Should see: arn:aws:iam::751442549699:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/CLUSTER_ID
```

## Fix: Update Trust Policy

### Step 1: Get the OIDC Provider ID

```bash
# Get OIDC issuer URL
OIDC_URL=$(AWS_PROFILE=softmax-admin aws eks describe-cluster --name main \
  --query "cluster.identity.oidc.issuer" --output text)

# Extract just the ID part
OIDC_ID=$(echo $OIDC_URL | sed 's|https://oidc.eks.us-east-1.amazonaws.com/id/||')

echo "OIDC Provider ID: $OIDC_ID"
```

### Step 2: Create the Correct Trust Policy

**The fix is to use `StringLike` for the wildcard pattern instead of `StringEquals`.**

```bash
# Create trust policy with StringLike for wildcard pattern
cat > /tmp/trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::751442549699:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/${OIDC_ID}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.us-east-1.amazonaws.com/id/${OIDC_ID}:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "oidc.eks.us-east-1.amazonaws.com/id/${OIDC_ID}:sub": "system:serviceaccount:monitoring:dashboard-cronjob-*"
        }
      }
    }
  ]
}
EOF

# Review the policy
cat /tmp/trust-policy.json | jq .
```

**Key changes**:
- Split condition into two parts: `StringEquals` and `StringLike`
- `:aud` stays under `StringEquals` (exact match, no wildcard)
- `:sub` moved to `StringLike` to support wildcard pattern `dashboard-cronjob-*`

### Step 3: Apply the Trust Policy

```bash
# Backup current policy (already done in step 2)

# Update the role
AWS_PROFILE=softmax-admin aws iam update-assume-role-policy \
  --role-name dashboard-cronjob \
  --policy-document file:///tmp/trust-policy.json

# Verify the update
AWS_PROFILE=softmax-admin aws iam get-role \
  --role-name dashboard-cronjob \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json
```

## Testing After Fix

### Test 1: Manual Job Run

```bash
# Create a manual test job
kubectl create job -n monitoring \
  --from=cronjob/dashboard-cronjob-dashboard-cronjob \
  test-irsa-$(date +%s)

# Wait for pod to start
sleep 15

# Get the job name
JOB_NAME=$(kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -n 1 | awk '{print $1}')

# Check logs
kubectl logs -n monitoring -l job-name=$JOB_NAME --tail=100

# Look for: "✅ github: 28 metrics collected"
# Should NOT see: "AccessDenied" errors
```

### Test 2: Wait for Next Scheduled Run

```bash
# Check when next run is scheduled
kubectl get cronjob -n monitoring dashboard-cronjob-dashboard-cronjob

# After the scheduled time, check the job
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -n 1

# Should see: Complete (not Failed)
```

## Expected Outcome

✅ Manual test job completes successfully
✅ Collectors authenticate to AWS Secrets Manager
✅ All 7 collectors run and push metrics to Datadog
✅ CronJob succeeds on next scheduled run (every 15 minutes)

## Rollback Plan

If the fix causes issues:

```bash
# Restore previous trust policy
AWS_PROFILE=softmax-admin aws iam update-assume-role-policy \
  --role-name dashboard-cronjob \
  --policy-document file:///tmp/trust-policy-backup.json

# Verify restoration
AWS_PROFILE=softmax-admin aws iam get-role \
  --role-name dashboard-cronjob \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json
```

## Alternative: Check if OIDC Provider Needs Setup

If the OIDC provider doesn't exist:

```bash
# Create OIDC provider (if missing)
eksctl utils associate-iam-oidc-provider \
  --cluster main \
  --region us-east-1 \
  --approve
```

## Timeline

- **Diagnostic**: 5 minutes
- **Policy creation**: 2 minutes
- **Policy application**: 1 minute
- **Testing**: 5 minutes
- **Total**: ~15 minutes

## Success Criteria

1. Trust policy includes correct OIDC provider ARN
2. Condition matches service account namespace and name
3. Manual test job succeeds
4. Scheduled CronJob runs successfully
5. All 7 collectors push metrics to Datadog

## Notes

- This issue started ~40 minutes ago, suggesting recent AWS configuration change
- The Secrets Manager permissions were fixed earlier today
- This is a different issue - the pod can't even assume the role to try accessing secrets
- Service account annotation is correct - only trust policy needs updating

---

**Created**: 2025-10-24
**Status**: Ready to execute
**Next Action**: Someone with AWS IAM admin access should run the diagnostic and fix steps
