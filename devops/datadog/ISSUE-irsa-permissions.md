# Issue: IRSA Permissions for Dashboard CronJob

## Status
**BLOCKED**: CronJob failing since 2025-10-24 ~17:00 UTC due to IAM permission issues

## Problem
Kubernetes pods cannot assume IAM role to access AWS Secrets Manager.

**Error**: `Not authorized to perform sts:AssumeRoleWithWebIdentity`

**Impact**:
- All collectors failing in production and dev environments
- No metrics being collected or pushed to Datadog
- CronJob deployment exists but is non-functional

## Root Cause
IAM role trust policy needs to be updated to allow the Kubernetes service account to assume the role via IRSA (IAM Roles for Service Accounts).

**Known Issue**: IAM trust policies with `StringEquals` do not support wildcard patterns. Must use `StringLike` for service account wildcards.

## Current Configuration

**Cluster**: main (us-east-1)
**AWS Account**: 751442549699
**Namespace**: monitoring
**Service Accounts**:
- Production: `dashboard-cronjob-dashboard-cronjob`
- Dev: `dashboard-cronjob-dev-dev`

**IAM Role**: `arn:aws:iam::751442549699:role/dashboard-cronjob`

**OIDC Provider**: Extract from cluster:
```bash
AWS_PROFILE=softmax-admin aws eks describe-cluster --name main \
  --query "cluster.identity.oidc.issuer" --output text
```

## Required Terraform Changes

### Location
File: `devops/tf/eks/irsa.tf` (or similar IAM configuration file)

### Change Required

Add or update the IAM role trust policy for `dashboard-cronjob`:

```hcl
# Dashboard CronJob IRSA Role
resource "aws_iam_role" "dashboard_cronjob" {
  name = "dashboard-cronjob"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::751442549699:oidc-provider/${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringLike = {
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:monitoring:dashboard-cronjob*"
          }
          StringEquals = {
            "${replace(data.aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
    Purpose     = "datadog-collectors"
  }
}

# Attach Secrets Manager read policy
resource "aws_iam_role_policy" "dashboard_cronjob_secrets" {
  name = "dashboard-cronjob-secrets-access"
  role = aws_iam_role.dashboard_cronjob.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:dev/datadog/collectors/*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:github/dashboard-token*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/*"
        ]
      }
    ]
  })
}
```

### Key Points

1. **Use `StringLike` not `StringEquals`** for the `:sub` condition with wildcard pattern `dashboard-cronjob*`
   - This allows both `dashboard-cronjob-dashboard-cronjob` (prod) and `dashboard-cronjob-dev-dev` (dev)

2. **OIDC Provider ARN** must match the cluster's OIDC provider
   - Extract from: `aws eks describe-cluster --name main`
   - Format: `arn:aws:iam::751442549699:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/<CLUSTER_ID>`

3. **Service Account Pattern**: `system:serviceaccount:monitoring:dashboard-cronjob*`
   - Namespace: `monitoring`
   - Service account prefix: `dashboard-cronjob`
   - Wildcard allows multiple environments (prod, dev, test)

4. **Secrets Access**: Grant read access to all collector secrets:
   - `dev/datadog/collectors/*` - All collector credentials
   - `github/dashboard-token*` - GitHub API access
   - `datadog/*` - Datadog API keys

## Verification Steps

After Terraform apply:

### 1. Verify IAM Role Trust Policy
```bash
AWS_PROFILE=softmax-admin aws iam get-role \
  --role-name dashboard-cronjob \
  --query 'Role.AssumeRolePolicyDocument' \
  --output json
```

Should show `StringLike` for the `:sub` condition.

### 2. Verify Service Account Annotation
```bash
kubectl get serviceaccount -n monitoring dashboard-cronjob-dashboard-cronjob -o yaml
kubectl get serviceaccount -n monitoring dashboard-cronjob-dev-dev -o yaml
```

Should have annotation:
```yaml
annotations:
  eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/dashboard-cronjob
```

### 3. Test with Manual Job
```bash
# Create test job from cronjob
kubectl create job -n monitoring test-irsa-$(date +%s) \
  --from=cronjob/dashboard-cronjob-dev-dev

# Watch logs for success
kubectl logs -n monitoring -l job-name=test-irsa-* -f
```

Should see:
- No `AccessDenied` errors
- Successful secret retrieval
- Metrics collection and push to Datadog
- "✅ Successfully pushed X metrics" messages

### 4. Verify Production CronJob
```bash
# Wait for next scheduled run (every 15 minutes)
kubectl get jobs -n monitoring -l cronjob=dashboard-cronjob-dashboard-cronjob

# Check latest pod logs
kubectl logs -n monitoring -l job-name=dashboard-cronjob-dashboard-cronjob-* --tail=100
```

## Related Files

**Helm Chart**: `devops/charts/dashboard-cronjob/`
- `templates/serviceaccount.yaml` - Defines service account with IRSA annotation
- `templates/cronjob.yaml` - CronJob that uses the service account
- `values.yaml` - Configuration values

**Collectors**: `devops/datadog/collectors/`
- Base collector uses AWS SDK to fetch secrets from Secrets Manager
- Falls back to environment variables if IRSA fails

**Documentation**:
- `devops/datadog/docs/DEPLOYMENT_GUIDE.md` - Deployment instructions
- `devops/datadog/collectors/*/README.md` - Per-collector configuration

## Success Criteria

- [ ] Terraform configuration added/updated
- [ ] `terraform plan` shows expected changes
- [ ] `terraform apply` executes successfully
- [ ] IAM role trust policy uses `StringLike` for service account pattern
- [ ] Service accounts have correct IRSA annotation
- [ ] Test job completes successfully without `AccessDenied` errors
- [ ] Production CronJob runs successfully
- [ ] Metrics appear in Datadog dashboard

## Timeline

**Issue Started**: 2025-10-24 ~17:00 UTC
**Current Status**: Blocked - waiting for Terraform changes
**Priority**: High - production monitoring is down

## Notes

- This is a known pattern issue with IRSA - `StringEquals` vs `StringLike`
- We've encountered this before, so the fix is well understood
- The Terraform changes are the only blocker - collectors are ready
- Once fixed, all 7 collectors will automatically start working
- No code changes needed - purely IAM configuration

## Contact

For questions about this issue:
- Collector implementation: See `devops/datadog/collectors/`
- IRSA configuration: See Terraform files in `devops/tf/eks/`
- Deployment: See `devops/datadog/docs/DEPLOYMENT_GUIDE.md`
