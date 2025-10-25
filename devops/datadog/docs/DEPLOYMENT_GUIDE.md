# Datadog Collector Deployment Guide

This guide covers deploying the dashboard metrics collectors to Kubernetes/EKS.

## Production Deployment

**Production deployment is fully automated via GitHub Actions:**

- **Trigger**: Automatic on merge to `main` branch (when changes affect collector code)
- **Process**: Builds Docker image → Pushes to ECR → Deploys via Helm
- **No manual steps required** for production deployment

See [CI/CD Integration](#cicd-integration) section below for details.

## Development/Testing Deployment

**For testing changes before merging to main:**

### Quick Reference: Common Workflow

```bash
# 1. Build Docker image from your branch
gh workflow run build-dashboard-image.yml --ref your-branch-name
# IMPORTANT: Use --ref to specify your feature branch, otherwise it builds from main
gh run watch  # Wait for completion (~1-2 min)

# 2. Get the image tag
git rev-parse --short=7 HEAD  # Output: abc1234
# Image will be tagged as: sha-abc1234

# 3. Update helmfile with your image tag
vim devops/charts/helmfile.yaml
# Uncomment dashboard-cronjob-dev section
# Set: image.tag: "sha-abc1234"

# 4. Deploy using helmfile (NOT helm directly)
cd devops/charts
helmfile apply -l name=dashboard-cronjob-dev

# 5. Test manually
kubectl create job --from=cronjob/dashboard-cronjob-dev-dashboard-cronjob \
  test-$(date +%s) -n monitoring

# 6. Watch it run
kubectl logs -n monitoring -l job-name=test-* --tail=100 -f
```

**Why helmfile for dev?** Using `helm upgrade` directly will create a separate service account without IRSA permissions. Helmfile applies `values-dev.yaml` which reuses production's service account.

## Overview

The dashboard collectors run as Kubernetes CronJobs that:

- Execute every 15 minutes
- Collect metrics from GitHub, Asana, Kubernetes, SkyPilot, WandB, etc.
- Push metrics to Datadog
- Use AWS Secrets Manager for authentication (GitHub token, Datadog API keys)

## Prerequisites

- AWS CLI configured with appropriate credentials
- kubectl installed
- helmfile installed
- Access to EKS cluster (IAM permissions)

## Quick Start

### 1. Setup kubectl access

```bash
# Configure kubectl for main cluster
aws eks update-kubeconfig --name main --region us-east-1

# Verify access
kubectl get namespaces
```

### 2. Production Deployment (Automated)

Production is automatically deployed by GitHub Actions on merge to main.

For manual production deployment (rarely needed):

```bash
cd devops/charts
helmfile apply -l name=dashboard-cronjob
```

### 3. Verify deployment

```bash
# Check cronjob status
kubectl get cronjobs -n monitoring

# Check recent jobs
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -5

# View logs from most recent job
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100
```

## Development/Testing Deployment

For testing changes before merging to main (this is the main focus of this guide):

### 1. Build Docker image for your branch

Trigger the GitHub Action to build an image from your feature branch:

```bash
# Trigger build from a specific branch (REQUIRED for feature branches)
gh workflow run build-dashboard-image.yml --ref robb/1022-datadog

# IMPORTANT: Without --ref, the workflow runs on main branch, not your feature branch!

# Wait for build to complete (~1-2 minutes)
gh run watch

# Check the image tag (uses git SHA)
git rev-parse --short=7 HEAD
# Image tag will be: sha-<first-7-chars-of-SHA>
```

The image will be pushed to ECR as: `751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:sha-<commit-sha>`

**Important Notes:**
- The workflow must be manually triggered via `gh workflow run` or the GitHub Actions UI. It does NOT automatically build on every push to your branch.
- When building from a feature branch (not `main`), the "Deploy helm chart" step will be **skipped**. This is expected - only builds from `main` trigger automatic deployment.
- You must manually deploy using `helmfile apply` (see step 3 below) to test your feature branch changes.

### 2. Uncomment dev section in helmfile.yaml

Edit `devops/charts/helmfile.yaml` and uncomment the dev section, updating the image tag:

```yaml
- name: dashboard-cronjob-dev
  chart: ./dashboard-cronjob
  version: 0.1.0
  namespace: monitoring
  values:
    - ./dashboard-cronjob/values-dev.yaml
    - image:
        tag: "sha-abc1234"  # Update with your image tag from step 1
```

**Why helmfile?** The dev deployment MUST use helmfile (not `helm upgrade` directly) because:
- Helmfile automatically applies `values-dev.yaml` which sets `serviceAccount.create: false`
- This makes dev reuse production's service account (which has IRSA permissions)
- Using `helm upgrade` directly will create a separate service account that lacks AWS permissions

### 3. Deploy development copy

```bash
cd devops/charts
helmfile apply -l name=dashboard-cronjob-dev
```

This deploys a `-dev` copy to the same `monitoring` namespace that:
- **Reuses the production service account** (no separate IRSA setup needed)
- Tags metrics with `env:development` for filtering in Datadog
- Runs alongside production without interference
- Uses the same 15-minute schedule as production

### 4. Monitor development deployment

```bash
# Check dev cronjob
kubectl get cronjobs -n monitoring | grep dev

# Manually trigger a test job
kubectl create job --from=cronjob/dashboard-cronjob-dev-dashboard-cronjob \
  test-$(date +%s) -n monitoring

# View logs
kubectl logs -n monitoring -l job-name=test-* --tail=100
```

### 5. Cleanup

When done testing:

```bash
cd devops/charts
helmfile destroy -l name=dashboard-cronjob-dev
```

Or comment out the dev section in helmfile.yaml and run:

```bash
helmfile apply
```

**Note**: Services suffixed with `-dev` are safe to destroy and clean up.

## Deployment Options

### Using custom image tag

Deploy a specific image version (useful for testing branches):

```bash
cd devops/charts
helmfile apply -i -l name=dashboard-cronjob-dev \
  --set image.tag=my-feature-branch
```

### Changing collection schedule

Edit `values-dev.yaml` or override in helmfile:

```yaml
- name: dashboard-cronjob-dev
  # ... other config ...
  values:
    - ./dashboard-cronjob/values-dev.yaml
    - schedule: "*/5 * * * *"  # Every 5 minutes
```

### Dry run (template only)

Preview changes without deploying:

```bash
cd devops/charts
helmfile diff -l name=dashboard-cronjob-dev
```

## Monitoring

### Check deployment health

```bash
# Overall status
kubectl get all -n monitoring

# CronJob details
kubectl describe cronjob dashboard-cronjob -n monitoring

# Recent job history
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp
```

### View logs

```bash
# Logs from most recent job
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100

# Follow logs in real-time
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --follow

# Logs from specific job
kubectl logs -n monitoring job/<job-name>
```

### Manually trigger collection

```bash
# Create a manual job from the cronjob
kubectl create job --from=cronjob/dashboard-cronjob manual-run-$(date +%s) -n monitoring

# Watch it run
kubectl get jobs -n monitoring --watch

# View logs
kubectl logs -n monitoring job/manual-run-<timestamp>
```

## Troubleshooting

### Job fails to start

Check the cronjob configuration:

```bash
kubectl describe cronjob dashboard-cronjob -n monitoring
```

Check pod events:

```bash
kubectl get events -n monitoring --sort-by='.lastTimestamp'
```

### Authentication errors

The collector uses AWS Secrets Manager via IAM roles. Verify:

1. **ServiceAccount has correct IAM role annotation:**

   ```bash
   kubectl get serviceaccount -n monitoring -o yaml | grep role-arn
   # Should show: eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/dashboard-cronjob
   ```

2. **IAM role has Secrets Manager permissions:**
   - The role needs `secretsmanager:GetSecretValue` permission for:
     - `github/dashboard-token` (GitHub API token)
     - `datadog/api-key` (Datadog API key)
     - `datadog/app-key` (Datadog App key)

3. **Test secrets access manually:**
   ```bash
   # Run a test pod with the same service account
   kubectl run test-secrets --rm -i --tty \
     --image=amazon/aws-cli \
     --serviceaccount=dashboard-cronjob \
     --namespace=monitoring \
     -- secretsmanager get-secret-value --secret-id github/dashboard-token
   ```

### Metrics not appearing in Datadog

1. **Check job completed successfully:**

   ```bash
   kubectl get jobs -n monitoring
   # Look for "1/1" in COMPLETIONS column
   ```

2. **Check logs for errors:**

   ```bash
   kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100
   ```

3. **Verify metrics were submitted:** Look for "Successfully pushed N metrics to Datadog" in logs

4. **Check Datadog for metrics:**
   - Go to Datadog Metrics Explorer
   - Search for `github.*` metrics
   - Check timestamp matches recent collection

### Too many API calls / rate limiting

If you see rate limit errors:

1. **Reduce collection frequency:**

   Edit `values.yaml` and change the schedule:
   ```yaml
   schedule: "*/30 * * * *"  # Change from 15 to 30 minutes
   ```

   Then deploy:
   ```bash
   cd devops/charts
   helmfile apply -l name=dashboard-cronjob
   ```

2. **Check current API usage:**
   ```bash
   # View logs to see API call counts
   kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=200 | grep -i "rate\|limit"
   ```

## Rollback

### Rollback to previous version

```bash
# View deployment history
helm history dashboard-cronjob -n monitoring

# Rollback to previous version
helm rollback dashboard-cronjob -n monitoring

# Rollback to specific revision
helm rollback dashboard-cronjob <revision> -n monitoring
```

### Delete deployment

```bash
# Delete via helmfile
cd devops/charts
helmfile delete -l name=dashboard-cronjob

# Or use helm directly
helm uninstall dashboard-cronjob -n monitoring

# Clean up manual jobs (optional)
kubectl delete jobs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob
```

## CI/CD Integration

### Automated Production Deployment

The production deployment is fully automated via GitHub Actions:

**Workflow:** `.github/workflows/build-dashboard-image.yml`

**Triggers:**
- **Automatic:** Push to `main` branch with changes to:
  - `devops/datadog/**` (collector code)
  - `packages/gitta/**` (Git utilities)
  - `devops/charts/dashboard-cronjob/**` (Helm chart)
  - `.github/workflows/build-dashboard-image.yml` (workflow itself)

- **Manual:** Via `workflow_dispatch` for any branch:
  ```bash
  gh workflow run build-dashboard-image.yml --ref your-branch-name
  ```

**Automated Process:**
1. Builds Docker image with tag `sha-<7-digit-git-sha>`
2. Pushes image to ECR: `751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:sha-<sha>`
3. **Production only:** Deploys to production using `helm upgrade` with new image tag
4. Production deployment uses the chart's default values (no `-dev` suffix)

**Key Difference: Production vs Dev Deployment**

| Aspect | Production (CI/CD) | Dev (Manual) |
|--------|-------------------|--------------|
| Deployment method | `helm upgrade` (GitHub Actions) | `helmfile apply` (manual) |
| Configuration | Default `values.yaml` | `values-dev.yaml` (via helmfile) |
| Service account | Creates `dashboard-cronjob-dashboard-cronjob` | Reuses production SA |
| Metric tags | `env:production` | `env:development` |
| When | Automatic on merge to main | Manual for testing |

**Why the difference?**
- Production uses `helm upgrade` because it only manages one release
- Dev uses `helmfile` to properly merge `values-dev.yaml` which sets `serviceAccount.create: false`
- This is why you can't just run `helm upgrade` for dev - it won't apply the values-dev configuration

### Manual Deployment (Without CI/CD)

To deploy without merging to main, use the development deployment process above (Build → Update helmfile → Deploy with helmfile).

## Related Documentation

- **Helm Chart**: `devops/charts/dashboard-cronjob/README.md`
- **Helmfile**: `devops/charts/helmfile.yaml`
- **Collector Architecture**: `devops/datadog/docs/COLLECTORS_ARCHITECTURE.md`
- **Metrics Catalog**: `devops/datadog/docs/CI_CD_METRICS.md`
- **Adding New Collectors**: `devops/datadog/docs/ADDING_NEW_COLLECTOR.md`
