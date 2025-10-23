# Datadog Collector Deployment Guide

This guide covers deploying the GitHub metrics collector to Kubernetes/EKS.

## Overview

The GitHub collector runs as a Kubernetes CronJob that:
- Executes every 15 minutes
- Collects 25 GitHub repository metrics
- Pushes metrics to Datadog
- Uses AWS Secrets Manager for authentication (GitHub token, Datadog API keys)

## Prerequisites

- AWS CLI configured with appropriate credentials
- kubectl installed
- helm installed
- Access to EKS cluster (IAM permissions)

## Quick Start

### 1. Setup kubectl access

```bash
./devops/k8s/setup-k8s.sh main
```

This will:
- Configure kubectl for the EKS cluster
- Verify access
- Show available namespaces

### 2. Deploy to production

```bash
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main
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

## Staging Environment

For testing changes before production deployment:

### 1. Create staging namespace

```bash
kubectl create namespace monitoring-staging
```

Note: The cronjob uses IAM roles for AWS Secrets Manager access, so no additional secrets need to be copied to the staging namespace.

### 2. Deploy to staging

```bash
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob-staging \
  --namespace monitoring-staging \
  --cluster main \
  --set schedule="*/30 * * * *" \
  --set datadog.env="staging" \
  --create-namespace
```

### 3. Monitor staging deployment

```bash
# Watch for jobs to be created
kubectl get jobs -n monitoring-staging --watch

# View logs
kubectl logs -n monitoring-staging -l app.kubernetes.io/name=dashboard-cronjob --tail=100

# Manually trigger a job for immediate testing
kubectl create job --from=cronjob/dashboard-cronjob-staging test-run-$(date +%s) -n monitoring-staging
```

## Deployment Options

### Using custom image tag

Deploy a specific image version (useful for testing branches):

```bash
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --set image.tag=my-feature-branch
```

### Changing collection schedule

```bash
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --set schedule="*/5 * * * *"  # Every 5 minutes
```

### Dry run (template only)

Preview changes without deploying:

```bash
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --dry-run
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

3. **Verify metrics were submitted:**
   Look for "Successfully pushed N metrics to Datadog" in logs

4. **Check Datadog for metrics:**
   - Go to Datadog Metrics Explorer
   - Search for `github.*` metrics
   - Check timestamp matches recent collection

### Too many API calls / rate limiting

If you see rate limit errors:

1. **Reduce collection frequency:**
   ```bash
   ./devops/k8s/deploy-helm-chart.sh \
     --chart devops/charts/dashboard-cronjob \
     --release dashboard-cronjob \
     --namespace monitoring \
     --set schedule="*/30 * * * *"  # Change from 15 to 30 minutes
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
# Delete the helm release
helm uninstall dashboard-cronjob -n monitoring

# Clean up manual jobs (optional)
kubectl delete jobs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob
```

## CI/CD Integration

The deployment is automated via GitHub Actions on push to `main`:

1. **Workflow**: `.github/workflows/build-dashboard-image.yml`
2. **Triggers**: Push to `main` with changes to:
   - `softmax/**`
   - `common/**`
   - `devops/datadog/**`
   - `install.sh`

3. **Process**:
   - Builds Docker image
   - Pushes to ECR
   - Deploys to production using `helm upgrade`

To deploy without merging to main, use the manual deployment process above.

## Related Documentation

- **Helm Chart**: `devops/charts/dashboard-cronjob/README.md`
- **Collector Architecture**: `devops/datadog/docs/COLLECTORS_ARCHITECTURE.md`
- **Metrics Catalog**: `devops/datadog/docs/CI_CD_METRICS.md`
- **K8s Helpers**: `devops/k8s/README.md`
- **Adding New Collectors**: `devops/datadog/docs/ADDING_NEW_COLLECTOR.md`
