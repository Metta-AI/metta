# Kubernetes Deployment Helpers

This directory contains helper scripts for deploying applications to Kubernetes/EKS.

## Overview

The Metta project uses:
- **AWS EKS** for Kubernetes hosting
- **Helm** for package management
- **Helmfile** for declarative deployment management
- **GitHub Actions** for automated CI/CD

## Scripts

### `deploy-helm-chart.sh`
Deploy a Helm chart to a specific namespace with customizable values.

```bash
# Deploy dashboard cronjob to production
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main

# Deploy to staging with custom image tag
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring-staging \
  --cluster main \
  --set image.tag=my-test-branch

# Dry run (template only, don't deploy)
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main \
  --dry-run
```

### `setup-k8s.sh`
Set up kubectl and helm configuration for EKS access.

```bash
# Setup for main cluster
./devops/k8s/setup-k8s.sh main

# Setup for specific region
./devops/k8s/setup-k8s.sh main us-west-2
```

## Deployment Workflow

### Automated (CI/CD)
When you push to `main`, GitHub Actions automatically:
1. Builds the Docker image
2. Pushes to ECR
3. Deploys to production using `helm upgrade`

See `.github/workflows/build-dashboard-image.yml` for details.

### Manual Deployment

For testing or manual deployment:

1. **Setup kubectl access:**
   ```bash
   ./devops/k8s/setup-k8s.sh main
   ```

2. **Deploy the chart:**
   ```bash
   ./devops/k8s/deploy-helm-chart.sh \
     --chart devops/charts/dashboard-cronjob \
     --release dashboard-cronjob \
     --namespace monitoring \
     --cluster main
   ```

3. **Monitor the deployment:**
   ```bash
   # Check cronjob status
   kubectl get cronjobs -n monitoring

   # Check recent jobs
   kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp

   # View logs
   kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100
   ```

### Staging Environment

To deploy to a staging environment:

1. **Create staging namespace:**
   ```bash
   kubectl create namespace monitoring-staging
   ```

2. **Copy secrets to staging:**
   ```bash
   # The cronjob needs the same IAM role for AWS Secrets Manager access
   # No additional secrets needed as authentication happens via IAM
   ```

3. **Deploy to staging:**
   ```bash
   ./devops/k8s/deploy-helm-chart.sh \
     --chart devops/charts/dashboard-cronjob \
     --release dashboard-cronjob-staging \
     --namespace monitoring-staging \
     --cluster main \
     --set schedule="*/30 * * * *"  # Run every 30 minutes instead of 15
   ```

## Troubleshooting

### Check cronjob configuration
```bash
kubectl get cronjob dashboard-cronjob -n monitoring -o yaml
```

### View recent job history
```bash
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp
```

### Debug failed jobs
```bash
# Get the failed job name
kubectl get jobs -n monitoring

# View logs from the failed job
kubectl logs -n monitoring job/<job-name>
```

### Force trigger a job manually
```bash
kubectl create job --from=cronjob/dashboard-cronjob manual-run-$(date +%s) -n monitoring
```

### Rollback a deployment
```bash
helm rollback dashboard-cronjob -n monitoring
```

## Prerequisites

- AWS CLI configured with appropriate credentials
- kubectl installed
- helm installed
- Access to EKS cluster (IAM permissions)

## Related Files

- **Helm Chart**: `devops/charts/dashboard-cronjob/`
- **Helmfile**: `devops/charts/helmfile.yaml`
- **CI/CD**: `.github/workflows/build-dashboard-image.yml`
- **Collector Code**: `devops/datadog/`
