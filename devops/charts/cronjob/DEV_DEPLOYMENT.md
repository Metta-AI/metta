# Dev Cronjob Deployment Guide

This guide explains how to deploy and test cronjobs in dev mode from feature branches.

## Quick Start

To deploy a dev cronjob from your feature branch:

1. **Trigger the workflow** from your branch:
   ```bash
   gh workflow run deploy-pr-similarity-cache-cronjob.yml \
     --ref your-branch-name \
     -f dev=true
   ```

2. **Uncomment the dev release** in `devops/charts/helmfile.yaml`:
   ```yaml
   - name: pr-similarity-cache-cronjob-dev
     <<: *cronjob_template
     values:
       - schedule: "*/5 * * * *"  # Every 5 minutes for testing
       - datadog:
           service: pr-similarity-cache-refresh-dev
           env: dev
   ```

3. **Watch the deployment**:
   ```bash
   gh run watch
   ```

4. **Monitor the cronjob**:
   ```bash
   kubectl get cronjobs -n monitoring | grep dev
   kubectl get jobs -n monitoring | grep dev
   kubectl logs -n monitoring job/<job-name>
   ```

## What Happens

When you run the workflow with `dev=true`:

- **Image tag**: `metta-cronjobs:sha-abc123-{name}-dev` (e.g., `sha-abc123-pr-similarity-dev`)
- **Helm release**: `{name}-cronjob-dev` (e.g., `pr-similarity-cache-cronjob-dev`)
- **Deployment**: To `monitoring` namespace alongside prod cronjobs

## Available Workflows

- `deploy-pr-similarity-cache-cronjob.yml` - PR similarity cache builder
- `deploy-dashboard-cronjob.yml` - Dashboard metrics collector

## Cleanup

When you're done testing:

1. **Delete the dev cronjob**:
   ```bash
   helm uninstall -n monitoring pr-similarity-cache-cronjob-dev
   ```

2. **Re-comment the dev release** in helmfile.yaml

## Troubleshooting

### Job not running
```bash
# Check cronjob schedule
kubectl get cronjob -n monitoring pr-similarity-cache-cronjob-dev -o yaml

# Manually trigger a run
kubectl create job -n monitoring test-run --from=cronjob/pr-similarity-cache-cronjob-dev
```

### Check logs
```bash
# List recent jobs
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -5

# View logs
kubectl logs -n monitoring job/<job-name>
```

### Image pull errors
```bash
# Verify image exists in ECR
aws ecr describe-images --repository-name metta-cronjobs --region us-east-1 \
  --image-ids imageTag=sha-abc123-pr-similarity-dev
```

## Best Practices

1. **Test locally first**: Use Docker to test your Dockerfile changes before deploying
2. **Use short schedules**: Set `schedule: "*/5 * * * *"` for faster iteration
3. **Clean up after**: Always delete dev cronjobs when done to avoid clutter
4. **Monitor costs**: Dev cronjobs run frequently - don't forget about them!
