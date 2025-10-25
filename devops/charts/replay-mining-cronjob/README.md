# Replay Mining CronJob

Kubernetes CronJob that automatically mines game replays into supervised learning datasets daily.

## Overview

**Schedule**: Daily at 2 AM UTC
**Output**: `s3://softmax-public/datasets/replays/replays_YYYYMMDD.parquet`
**Process**: Queries yesterday's replays → Downloads → Processes → Saves to S3

For **Python API and usage documentation**, see the docstrings in `metta/tools/replay_dataset/`.

## Quick Deploy

```bash
cd devops/charts/
helmfile apply -l name=replay-mining-cronjob
```

## Monitoring

```bash
# Check cronjob status
kubectl get cronjobs -n monitoring

# See recent jobs
kubectl get jobs -n monitoring -l app.kubernetes.io/name=replay-mining-cronjob

# Check logs
kubectl logs -n monitoring job/replay-mining-cronjob-XXXXXXXX

# List datasets in S3
aws s3 ls s3://softmax-public/datasets/replays/
```

## Configuration

Configured in `values.yaml`:

```yaml
schedule: "0 2 * * *"  # Daily at 2 AM UTC

command: ["uv", "run", "python", "-m", "metta.tools.replay_dataset.replay_mine"]

resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 8Gi

serviceAccount:
  create: true
```

## Manual Triggers

### Single Date Backfill

```bash
# Create one-off job from cronjob template
kubectl create job --from=cronjob/replay-mining-cronjob \
  backfill-20251015 -n monitoring

# Check status
kubectl get jobs -n monitoring
kubectl logs -n monitoring job/backfill-20251015
```

### Multiple Dates

Run locally with a loop (faster than creating multiple K8s jobs):

```bash
for i in {1..7}; do
  date=$(date -d "$i days ago" +%Y-%m-%d)
  uv run python -m metta.tools.replay_dataset.replay_mine --date $date
done
```

### Job Failed

```bash
# Get logs from most recent job
kubectl logs -n monitoring job/$(kubectl get jobs -n monitoring \
  -l app.kubernetes.io/name=replay-mining-cronjob \
  --sort-by=.status.startTime -o jsonpath='{.items[-1].metadata.name}')
```

#
