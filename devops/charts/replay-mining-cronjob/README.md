# Replay Mining CronJob

Kubernetes CronJob that automatically mines game replays into supervised learning datasets daily.

## Overview

**Schedule**: Daily at 2 AM UTC
**Output**: `s3://softmax-public/datasets/replays/replays_YYYYMMDD.parquet`
**Process**: Queries yesterday's replays → Downloads → Processes → Saves to S3

For **Python API and usage documentation**, see the docstrings in `metta/tools/replay/`.

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

command: ["uv", "run", "python", "-m", "metta.tools.replay.mine"]

resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 8Gi

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/replay-mining-cronjob
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
  uv run python -m metta.tools.replay.replay_mine --date $date
done
```

## Troubleshooting

### Job Failed

```bash
# Get logs from most recent job
kubectl logs -n monitoring job/$(kubectl get jobs -n monitoring \
  -l app.kubernetes.io/name=replay-mining-cronjob \
  --sort-by=.status.startTime -o jsonpath='{.items[-1].metadata.name}')
```

### No Replays Found

- Check if evaluations ran on that date
- Verify date format is YYYY-MM-DD
- Query stats API: `https://api.observatory.softmax-research.net`

### S3 Permission Errors

- Verify IAM role exists: `devops/tf/eks/replay-mining.tf`
- Check service account annotation matches IAM role ARN
- Role needs: `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`

### Pod OOM (Out of Memory)

Increase memory limits in `values.yaml`:

```yaml
resources:
  limits:
    memory: 16Gi  # Increased from 8Gi
```

## IAM Permissions

The cronjob uses IRSA (IAM Roles for Service Accounts):

**Role**: `replay-mining-cronjob` (defined in Terraform)
**Permissions**:
- Read replays from S3
- Write datasets to S3
- List bucket contents

See [devops/tf/eks/replay-mining.tf](../../tf/eks/replay-mining.tf) for full configuration.

## Helm Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `schedule` | Cron schedule | `"0 2 * * *"` (2 AM UTC) |
| `command` | Command to run | `["uv", "run", "python", "-m", "metta.tools.replay.mine"]` |
| `image.registry` | Docker registry | `751442549699.dkr.ecr.us-east-1.amazonaws.com` |
| `image.name` | Image name | `metta-policy-evaluator` |
| `image.tag` | Image tag | `latest` |
| `resources.limits.cpu` | CPU limit | `2000m` |
| `resources.limits.memory` | Memory limit | `8Gi` |
| `concurrencyPolicy` | Concurrent execution policy | `Forbid` |
| `successfulJobsHistoryLimit` | Keep N successful jobs | `3` |
| `failedJobsHistoryLimit` | Keep N failed jobs | `3` |

## Related Documentation

- **Python API**: Import from `metta.tools.replay` package
- **IAM Configuration**: [devops/tf/eks/replay-mining.tf](../../tf/eks/replay-mining.tf)
- **Helm Template**: [templates/cronjob.yaml](templates/cronjob.yaml)
- **Usage**: `uv run python -m metta.tools.replay.replay_mine --help`
