# Dashboard CronJob

Runs the softmax dashboard metrics collection every 15 minutes and pushes to Datadog.

## What it does

Collects system health metrics:
- GitHub CI status (tests passing/failing)
- Hotfix and revert commit counts
- More metrics can be added in `softmax/src/softmax/dashboard/metrics.py`

## Deploy

Automatically deployed by GitHub Actions when you merge to `main`.

Or deploy manually:
```bash
cd devops/charts
helmfile apply -l name=dashboard-cronjob
```

## Add a new script

To add another cronjob (e.g., weekly reports):

1. **Add your script** to `softmax/src/softmax/` (uses the same Docker image)

2. **Add to `helmfile.yaml`** (reuse the SAME chart):
   ```yaml
   - name: weekly-report
     chart: ./dashboard-cronjob  # Same chart, different config
     version: 0.1.0
     namespace: monitoring
     values:
       - schedule: "0 9 * * MON"  # Monday at 9am
         command: ["uv", "run", "python", "-m", "softmax.reports.weekly"]
   ```

That's it! One chart, multiple deployments with different schedules and commands.

## Configuration

Edit `values.yaml`:
- `schedule`: Cron schedule (default: every 15 minutes)
- `command`: Command to run in the container
- `resources`: CPU/memory limits
- `datadog.env`: Environment tag for metrics

## Troubleshooting

Check logs:
```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100
```

View job history:
```bash
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp
```
