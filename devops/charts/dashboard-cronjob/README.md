# Datadog Metrics Collector CronJob

Runs the Datadog metrics collectors every 15 minutes and pushes metrics to Datadog.

## What it does

Collects GitHub repository metrics using the modular collector architecture:
- Pull request metrics (open, merged, closed, cycle time, stale PRs, review coverage)
- Branch metrics (active branch count)
- Commit & code metrics (commits, hotfixes, reverts, lines added/deleted, files changed)
- CI/CD metrics (workflow runs, failures, duration percentiles)
- Developer metrics (active developers, commits per developer)

**Total: 25 metrics** - See `devops/datadog/docs/CI_CD_METRICS.md` for complete catalog

## Deploy

Automatically deployed by GitHub Actions when you merge to `main`.

Or deploy manually:
```bash
cd devops/charts
helmfile apply -l name=dashboard-cronjob
```

## Add a new collector

To add another collector (e.g., AWS infrastructure metrics):

1. **Create your collector** following the modular architecture:
   - See `devops/datadog/docs/ADDING_NEW_COLLECTOR.md` for step-by-step guide
   - Create `devops/datadog/collectors/aws/collector.py` extending `BaseCollector`
   - Implement `collect_metrics()` method

2. **Add to `helmfile.yaml`** (reuse the SAME chart):
   ```yaml
   - name: aws-collector
     chart: ./dashboard-cronjob  # Same chart, different collector
     version: 0.1.0
     namespace: monitoring
     values:
       - schedule: "*/15 * * * *"  # Every 15 minutes
         command: ["uv", "run", "python", "devops/datadog/run_collector.py", "aws", "--push"]
         datadog:
           service: "aws-collector"
   ```

That's it! One chart, multiple collectors with different schedules and commands.

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
