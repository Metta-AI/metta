# Softmax CronJobs

Helm chart for helm releases that should run on the docker image `softmax-dashboard`, built from softmax/Dockerfile.

## Configuration

In `values.yaml`:

- `schedule`: Cron schedule (default: every 15 minutes)
- `resources`: CPU/memory limits
- `datadog.env`: Environment tag for metrics

## Troubleshooting

Check logs for e.g. dashboard-cronjob release:

```bash
kubectl logs -n cronjobs -l app.kubernetes.io/name=dashboard-cronjob --tail=100
```

View job history:

```bash
kubectl get jobs -n cronjobs --sort-by=.metadata.creationTimestamp
```
