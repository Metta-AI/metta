# Metta Infra Health Collectors

This package powers the Datadog ingestion pipeline described in `devops/datadog/DATADOG_INGESTION_PLAN.md`.

## CLI

```
uv run python -m devops.datadog.cli list
uv run python -m devops.datadog.cli collect ci --dry-run
uv run python -m devops.datadog.cli collect ci --push
```

`--push` submits via the Datadog v2 Metrics API (API/app keys are pulled from `DD_*` env vars or AWS Secrets Manager secrets `datadog/api-key` + `datadog/app-key`).

## Collectors

| Slug | Description | Schedule (Helm) | Notes |
| --- | --- | --- | --- |
| `ci` | Pulls GitHub workflow stats, flaky counts, cancellation counts, and weekly PR label metrics. | `*/10 * * * *` | Requires `GITHUB_DASHBOARD_TOKEN` (or `GITHUB_TOKEN`) with `repo` scope for higher rate limits. |
| `training` | Converts structured stable-suite outputs into metrics. | `0 * * * *` | Set `TRAINING_HEALTH_FILE` (JSON) when publishing training results. |
| `eval` | Converts structured remote/local eval outputs into metrics. | `0 * * * *` | Set `EVAL_HEALTH_FILE` (JSON) when publishing evaluation results. |

### Training/Eval file schema

See `devops/datadog/data/README.md`. Pipelines should write JSON lists (or `{ "records": [...] }`) where each entry contains:

```json
{
  "metric": "hearts",
  "workflow_name": "multigpu_arena_basic_easy_shaped",
  "task": "runs_successfully",
  "check": "reward_threshold",
  "condition": "> 0.5",
  "value": 0.91,
  "status": "pass",
  "timestamp": "2025-01-14T15:00:00Z",
  "tags": {
    "commit": "abc123",
    "workflow": "training"
  }
}
```

## Kubernetes deployments

Three CronJobs are defined via `devops/charts/helmfile.yaml` using the shared `cronjob` chart.

| Release | Command | Datadog service |
| --- | --- | --- |
| `dashboard-cronjob` | `collect ci --push` | `dashboard-collector` |
| `training-health-cronjob` | `collect training --push` | `training-health-collector` |
| `eval-health-cronjob` | `collect eval --push` | `eval-health-collector` |

Each pod inherits Datadog tags (`DD_ENV`, `DD_SERVICE`, `DD_VERSION`) and supports additional env vars via `extraEnv`.

## Next steps

1. Wire the stable-suite + evaluation pipelines to publish JSON snapshots and mount them in the CronJobs (via S3 sync, EFS, or HTTP endpoints).
2. Build the Datadog dashboard per Nishad's mock (heatmap view + composite tiles).
3. Enable Datadog TV/presentation mode on the office display.
