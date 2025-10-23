# Weights & Biases (WandB) Collector

Collects training run and experiment metrics from WandB API.

**Status**: 📋 **Planned**

## Metrics to Collect

### Run Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.runs.active` | Currently running experiments | count | High |
| `wandb.runs.completed_7d` | Runs completed in last 7 days | count | Medium |
| `wandb.runs.failed_7d` | Runs failed in last 7 days | count | High |
| `wandb.runs.crashed_7d` | Runs that crashed in last 7 days | count | High |
| `wandb.runs.avg_duration_hours` | Average run duration | hours | Medium |
| `wandb.runs.total_7d` | Total runs started in last 7 days | count | Low |

### Project Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.projects.active` | Projects with runs in last 7 days | count | Medium |
| `wandb.projects.total_runs` | Total runs across all projects | count | Low |

### Performance Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.metrics.best_accuracy` | Best accuracy across recent runs | percent | Medium |
| `wandb.metrics.best_loss` | Lowest loss across recent runs | float | Medium |
| `wandb.metrics.avg_accuracy` | Average accuracy for runs in last 7d | percent | Low |
| `wandb.runs.converged_7d` | Runs that reached convergence criteria | count | Medium |

### Resource Utilization

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.compute.avg_gpu_utilization` | Average GPU utilization across active runs | percent | Medium |
| `wandb.compute.total_gpu_hours_7d` | Total GPU hours consumed in 7 days | hours | High |
| `wandb.storage.artifacts_gb` | Total artifact storage used | gb | Low |
| `wandb.storage.logs_gb` | Total log storage used | gb | Low |

### Experiment Tracking

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.experiments.active` | Active experiment groups | count | Low |
| `wandb.experiments.sweeps_running` | Active hyperparameter sweeps | count | Medium |
| `wandb.experiments.sweeps_completed_7d` | Sweeps completed in last 7 days | count | Low |

### Team Activity

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `wandb.users.active_7d` | Unique users who started runs | count | Medium |
| `wandb.runs.per_user_7d` | Average runs per user | count | Low |

## Configuration

### Required Secrets (AWS Secrets Manager)

- `wandb/api-key` - WandB API key (or personal access token)

### Environment Variables

- `WANDB_ENTITY` - WandB team/entity name (default: from config)
- `WANDB_PROJECT` - Default project to track (optional, can query all)

### Collection Schedule

**Recommended**: Every 30 minutes (`*/30 * * * *`)

- Training runs are long-running (hours/days)
- Less frequent updates needed than infrastructure metrics
- Reduces API calls to WandB

## API Endpoints

WandB Public API:

```
GET /api/v1/{entity}/runs                    # List runs
GET /api/v1/{entity}/{project}/runs          # Project-specific runs
GET /api/v1/{entity}/{project}/runs/{run_id} # Run details
GET /api/v1/{entity}/projects                # List projects
GET /api/v1/{entity}/sweeps                  # List sweeps
```

Reference: https://docs.wandb.ai/ref/python/public-api

## Implementation Notes

### Sample Metric Implementation

```python
from datetime import datetime, timedelta, timezone
import wandb

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret


def _get_wandb_api():
    """Initialize WandB API client."""
    api_key = get_secret("wandb/api-key")
    return wandb.Api(api_key=api_key)


@metric("wandb.runs.active", unit="count")
def get_active_runs() -> int:
    """Number of currently running WandB experiments."""
    api = _get_wandb_api()
    entity = os.environ.get("WANDB_ENTITY", "softmax-research")

    # Query runs with state=running
    runs = api.runs(
        path=f"{entity}",
        filters={"state": "running"}
    )
    return len(list(runs))


@metric("wandb.runs.completed_7d", unit="count")
def get_completed_runs_7d() -> int:
    """Runs completed in last 7 days."""
    api = _get_wandb_api()
    entity = os.environ.get("WANDB_ENTITY", "softmax-research")

    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    runs = api.runs(
        path=f"{entity}",
        filters={
            "state": "finished",
            "created_at": {"$gt": since}
        }
    )
    return len(list(runs))


@metric("wandb.metrics.best_accuracy", unit="percent")
def get_best_accuracy() -> float | None:
    """Best accuracy across recent runs."""
    api = _get_wandb_api()
    entity = os.environ.get("WANDB_ENTITY", "softmax-research")

    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    runs = api.runs(
        path=f"{entity}",
        filters={
            "state": "finished",
            "created_at": {"$gt": since}
        }
    )

    accuracies = []
    for run in runs:
        if "accuracy" in run.summary:
            accuracies.append(run.summary["accuracy"])

    if not accuracies:
        return None

    return max(accuracies) * 100  # Convert to percentage
```

## Dashboard Widgets

Recommended visualizations:

1. **Run Status Overview**
   - Query value: Active runs
   - Timeseries: Completed vs Failed runs (7d)
   - Query value: Success rate %

2. **Performance Tracking**
   - Query value: Best accuracy (recent)
   - Query value: Best loss (recent)
   - Timeseries: Average accuracy over time

3. **Resource Usage**
   - Query value: Total GPU hours (7d)
   - Query value: Average GPU utilization
   - Timeseries: GPU hours per day

4. **Team Activity**
   - Query value: Active users (7d)
   - Timeseries: Runs per day
   - Top list: Most active users

## Alerting Recommendations

```yaml
# High failure rate
Query: (sum:wandb.runs.failed_7d{} / sum:wandb.runs.total_7d{}) > 0.2
Alert: WandB run failure rate above 20%

# No active runs (might indicate issues)
Query: avg(last_2h):sum:wandb.runs.active{} == 0
Alert: No active training runs for 2+ hours

# High crash rate
Query: (sum:wandb.runs.crashed_7d{} / sum:wandb.runs.total_7d{}) > 0.1
Alert: Run crash rate above 10%

# Low GPU utilization
Query: avg(last_1h):avg:wandb.compute.avg_gpu_utilization{} < 50
Alert: Average GPU utilization below 50%
```

## Dependencies

- `wandb` - Official WandB Python SDK
- `boto3` - AWS Secrets Manager access

## Challenges & Considerations

1. **API Rate Limits**: WandB may have rate limits, cache responses where possible
2. **Large Projects**: Projects with thousands of runs may need pagination
3. **Metric Variability**: Different experiments track different metrics (accuracy, loss, etc.)
4. **Historical Data**: Decide lookback window (7d, 30d) based on training frequency

## Related Documentation

- [WandB Public API Docs](https://docs.wandb.ai/ref/python/public-api)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)

## Next Steps

1. Obtain WandB API key (personal access token)
2. Store secret in AWS Secrets Manager
3. Verify entity/project names
4. Implement collector following template
5. Test with actual project data
6. Deploy to staging
7. Create Datadog dashboard

## Maintenance

- **Owner**: ML Team / Research
- **Priority**: Medium (training visibility important)
- **API Version**: WandB Public API v1
- **Estimated Effort**: 3-4 hours implementation
- **Last Updated**: 2025-10-22
