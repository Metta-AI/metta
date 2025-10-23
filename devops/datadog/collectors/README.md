# Data Collectors

Modular metric collectors for various services, all submitting to Datadog.

## Overview

Each collector is a self-contained module that:
1. Authenticates with its service (GitHub, Skypilot, WandB, etc.)
2. Fetches relevant data via API
3. Transforms data into Datadog metrics
4. Handles errors gracefully
5. Reports its own health

All collectors share a common base class and follow consistent patterns.

## Available Collectors

| Collector | Status | Priority | Metrics | Schedule | Description |
|-----------|--------|----------|---------|----------|-------------|
| [github](github/) | ✅ Implemented | High | 17+ | 15 min | PRs, commits, CI/CD, branches, developers |
| [skypilot](skypilot/) | 📋 Planned | High | 15+ | 10 min | Jobs, clusters, compute costs |
| [wandb](wandb/) | 📋 Planned | Medium | 15+ | 30 min | Training runs, experiments, GPU hours |
| [ec2](ec2/) | 📋 Planned | High | 20+ | 5 min | Instances, costs, utilization |
| [asana](asana/) | 📋 Planned | Low | 15+ | 6 hours | Tasks, projects, velocity |

## Quick Reference

### GitHub Collector ✅

**Implemented** - Currently in `softmax/dashboard/metrics.py`, needs migration

**Key Metrics**:
- `github.prs.open`, `github.prs.merged_7d` - PR tracking
- `github.commits.total_7d`, `github.commits.per_developer_7d` - Commit activity
- `github.ci.workflow_runs_7d`, `github.ci.failed_workflows_7d` - CI/CD health
- `github.branches.active` - Branch management
- `github.developers.active_7d` - Team activity

**Docs**: [github/README.md](github/README.md) | [CI/CD Metrics](../docs/CI_CD_METRICS.md)

### Skypilot Collector 📋

**Planned** - Job orchestration and compute tracking

**Key Metrics**:
- `skypilot.jobs.running`, `skypilot.jobs.queued` - Job status
- `skypilot.compute.cost_7d` - Cost tracking
- `skypilot.clusters.active` - Resource usage
- `skypilot.jobs.avg_duration_minutes` - Performance

**Docs**: [skypilot/README.md](skypilot/README.md)

### WandB Collector 📋

**Planned** - Training experiment tracking

**Key Metrics**:
- `wandb.runs.active`, `wandb.runs.completed_7d` - Run status
- `wandb.metrics.best_accuracy` - Model performance
- `wandb.compute.total_gpu_hours_7d` - Resource usage
- `wandb.users.active_7d` - Team activity

**Docs**: [wandb/README.md](wandb/README.md)

### EC2 Collector 📋

**Planned** - AWS infrastructure monitoring

**Key Metrics**:
- `ec2.instances.running`, `ec2.instances.spot` - Instance counts
- `ec2.cost.running_hourly`, `ec2.cost.monthly_estimate` - Cost tracking
- `ec2.utilization.idle_instances` - Waste detection
- `ec2.ebs.unattached_volumes` - Orphaned resources

**Docs**: [ec2/README.md](ec2/README.md)

### Asana Collector 📋

**Planned** - Project management metrics (low priority)

**Key Metrics**:
- `asana.tasks.open`, `asana.tasks.overdue` - Task tracking
- `asana.tasks.completion_rate_7d` - Team velocity
- `asana.projects.at_risk` - Project health
- `asana.tasks.high_priority` - Priority distribution

**Docs**: [asana/README.md](asana/README.md)

## Architecture

### Common Base Class

All collectors inherit from `BaseCollector`:

```python
from devops.datadog.collectors.base import BaseCollector

class MyCollector(BaseCollector):
    @property
    def name(self) -> str:
        return "myservice"

    def collect_metrics(self) -> dict[str, float]:
        # Return metrics dict
        return {"myservice.metric": 42.0}
```

### Metric Decorator

Metrics are registered using `@metric` decorator:

```python
from devops.datadog.common.decorators import metric

@metric("service.category.metric_name", unit="count")
def get_metric() -> int:
    # Fetch and return metric value
    return 100
```

### Automatic Features

Every collector automatically gets:
- **Health metrics**: `collector.{name}.duration_seconds`, `collector.{name}.error_count`
- **Error handling**: Individual metric failures don't crash collector
- **Logging**: Structured logging with context
- **Datadog submission**: Automatic metric batching and submission

## Deployment

All collectors deploy as Kubernetes CronJobs via single Helm chart:

```yaml
# devops/charts/datadog-collectors/values.yaml
collectors:
  github:
    enabled: true
    schedule: "*/15 * * * *"

  skypilot:
    enabled: true
    schedule: "*/10 * * * *"
```

Deploy:
```bash
cd devops/charts
helm upgrade --install datadog-collectors \
  ./datadog-collectors \
  --namespace monitoring
```

## Adding a New Collector

See [ADDING_NEW_COLLECTOR.md](../docs/ADDING_NEW_COLLECTOR.md) for step-by-step guide.

Quick steps:
1. Create `collectors/{service}/` directory
2. Implement `collector.py` (inherit `BaseCollector`)
3. Define metrics in `metrics.py` using `@metric` decorator
4. Add secrets to AWS Secrets Manager
5. Configure in Helm `values.yaml`
6. Test locally: `python -m devops.datadog.scripts.run_collector {service}`
7. Deploy with Helm

## Secrets Management

All credentials stored in AWS Secrets Manager:

```
{service}/api-key
{service}/api-url
{service}/access-token
```

Examples:
- `github/dashboard-token`
- `skypilot/api-key`
- `wandb/api-key`
- `ec2/access-key-id`
- `asana/personal-access-token`

## Testing

### Local Testing

```bash
# Test collector without pushing to Datadog
python -m devops.datadog.scripts.run_collector github --verbose

# Push to Datadog
python -m devops.datadog.scripts.run_collector github --push
```

### Unit Tests

```bash
pytest devops/datadog/tests/collectors/test_github.py -v
```

## Monitoring Collector Health

Each collector emits health metrics:

```yaml
# Alert on collector failure
Query: avg(last_30m):sum:collector.github.error_count{} > 5
Alert: GitHub collector experiencing high error rate

# Alert on missing data
Query: avg(last_30m):sum:github.prs.open{} < 1
Alert: GitHub collector hasn't reported metrics in 30 minutes
```

## Documentation

- [Collectors Architecture](../docs/COLLECTORS_ARCHITECTURE.md) - System design and principles
- [Adding New Collector](../docs/ADDING_NEW_COLLECTOR.md) - Implementation guide
- [Datadog Integration Analysis](../docs/DATADOG_INTEGRATION_ANALYSIS.md) - Current state
- [CI/CD Metrics](../docs/CI_CD_METRICS.md) - GitHub metrics catalog

## Roadmap

### Phase 1: Current State ✅
- GitHub collector implemented (in `softmax/dashboard`)
- 17 metrics collecting
- Deployed as CronJob

### Phase 2: Architecture Migration 📋
- Refactor GitHub to new structure
- Implement base collector pattern
- Create shared utilities

### Phase 3: New Collectors 📋
Priority order:
1. **Skypilot** (compute cost visibility)
2. **EC2** (infrastructure visibility)
3. **WandB** (training visibility)
4. **Asana** (optional, if needed)

### Phase 4: Enhancements 📋
- Dashboard auto-generation from metadata
- Historical data backfill
- Metric validation and schemas
- Collector dependency management

## Contributing

When adding a new collector:
1. Create comprehensive README in collector directory
2. List all planned metrics with priorities
3. Document API endpoints and authentication
4. Include sample code and dashboard recommendations
5. Add alerting recommendations

## Questions?

See documentation or ask in:
- Slack: #datadog-collectors
- Issues: Tag with `datadog-collector`
