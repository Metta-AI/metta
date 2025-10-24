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

| Collector                   | Status         | Priority | Metrics | Description                                         |
| --------------------------- | -------------- | -------- | ------- | --------------------------------------------------- |
| [github](github/)           | ✅ Implemented | High     | 28      | PRs, commits, CI/CD, branches, developers           |
| [skypilot](skypilot/)       | ✅ Implemented | High     | 30      | Jobs, clusters, runtime stats, resource utilization |
| [asana](asana/)             | ✅ Implemented | Medium   | 14      | Project health, bugs tracking, team velocity        |
| [ec2](ec2/)                 | ✅ Implemented | High     | 19      | Instances, costs, utilization, EBS volumes          |
| [wandb](wandb/)             | ✅ Implemented | High     | 10      | Training runs, model performance, GPU hours         |
| [kubernetes](kubernetes/)   | ✅ Implemented | High     | 15      | Resource efficiency, pod health, waste tracking     |
| [health_fom](health_fom/)   | ✅ Implemented | High     | 7       | Normalized health scores (0.0-1.0) for CI/CD        |

**Total**: 123 metrics across 7 collectors

**Note**: All collectors run together on a **unified 15-minute schedule** via a single CronJob for operational
simplicity.

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

### Skypilot Collector ✅

**Implemented** - Job orchestration and compute tracking (30 metrics)

**Job Status Metrics**:

- `skypilot.jobs.running`, `skypilot.jobs.queued` - Job status
- `skypilot.jobs.failed`, `skypilot.jobs.failed_7d` - Failure tracking
- `skypilot.jobs.succeeded`, `skypilot.jobs.cancelled` - Completion status
- `skypilot.clusters.active` - Active clusters

**Runtime Distribution** (for running jobs):

- `skypilot.jobs.runtime_seconds.{min,max,avg,p50,p90,p99}` - Statistical distribution
- `skypilot.jobs.runtime_buckets.{0_1h,1_4h,4_24h,over_24h}` - Histogram buckets

**Resource Utilization**:

- `skypilot.resources.gpus.{l4,a10g,h100}_count` - GPU types in use
- `skypilot.resources.gpus.total_count` - Total GPUs
- `skypilot.resources.{spot,ondemand}_jobs` - Spot vs on-demand split

**Reliability Metrics**:

- `skypilot.jobs.with_recoveries` - Jobs that recovered from failures
- `skypilot.jobs.recovery_count.{avg,max}` - Recovery statistics

**Regional Distribution**:

- `skypilot.regions.{us_east_1,us_west_2,other}` - Geographic distribution

**Team Activity**:

- `skypilot.users.active_count` - Number of active users

**Usage**:

```bash
# Collect metrics (dry-run)
metta datadog collect skypilot

# Push metrics to Datadog
metta datadog collect skypilot --push
```

**Dashboard Ideas**:

- **Runtime heatmap**: Visualize job duration distribution to spot stuck jobs
- **Resource efficiency**: Track spot vs on-demand ratio over time
- **Alert on p99 > 48h**: Catch long-running jobs that might be stuck

### WandB Collector ✅

**Implemented** - Training experiment tracking (10 metrics)

**Key Metrics**:

- `wandb.runs.active`, `wandb.runs.completed_7d` - Run status
- `wandb.metrics.best_accuracy`, `wandb.metrics.latest_loss` - Model performance
- `wandb.training.total_gpu_hours_7d`, `wandb.training.gpu_utilization_avg` - Resource usage

**Docs**: [wandb/README.md](wandb/README.md)

### Kubernetes Collector ✅

**Implemented** - Resource efficiency and pod health (15 metrics)

**Resource Efficiency**:

- `k8s.resources.cpu_waste_cores`, `k8s.resources.memory_waste_gb` - Wasted resources
- `k8s.resources.cpu_efficiency_pct`, `k8s.resources.memory_efficiency_pct` - Utilization
- `k8s.resources.overallocated_pods` - Pods using <20% of requested resources

**Pod Health**:

- `k8s.pods.crash_looping`, `k8s.pods.failed`, `k8s.pods.pending` - Pod status
- `k8s.pods.oomkilled_24h`, `k8s.pods.high_restarts` - Stability issues

**Underutilization**:

- `k8s.pods.idle_count`, `k8s.pods.low_cpu_usage`, `k8s.pods.low_memory_usage` - Idle resources
- `k8s.deployments.zero_replicas` - Unused deployments

**Usage**:

```bash
# Collect metrics (dry-run)
uv run python devops/datadog/run_collector.py kubernetes --verbose

# Push metrics to Datadog
uv run python devops/datadog/run_collector.py kubernetes --push
```

**Docs**: [kubernetes/README.md](kubernetes/README.md)

### EC2 Collector ✅

**Implemented** - AWS infrastructure monitoring (19 metrics)

**Instance Metrics**:

- `ec2.instances.total`, `ec2.instances.running`, `ec2.instances.stopped` - Instance counts
- `ec2.instances.spot`, `ec2.instances.ondemand` - Instance lifecycle
- `ec2.instances.gpu_count`, `ec2.instances.cpu_count` - Resource counts
- `ec2.instances.idle` - Idle instance count
- `ec2.instances.avg_age_days`, `ec2.instances.oldest_age_days` - Instance age

**EBS Metrics**:

- `ec2.ebs.volumes.total`, `ec2.ebs.volumes.attached`, `ec2.ebs.volumes.unattached` - Volume counts
- `ec2.ebs.volumes.size_gb` - Total storage size
- `ec2.ebs.snapshots.total`, `ec2.ebs.snapshots.size_gb` - Snapshot tracking

**Cost Metrics**:

- `ec2.cost.running_hourly_estimate`, `ec2.cost.monthly_estimate` - Cost tracking
- `ec2.cost.spot_savings_pct` - Spot instance savings percentage

**Usage**:

```bash
# Collect metrics (dry-run)
uv run python devops/datadog/run_collector.py ec2 --verbose

# Push metrics to Datadog
uv run python devops/datadog/run_collector.py ec2 --push
```

**Setup**: Requires AWS credentials with EC2 read permissions. See [SECRETS_SETUP.md](../SECRETS_SETUP.md).

### Asana Collector ✅

**Implemented** - Project management and bugs tracking (14 metrics)

**Project Health Metrics**:

- `asana.projects.active` - Active projects count
- `asana.projects.on_track` - Projects on track
- `asana.projects.at_risk` - Projects at risk
- `asana.projects.off_track` - Projects off track

**Bugs Project Metrics** (if configured):

- `asana.projects.bugs.total_open` - Total open bugs
- `asana.projects.bugs.triage_count` - Bugs in triage
- `asana.projects.bugs.active_count` - Bugs being worked on
- `asana.projects.bugs.backlog_count` - Bugs in backlog
- `asana.projects.bugs.completed_7d` - Bugs completed in 7 days
- `asana.projects.bugs.completed_30d` - Bugs completed in 30 days
- `asana.projects.bugs.created_7d` - New bugs in 7 days
- `asana.projects.bugs.avg_age_days` - Average bug age
- `asana.projects.bugs.oldest_bug_days` - Oldest bug age

**Usage**:

```bash
# Collect metrics (dry-run)
uv run python devops/datadog/run_collector.py asana --verbose

# Push metrics to Datadog
uv run python devops/datadog/run_collector.py asana --push
```

**Setup**: See [SECRETS_SETUP.md](../SECRETS_SETUP.md) for configuring Asana access token and workspace/project IDs.

### Health FoM Collector ✅

**Implemented** - Normalized health metrics (7 metrics)

**Figure of Merit Scores** (0.0-1.0 scale):

- `health.ci.tests_passing.fom` - Unit tests passing on main
- `health.ci.benchmarks_passing.fom` - Benchmarks passing on main
- `health.ci.workflow_success_rate.fom` - CI workflow success rate
- `health.ci.duration.fom` - CI workflow duration (faster = higher score)
- `health.commits.quality.fom` - Commit quality (fewer reverts/hotfixes = higher)
- `health.prs.velocity.fom` - PR merge velocity
- `health.prs.quality.fom` - PR review coverage

**Scoring Logic**:

- `1.0` = Perfect (green)
- `0.7-1.0` = Good (green-yellow)
- `0.3-0.7` = Warning (yellow-orange)
- `0.0-0.3` = Critical (orange-red)

**Features**:

- Reads existing metrics from Datadog (no external APIs)
- Applies normalization formulas to convert raw metrics to 0.0-1.0 scores
- Used in System Health Rollup dashboards (7×7 grid)

**Usage**:

```bash
# Collect health scores (reads from Datadog)
uv run python devops/datadog/run_collector.py health_fom --verbose

# Push to Datadog
uv run python devops/datadog/run_collector.py health_fom --push
```

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
    schedule: '*/15 * * * *'

  skypilot:
    enabled: true
    schedule: '*/10 * * * *'
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
3. Add secrets to AWS Secrets Manager (see [SECRETS_SETUP.md](../SECRETS_SETUP.md))
4. Configure in Helm `values.yaml` (environment variables for configuration)
5. Test locally: `uv run python devops/datadog/run_collector.py {service} --verbose`
6. Deploy with Helm

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
- `asana/access-token`
- `datadog/api-key`
- `datadog/app-key`

## Testing

### Local Testing

```bash
# Test collector without pushing to Datadog
uv run python devops/datadog/run_collector.py github --verbose

# Push to Datadog
uv run python devops/datadog/run_collector.py github --push

# Validate secrets configuration
uv run python devops/datadog/scripts/validate_secrets.py
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

### Phase 1: Core Collectors ✅

- ✅ GitHub collector (28 metrics)
- ✅ Skypilot collector (30 metrics)
- ✅ Asana collector (14 metrics)
- ✅ EC2 collector (19 metrics)
- ✅ WandB collector (10 metrics)
- ✅ AWS Secrets Manager integration
- ✅ Deployed as Kubernetes CronJobs

### Phase 2: Additional Collectors 📋

Priority order:

1. **Custom metrics** (as needed)

### Phase 3: Enhancements 📋

- Dashboard auto-generation from metadata
- Historical data backfill
- Metric validation and schemas
- Collector dependency management
- Alert templates based on collector metrics

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
