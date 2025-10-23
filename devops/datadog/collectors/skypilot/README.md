# Skypilot Collector

Collects job execution and compute cost metrics from Skypilot API.

**Status**: 📋 **Planned**

## Metrics to Collect

### Job Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `skypilot.jobs.running` | Currently running jobs | count | High |
| `skypilot.jobs.queued` | Jobs waiting to start | count | High |
| `skypilot.jobs.completed_7d` | Jobs completed in last 7 days | count | Medium |
| `skypilot.jobs.failed_7d` | Jobs failed in last 7 days | count | High |
| `skypilot.jobs.success_rate_7d` | Job success rate (%) | percent | Medium |
| `skypilot.jobs.avg_duration_minutes` | Average job duration for completed jobs | minutes | Medium |
| `skypilot.jobs.avg_queue_time_minutes` | Average time in queue before starting | minutes | Low |
| `skypilot.jobs.longest_running_hours` | Duration of longest running job | hours | Low |

### Cluster Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `skypilot.clusters.active` | Number of active clusters | count | High |
| `skypilot.clusters.idle` | Clusters with no running jobs | count | Medium |
| `skypilot.clusters.total_nodes` | Total compute nodes across all clusters | count | Medium |
| `skypilot.clusters.avg_utilization_percent` | Average cluster CPU utilization | percent | High |

### Cost Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `skypilot.compute.cost_7d` | Total compute cost in last 7 days | usd | High |
| `skypilot.compute.cost_24h` | Compute cost in last 24 hours | usd | High |
| `skypilot.compute.cost_per_job_avg` | Average cost per job | usd | Medium |
| `skypilot.compute.estimated_monthly_cost` | Projected monthly cost based on 7-day avg | usd | High |
| `skypilot.storage.cost_7d` | Storage cost in last 7 days | usd | Low |

### Instance Metrics

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `skypilot.instances.running` | Currently running instances | count | Medium |
| `skypilot.instances.spot` | Running spot instances | count | Low |
| `skypilot.instances.on_demand` | Running on-demand instances | count | Low |
| `skypilot.instances.preempted_7d` | Spot instances preempted in last 7 days | count | Medium |

### Training Run Metrics (if available)

| Metric | Description | Unit | Priority |
|--------|-------------|------|----------|
| `skypilot.training.runs_active` | Active training runs | count | Medium |
| `skypilot.training.runs_completed_7d` | Training runs completed in 7 days | count | Low |
| `skypilot.training.avg_cost_per_run` | Average cost per training run | usd | Medium |

## Configuration

### Required Secrets (AWS Secrets Manager)

- `skypilot/api-key` - Skypilot API authentication key
- `skypilot/api-url` - Base URL for Skypilot API

### Environment Variables

- `SKYPILOT_REGION` - Default region for queries (optional)
- `SKYPILOT_CLUSTER_FILTER` - Filter clusters by tag/name (optional)

### Collection Schedule

**Recommended**: Every 10 minutes (`*/10 * * * *`)

- Job status changes frequently
- Cost tracking needs regular updates
- Cluster utilization important for optimization

## API Endpoints (Expected)

Based on typical job orchestration APIs:

```
GET /api/v1/jobs                    # List jobs with filters
GET /api/v1/jobs/{id}              # Job details
GET /api/v1/clusters               # List clusters
GET /api/v1/clusters/{id}/stats    # Cluster utilization
GET /api/v1/billing/costs          # Cost data with date filters
GET /api/v1/instances              # Instance information
```

## Implementation Notes

### Sample Metric Implementation

```python
from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret
import httpx


def _get_api_client() -> httpx.Client:
    """Create authenticated Skypilot API client."""
    api_key = get_secret("skypilot/api-key")
    api_url = get_secret("skypilot/api-url")

    return httpx.Client(
        base_url=api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        timeout=30.0,
    )


@metric("skypilot.jobs.running", unit="count")
def get_running_jobs() -> int:
    """Number of currently running Skypilot jobs."""
    with _get_api_client() as client:
        response = client.get("/jobs", params={"status": "running"})
        response.raise_for_status()
        return len(response.json()["jobs"])


@metric("skypilot.compute.cost_7d", unit="usd")
def get_compute_cost_7d() -> float:
    """Total compute cost in last 7 days (USD)."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with _get_api_client() as client:
        response = client.get("/billing/costs", params={"since": since})
        response.raise_for_status()
        return response.json()["total_cost"]
```

## Dashboard Widgets

Recommended visualizations:

1. **Job Status Overview**
   - Query value: Running jobs
   - Query value: Queued jobs
   - Timeseries: Jobs completed vs failed (7d)

2. **Cost Tracking**
   - Query value: Cost (24h)
   - Query value: Projected monthly cost
   - Timeseries: Daily cost trend

3. **Cluster Efficiency**
   - Query value: Active clusters
   - Query value: Average utilization %
   - Timeseries: Cluster count over time

4. **Job Performance**
   - Query value: Average job duration
   - Timeseries: Job success rate
   - Top list: Most expensive jobs

## Alerting Recommendations

```yaml
# High cost alert
Query: avg(last_24h):sum:skypilot.compute.cost_24h{} > 1000
Alert: Daily compute cost exceeds $1000

# Low utilization alert
Query: avg(last_1h):avg:skypilot.clusters.avg_utilization_percent{} < 20
Alert: Cluster utilization below 20% for >1 hour

# High failure rate
Query: (sum:skypilot.jobs.failed_7d{} / sum:skypilot.jobs.completed_7d{}) > 0.1
Alert: Job failure rate above 10%

# Idle clusters
Query: avg(last_30m):sum:skypilot.clusters.idle{} > 2
Alert: More than 2 idle clusters for 30+ minutes
```

## Dependencies

- `httpx` - HTTP client
- `boto3` - AWS Secrets Manager access
- `datetime`, `timedelta` - Date filtering

## Related Documentation

- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)

## Next Steps

1. Verify Skypilot API documentation
2. Obtain API credentials
3. Store secrets in AWS Secrets Manager
4. Implement collector following template
5. Test locally with dry run
6. Deploy to staging
7. Set up Datadog monitors

## Maintenance

- **Owner**: DevOps Team / ML Infrastructure
- **Priority**: High (cost tracking critical)
- **API Version**: TBD (check with Skypilot team)
- **Estimated Effort**: 2-4 hours implementation
- **Last Updated**: 2025-10-22
