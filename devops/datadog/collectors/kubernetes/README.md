# Kubernetes Collector

Monitors cluster resource efficiency and pod health.

## Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `k8s.resources.waste` | gauge | `resource:cpu\|memory`, `unit:cores\|gb` | Unused requested resources |
| `k8s.resources.efficiency` | gauge | `resource:cpu\|memory` | Percentage of requested resources used |
| `k8s.pods` | gauge | `status:overallocated`, `issue:crash_looping\|high_restarts\|oomkilled\|image_pull_error`, `phase:failed\|pending`, `utilization:idle\|low_cpu\|low_memory`, `timeframe:24h` | Pod counts by status/issue |
| `k8s.deployments` | gauge | `status:zero_replicas` | Deployments scaled to 0 |

**Total**: 4 metric names with dimensional tags

## Configuration

```bash
# Environment Variables
K8S_CLUSTER_NAME=main   # Cluster identifier
```

Requires cluster permissions:
- Read pods, deployments, pod metrics (metrics.k8s.io)
- Deployed with service account and RBAC

## Usage

```bash
# Test locally (uses ~/.kube/config)
uv run python devops/datadog/scripts/run_collector.py kubernetes --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py kubernetes --push
```

## Dashboard Queries

```python
# Resource waste
sum:k8s.resources.waste{resource:cpu,unit:cores}

# Efficiency
avg:k8s.resources.efficiency{resource:memory}

# Pod health
sum:k8s.pods{issue:crash_looping}

# Underutilization
sum:k8s.pods{utilization:idle}
```
