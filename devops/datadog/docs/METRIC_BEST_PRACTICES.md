# Metric Collection Best Practices

Guide for designing effective metric collection strategies for Datadog monitoring.

## Core Principles

### 1. Prefer Instantaneous Values Over Pre-Aggregated Metrics

**Do This**:
```python
# Emit individual run metrics with tags
for run in runs:
    metrics[f"wandb.ptm.success.duration_hours"].append(
        (duration, ["run_id:abc123", "state:success"])
    )
```

**Not This**:
```python
# Pre-calculate averages in collector
metrics["wandb.ptm.avg_duration_hours"] = sum(durations) / len(durations)
```

**Why**:
- **Flexibility**: Dashboard can aggregate over any time window (1h, 24h, 7d) without redeploying collector
- **Granularity**: Can compute avg(), p50(), p95(), max(), min() in Datadog
- **Filtering**: Can slice by tags (state, type, etc.) in queries
- **Historical Analysis**: Can re-aggregate historical data with different functions

### 2. Use Tags for Dimensions, Not Separate Metrics

**Do This**:
```python
# Single metric with state tag
metrics["wandb.ptm.duration_hours"].append(
    (2.5, ["state:success", "run_type:ptm"])
)
metrics["wandb.ptm.duration_hours"].append(
    (0.8, ["state:failure", "run_type:ptm"])
)
```

**Not This**:
```python
# Separate metrics for each state
metrics["wandb.ptm.success.duration_hours"] = 2.5
metrics["wandb.ptm.failure.duration_hours"] = 0.8
```

**Why**:
- **Consistent Queries**: Same metric name, filter by tags
- **Easier Comparison**: Can graph both on same chart
- **Lower Metric Count**: Datadog bills by unique metric names
- **Better Organization**: Related data grouped together

**Exception**: When the metric has fundamentally different meaning (e.g., `requests.count` vs `requests.latency_ms`), use separate metric names.

### 3. Choose Appropriate Cardinality for Tags

**Good Cardinality** (bounded, predictable):
- `state:success|failure|active` (3 values)
- `run_type:ptm|sweep|all` (3 values)
- `region:us-east-1|us-west-2` (few values)
- `environment:prod|dev|staging` (few values)

**Bad Cardinality** (unbounded, explosive):
- `run_id:abc123...` **if querying across all runs** (thousands of unique IDs)
- `user_id:12345...` (potentially millions)
- `timestamp:2025-01-27T...` (infinite)

**When to Use High Cardinality Tags**:
- For deduplication (like `run_id` in our WandB collector)
- When you need to track individual entities
- As long as you're not creating a unique metric per tag value

**Datadog Limits**:
- 1000 unique tag value combinations per metric (soft limit)
- Beyond this, metrics may be dropped or aggregated

### 4. Let Datadog Handle Time Windows

**Do This**:
```python
# Emit metrics with current timestamp, let Datadog aggregate over time
for run in recent_runs:
    emit_metric("duration_hours", run.duration, tags)
```

**Not This**:
```python
# Hard-code time windows in collector
last_24h_runs = [r for r in runs if r.created_at > 24h_ago]
metrics["runs_24h"] = len(last_24h_runs)
```

**Why**:
- Dashboard can query any time range without code changes
- Can create rolling windows (last 1h, 3h, 7d) in dashboard
- Historical data can be re-queried with different windows
- No need to redeploy when you want different time ranges

**Exception**: When the source data is already aggregated by time window (e.g., daily reports from external systems).

### 5. Choose Meaningful Metric Names

**Good Naming**:
```
wandb.ptm.success.duration_hours
wandb.ptm.success.sps
github.workflow.run.duration_s
ec2.instance.cost_usd
```

**Bad Naming**:
```
metric1
training_data
val
xyz
```

**Naming Convention**:
```
<source>.<category>.<subcategory>.<metric>[_<unit>]

wandb.ptm.success.duration_hours
└─┬─┘ └┬┘ └──┬──┘ └────┬────┘└─┬─┘
  │    │     │         │       └── Unit (optional but recommended)
  │    │     │         └────────── Metric name (what you're measuring)
  │    │     └──────────────────── Subcategory (optional)
  │    └────────────────────────── Category (logical grouping)
  └─────────────────────────────── Source (system/service)
```

**Units**:
- Time: `_s`, `_ms`, `_hours`
- Size: `_bytes`, `_mb`, `_gb`
- Money: `_usd`, `_cents`
- Rate: `_per_second`, `_per_minute`
- Percentage: `_pct`, `_ratio`

### 6. Emit Metrics at the Right Frequency

**High Frequency** (every collection cycle):
- Resource usage (CPU, memory, active runs)
- System health (error rates, latencies)
- Counts (requests, jobs)

**Low Frequency** (when events occur):
- Individual run completions (with run_id tag for deduplication)
- Deployments
- Incidents

**Our Collection Frequency**: 15 minutes (good for most use cases)

**Datadog Deduplication**: Uses combination of metric name + tags + timestamp to deduplicate. Submit the same run multiple times with same `run_id` tag, and Datadog will only store it once per timestamp window.

## Metric Types

### Gauges (Most Common)

Values that can go up or down, representing a point-in-time measurement.

**Examples**:
- `wandb.*.active.duration_hours` (current active runs - use count() to get total)
- `ec2.instance.cpu_utilization_pct` (current CPU usage)
- `kubernetes.pod.memory_mb` (current memory)

**When to Use**: Almost always, especially for per-run metrics

### Counts

Incrementing counters that only go up.

**Examples**:
- `github.workflow.runs.total` (cumulative count)
- `requests.total` (all-time request count)

**When to Use**: Rarely needed - usually better to emit gauges and use Datadog's `count()` function in queries

### Rates

Events per unit time.

**Examples**:
- `requests.per_second`
- `github.commits.per_hour`

**When to Use**: Also rarely needed - emit counts and use Datadog's `rate()` function

## Common Patterns

### Pattern 1: Per-Event Metrics with Deduplication

**Use Case**: Track individual events (training runs, deployments, incidents)

**Implementation**:
```python
for run in runs:
    metrics["wandb.ptm.duration_hours"].append(
        (run.duration, [
            f"run_id:{run.id}",  # Deduplication key
            f"state:{run.state}",
            f"run_type:ptm"
        ])
    )
```

**Dashboard Query**:
```
avg:wandb.ptm.duration_hours{state:success}  # Average duration of successful runs
p95:wandb.ptm.duration_hours{state:success}  # 95th percentile
count:wandb.ptm.duration_hours{state:failure} # Number of failures
```

### Pattern 2: Count Metrics via Aggregation

**Use Case**: Count events without creating separate count metrics

**Implementation**:
```python
# Don't emit counts - emit per-event metrics instead
for run in runs:
    if run.state == "running":
        metrics["wandb.ptm.active.duration_hours"].append(
            (run.elapsed_time, [f"run_id:{run.id}", "state:active"])
        )
```

**Dashboard Query**:
```
count:wandb.*.active.duration_hours{*}  # Total active runs (Datadog counts them)
count:wandb.ptm.success.duration_hours{*}  # Successful PTM runs
count:wandb.ptm.*.duration_hours{*}  # All PTM runs (success + failure + active)
```

**Why Better Than Separate Count Metrics**:
- No duplicate metrics (don't need both `duration` and `count`)
- Works over any time window
- Can filter by tags in count query
- One less metric to maintain

### Pattern 3: Resource Attribution

**Use Case**: Track resource usage per dimension

**Implementation**:
```python
for instance in instances:
    metrics["ec2.instance.cost_usd"].append(
        (instance.cost, [
            f"instance_type:{instance.type}",
            f"environment:{instance.env}",
            f"team:{instance.owner}"
        ])
    )
```

**Dashboard Query**:
```
sum:ec2.instance.cost_usd{*} by {team}  # Cost by team
sum:ec2.instance.cost_usd{environment:prod}  # Production costs
```

## Performance Considerations

### Metric Volume

**Datadog Limits**:
- Custom metrics billed by unique metric name + tag combination
- Each unique combination counts as separate metric
- Example: `wandb.ptm.duration_hours{state:success}` and `wandb.ptm.duration_hours{state:failure}` = 2 billable metrics

**Optimization**:
- Use tags wisely (limit unique combinations)
- Avoid high-cardinality tags unless necessary
- Batch metric submissions (we submit all collectors together every 15 minutes)

### Collection Performance

**Efficient Collection**:
```python
# Fetch once, categorize in memory
runs = api.fetch_all_runs(filters={"created_at": {"$gte": one_day_ago}})
for run in runs:
    # Categorize and emit
```

**Inefficient Collection**:
```python
# Multiple API calls
success_runs = api.fetch_runs(state="finished")  # API call 1
failed_runs = api.fetch_runs(state="failed")    # API call 2
sweep_runs = api.fetch_runs(tags=["sweep"])     # API call 3
```

### Timeout Protection

All collectors have 120s timeout to prevent indefinite hangs:
```python
@timeout_decorator.timeout(120)
def collect_metrics(self):
    # Collection logic
```

## Anti-Patterns to Avoid

### ❌ Pre-Aggregating Time Windows

**Don't**:
```python
metrics["runs_last_1h"] = count_runs(last_1h)
metrics["runs_last_24h"] = count_runs(last_24h)
metrics["runs_last_7d"] = count_runs(last_7d)
```

**Do**:
```python
for run in runs:
    metrics["runs.count"].append((1, [f"run_id:{run.id}"]))
# Dashboard queries with different time windows
```

### ❌ Baking State into Metric Names

**Don't**:
```python
metrics["runs.success.count"] = 10
metrics["runs.failure.count"] = 2
```

**Do**:
```python
metrics["runs.count"].append((10, ["state:success"]))
metrics["runs.count"].append((2, ["state:failure"]))
```

### ❌ Metric Name Explosions

**Don't**:
```python
for user_id in users:
    metrics[f"user_{user_id}_login_count"] = count
```

**Do**:
```python
for user_id, count in user_logins.items():
    metrics["user.login.count"].append((count, [f"user_id:{user_id}"]))
```

### ❌ Sampling Without Reason

**Don't**:
```python
if random.random() < 0.1:  # Sample 10%
    emit_metric(...)
```

**Do**:
```python
# Emit all data points, let Datadog aggregate
for datapoint in all_datapoints:
    emit_metric(...)
```

**Exception**: When data volume is truly massive (millions of events per minute), consider sampling or using Datadog's distribution metrics.

### ❌ Missing Units in Names

**Don't**:
```python
metrics["duration"] = 3.5  # What unit? Seconds? Hours?
metrics["cost"] = 42.5     # USD? Cents?
```

**Do**:
```python
metrics["duration_hours"] = 3.5
metrics["cost_usd"] = 42.5
```

## Migration Guide: Aggregated → Instantaneous

If you have existing collectors with pre-aggregated metrics, here's how to migrate:

### Before (Aggregated)
```python
def collect_metrics(self):
    runs = fetch_runs()

    completed = [r for r in runs if r.state == "finished"]
    failed = [r for r in runs if r.state == "failed"]

    return {
        "runs.completed_24h": len(completed),
        "runs.failed_24h": len(failed),
        "runs.avg_duration_hours": sum(r.duration for r in runs) / len(runs),
        "runs.success_rate_pct": len(completed) / len(runs) * 100
    }
```

### After (Instantaneous)
```python
def collect_metrics(self):
    runs = fetch_runs()
    metrics = {}

    for run in runs:
        state = "success" if run.state == "finished" else "failure"
        tags = [f"run_id:{run.id}", f"state:{state}"]

        # Emit per-run duration
        if "runs.duration_hours" not in metrics:
            metrics["runs.duration_hours"] = []
        metrics["runs.duration_hours"].append((run.duration, tags))

    return metrics
```

### Dashboard Query Migration

**Before**:
```
runs.completed_24h  # Pre-calculated count over 24h
runs.avg_duration_hours  # Pre-calculated average
```

**After**:
```
count:runs.duration_hours{state:success}  # Count successful runs
avg:runs.duration_hours{*}  # Average duration
```

## Testing Your Metrics

### Local Testing

```bash
# Test without pushing to Datadog
uv run python devops/datadog/scripts/run_collector.py <collector_name>

# Verify metric structure
uv run python devops/datadog/scripts/run_collector.py <collector_name> --json

# Push to Datadog dev environment
uv run python devops/datadog/scripts/run_collector.py <collector_name> --push
```

### Verify in Datadog

1. **Metric Explorer**: Search for your metric name
2. **Check Tags**: Ensure tags appear correctly
3. **Test Aggregations**: Try avg(), p95(), count() queries
4. **Verify Deduplication**: Submit same run_id multiple times, ensure single data point

## Summary Checklist

When designing a new metric:

- [ ] Use instantaneous values, not pre-aggregated
- [ ] Let Datadog handle time windows
- [ ] Use tags for dimensions (state, type, etc.)
- [ ] Choose bounded tag cardinality (<100 unique values per tag)
- [ ] Include units in metric name (`_hours`, `_usd`, `_pct`)
- [ ] Use meaningful hierarchical names (`source.category.metric_unit`)
- [ ] Emit gauges for most use cases
- [ ] Include deduplication tag (like `run_id`) when tracking events
- [ ] Test locally before deploying
- [ ] Document your metrics in collector README

## Related Documentation

- [WandB Collector README](../collectors/wandb/README.md) - Example of instantaneous metric collection
- [Collectors Architecture](COLLECTORS_ARCHITECTURE.md) - Overall system design
- [Adding New Collector](ADDING_NEW_COLLECTOR.md) - Implementation guide
- [Datadog Metrics Guide](https://docs.datadoghq.com/metrics/) - Official Datadog documentation
