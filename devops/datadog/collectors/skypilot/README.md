# Skypilot Collector

Monitors job orchestration and GPU resource utilization.

## Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `skypilot.clusters` | gauge | `status:active` | Active cluster count |
| `skypilot.jobs` | gauge | `status:queued\|running\|failed\|succeeded\|cancelled`, `runtime_bucket:0_1h\|1_4h\|4_24h\|over_24h`, `pricing:spot\|ondemand`, `region:us_east_1\|us_west_2\|other`, `has_recoveries:true`, `timeframe:7d` | Job counts |
| `skypilot.jobs.runtime_seconds` | gauge | `metric:min\|max\|avg\|p50\|p90\|p99` | Runtime percentiles |
| `skypilot.resources.gpus` | gauge | `type:l4\|a10g\|h100\|total` | GPU counts by type |
| `skypilot.jobs.recovery_count` | gauge | `metric:avg\|max` | Recovery statistics |
| `skypilot.users` | gauge | `status:active` | Active user count |
| `skypilot.job.runtime_hours` | gauge | `job_id:{id}`, `user:{name}`, `region:{region}`, `gpu_type:{type}`, `instance_type:spot\|on_demand`, `status:{status}` | Per-job runtime |
| `skypilot.job.estimated_cost_hourly` | gauge | `job_id:{id}`, `gpu_type:{type}`, ... | Per-job hourly cost |
| `skypilot.job.gpu_count` | gauge | `job_id:{id}`, `gpu_type:{type}`, ... | Per-job GPU count |
| `skypilot.job.queue_wait_seconds` | gauge | `job_id:{id}`, ... | Queue wait time (queued jobs) |
| `skypilot.job.setup_seconds` | gauge | `job_id:{id}`, ... | Provisioning time |
| `skypilot.job.total_duration_hours` | gauge | `job_id:{id}`, ... | Total duration (completed jobs) |
| `skypilot.job.total_cost_usd` | gauge | `job_id:{id}`, `gpu_type:{type}`, ... | Total job cost |
| `skypilot.job.time_to_failure_hours` | gauge | `job_id:{id}`, ... | Runtime before failure |

**Total**: 14 metric names (8 aggregate + 6 per-job) with dimensional tags

## Configuration

No secrets required - uses Skypilot CLI access.

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py skypilot --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py skypilot --push
```

## Dashboard Queries

```python
# Running jobs
sum:skypilot.jobs{status:running}

# GPU utilization by type
sum:skypilot.resources.gpus{type:h100}

# Job success rate
sum:skypilot.jobs{status:succeeded} / (sum:skypilot.jobs{status:succeeded} + sum:skypilot.jobs{status:failed})

# P90 runtime
p90:skypilot.jobs.runtime_seconds{}

# Cost tracking
sum:skypilot.job.total_cost_usd{user:username}
```
