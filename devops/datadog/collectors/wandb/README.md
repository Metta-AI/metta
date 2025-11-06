# WandB Collector

Collects per-run training metrics from Weights & Biases.

## Metrics

All metrics use per-run values with tags for flexible aggregation.

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `wandb.runs` | gauge | `run_type:train\|sweep\|all`, `state:finished\|failed\|running`, `timeframe:7d` | Run counts |
| `wandb.run.duration_hours` | gauge | `run_id:{id}`, `run_type:{type}`, `state:{state}` | Per-run duration |
| `wandb.run.sps` | gauge | `run_id:{id}`, `run_type:{type}`, `state:{state}` | Per-run steps/second |
| `wandb.run.metric` | gauge | `run_id:{id}`, `metric_name:{name}` | Per-run metric values |
| `wandb.run.system` | gauge | `run_id:{id}`, `metric:gpu_utilization` | Per-run system metrics |

**Total**: 5 metric names with per-run tags

**Run Types**: Detected from run name patterns:
- `train`: GitHub CI runs (`github.sky.*`)
- `sweep`: Hyperparameter optimization (contains "sweep")
- `all`: Everything else

## Configuration

```bash
# Secrets (AWS Secrets Manager)
wandb/api-key          # WandB API key

# Environment Variables
WANDB_ENTITY=metta-research   # WandB entity
WANDB_PROJECT=metta           # WandB project
```

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py wandb --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py wandb --push
```

## Dashboard Queries

```python
# Average SPS for successful train runs
avg:wandb.run.sps{run_type:train,state:finished}

# Run count by type
count:wandb.runs{run_type:train,state:finished,timeframe:7d}

# Success rate
count:wandb.runs{state:finished} / (count:wandb.runs{state:finished} + count:wandb.runs{state:failed})

# P95 duration
p95:wandb.run.duration_hours{run_type:train}
```

## Architecture

Emits per-run instantaneous values instead of pre-aggregated metrics. This allows:
- Flexible time windows in dashboards (1h, 24h, 7d, custom)
- Any aggregation function (avg, p50, p95, count)
- Filtering by run type, state, or individual runs
- Historical data re-querying with new aggregations
