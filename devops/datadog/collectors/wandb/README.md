# WandB Collector

Collects training run metrics from Weights & Biases (wandb.ai).

Optimized to focus on recent activity (last 24 hours) with server-side filtering to avoid fetching 26k+ historical runs.

## Metrics Collected

**Run Status Metrics (4 metrics)**:
- `wandb.runs.active` - Currently running experiments
- `wandb.runs.completed_24h` - Completed runs in last 24 hours
- `wandb.runs.failed_24h` - Failed/crashed runs in last 24 hours
- `wandb.runs.total_recent` - Total recent activity (24h + active)

**Performance Metrics (3 metrics)** - From GitHub CI runs:
- `wandb.metrics.latest_sps` - Latest training throughput (steps per second)
- `wandb.metrics.avg_heart_amount_24h` - Average agent survival metric (heart amount)
- `wandb.metrics.latest_queue_latency_s` - Latest SkyPilot queue latency

**Resource Usage Metrics (3 metrics)**:
- `wandb.training.avg_duration_hours` - Average training duration
- `wandb.training.gpu_utilization_avg` - Average GPU utilization
- `wandb.training.total_gpu_hours_24h` - Total GPU hours in last 24 hours

**Total**: 10 metrics

## Configuration

### Environment Variables

Required:
- `WANDB_API_KEY` - WandB API key (or stored in AWS Secrets Manager as `wandb/api-key`)
- `WANDB_ENTITY` - WandB entity/username (default: `pufferai`)
- `WANDB_PROJECT` - WandB project name (default: `metta`)

### AWS Secrets Manager

Store the WandB API key in AWS Secrets Manager:

```bash
# Store API key
aws secretsmanager create-secret \
  --name wandb/api-key \
  --secret-string "your_wandb_api_key"
```

### WandB API Key

Get your API key from https://wandb.ai/settings

## Usage

### Local Testing

```bash
# Test collection (dry run)
uv run python devops/datadog/scripts/run_collector.py wandb --verbose

# Test with push to Datadog
uv run python devops/datadog/scripts/run_collector.py wandb --push --verbose
```

### Kubernetes Deployment

Deploy as a CronJob using Helm:

```bash
helm upgrade --install wandb-collector devops/charts/dashboard-cronjob \
  --namespace monitoring \
  --set command='["uv","run","python","/workspace/metta/devops/datadog/scripts/run_collector.py","wandb","--push"]' \
  --set schedule="*/30 * * * *" \  # Every 30 minutes
  --set datadog.service="wandb-collector"
```

## Metrics Details

### Run Status Tracking (24h Window)

Tracks recent training pipeline health:
- Active runs indicate ongoing training
- Completed runs in last 24h show training velocity
- Failed runs highlight recent stability issues
- Total recent combines all activity for quick health check

### Performance Metrics (GitHub CI Runs)

Monitors training performance from automated CI runs (pattern: `github.sky.main.*`):
- Training throughput (SPS) tracks steps per second
- Agent survival (heart amount) measures agent performance
- Queue latency tracks SkyPilot infrastructure responsiveness

### Resource Usage (24h Window)

Tracks recent compute efficiency:
- Training duration helps estimate costs
- GPU utilization shows resource efficiency
- Total GPU hours enables short-term cost tracking

## Dependencies

- `wandb` - Official WandB Python client

## Implementation Notes

**Performance Optimizations:**
- Uses WandB API server-side filtering to avoid fetching 26k+ historical runs
- Focuses on 24-hour window for actionable monitoring data
- Timeout protection (120s) prevents indefinite hangs

**Data Collection:**
- Runs are filtered by state (running, finished, failed, crashed)
- Uses `created_at` filters with ISO timestamps for time-based queries
- GPU hours assumes 1 GPU per run (configurable in production)
- Performance metrics extracted from run summary (GitHub CI runs)
- Handles missing metrics gracefully (returns None)

**Metrics Focus:**
- Prioritizes GitHub CI runs (pattern: `github.sky.main.*`)
- Tracks Metta-specific metrics (SPS, heart amount, queue latency)
- Short time window provides recent, actionable insights

## Troubleshooting

### No metrics collected

- Verify `WANDB_API_KEY` is valid
- Check entity/project names are correct
- Ensure project has runs logged to it

### Missing performance metrics

- Ensure training code logs Metta metrics to WandB:
  - `overview/sps` (training throughput)
  - `env_agent/heart.amount` (agent survival)
  - `skypilot/queue_latency_s` (infrastructure latency)
- Check run summaries contain expected metrics
- Verify metric naming matches (case-sensitive)
- Confirm GitHub CI runs follow pattern `github.sky.main.*`

### Authentication errors

- Regenerate API key if expired
- Verify AWS Secrets Manager permissions
- Check IAM role has `secretsmanager:GetSecretValue`

## Related Documentation

- [WandB Python API](https://docs.wandb.ai/ref/python/public-api/)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
