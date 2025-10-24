# WandB Collector

Collects training run metrics from Weights & Biases (wandb.ai).

Optimized to focus on recent activity (last 24 hours) with server-side filtering to avoid fetching 26k+ historical runs.

## Metrics Collected

### Overall Run Metrics (4 metrics)
- `wandb.runs.active` - Currently running experiments
- `wandb.runs.completed_24h` - Completed runs in last 24 hours
- `wandb.runs.failed_24h` - Failed/crashed runs in last 24 hours
- `wandb.runs.total_recent` - Total recent activity (24h + active)

### Resource Usage Metrics (2 metrics)
- `wandb.training.avg_duration_hours` - Average training duration across all runs
- `wandb.training.total_gpu_hours_24h` - Total GPU hours consumed in last 24 hours

### Push-to-Main CI Metrics (9 metrics)
Tracks baseline performance of GitHub CI runs (pattern: `github.sky.*`):

**Run Statistics**:
- `wandb.push_to_main.runs_completed_24h` - Successfully completed CI runs
- `wandb.push_to_main.runs_failed_24h` - Failed/crashed CI runs
- `wandb.push_to_main.success_rate_pct` - Percentage of successful CI runs
- `wandb.push_to_main.avg_duration_hours` - Average CI run duration

**Training Throughput (Steps Per Second)**:
- `wandb.push_to_main.overview.steps_per_second` - Primary SPS metric
- `wandb.push_to_main.overview.epoch_steps_per_second` - Per-epoch SPS
- `wandb.push_to_main.overview.sps` - Alternative SPS metric
- `wandb.push_to_main.timing_cumulative.sps` - Cumulative timing SPS
- `wandb.push_to_main.timing_per_epoch.sps` - Per-epoch timing SPS

### Sweep/Hyperparameter Tuning Metrics (5 metrics)
Tracks experimental runs with `.sweep` in name or WandB sweep attribute:

- `wandb.sweep.runs_total_24h` - Total sweep runs in last 24 hours
- `wandb.sweep.runs_completed_24h` - Successfully completed sweep runs
- `wandb.sweep.runs_failed_24h` - Failed sweep runs
- `wandb.sweep.runs_active` - Currently running sweep experiments
- `wandb.sweep.success_rate_pct` - Sweep run success rate

**Total**: 20 metrics

## Configuration

### Environment Variables

Required:
- `WANDB_API_KEY` - WandB API key (or stored in AWS Secrets Manager)
- `WANDB_ENTITY` - WandB entity/username (default: `metta-research`)
- `WANDB_PROJECT` - WandB project name (default: `metta`)

### AWS Secrets Manager

Store credentials in AWS Secrets Manager (recommended for production):

```bash
# Create secret with all WandB configuration
aws secretsmanager create-secret \
  --name dev/datadog/collectors/wandb \
  --secret-string '{
    "WANDB_API_KEY": "your_api_key_here",
    "WANDB_ENTITY": "metta-research",
    "WANDB_PROJECT": "metta"
  }'
```

### WandB API Key

Get your API key from https://wandb.ai/settings

## Usage

### Local Testing

```bash
# Test collection (dry run - no push to Datadog)
uv run python devops/datadog/scripts/run_collector.py wandb --verbose

# Test with push to Datadog
uv run python devops/datadog/scripts/run_collector.py wandb --push --verbose
```

### Kubernetes Deployment

Deployed as part of the unified CronJob that runs all collectors every 15 minutes:

```bash
# Deploy dev environment
helm upgrade dashboard-cronjob-dev devops/charts/dashboard-cronjob \
  --namespace monitoring \
  --set nameOverride=dev \
  --set image.tag=sha-<commit>

# Deploy production
helm upgrade dashboard-cronjob devops/charts/dashboard-cronjob \
  --namespace monitoring
```

## Metrics Details

### Run Status Tracking (24h Window)

Tracks recent training pipeline health:
- Active runs indicate ongoing training
- Completed runs show training velocity
- Failed runs highlight stability issues
- Total recent combines all activity for quick health check

### Push-to-Main CI Tracking

Monitors baseline performance of automated CI runs following GitHub merge pattern `github.sky.*`:

**Why Important**: These runs represent the ground truth performance of the main branch over time.

**Run Naming Pattern**:
- `github.sky.main.<commit>.<config>.<timestamp>`
- `github.sky.pr<number>.<commit>.<config>.<timestamp>`

**Training Throughput**: Five different SPS (steps per second) metrics provide comprehensive view of training performance:
- `overview/steps_per_second` - Primary training throughput
- `overview/epoch_steps_per_second` - Per-epoch throughput
- `overview/sps` - Alternative measurement
- `timing_cumulative/sps` - Cumulative timing metric
- `timing_per_epoch/sps` - Per-epoch timing metric

**Crashed Run Handling**: The collector properly handles crashed runs by parsing JSON strings from WandB API (WandB quirk where `_json_dict` returns JSON string for crashed runs instead of dict).

### Sweep/Hyperparameter Tuning Tracking

Monitors experimental runs used for hyperparameter optimization:

**Detection**: Runs with `.sweep` in the name or WandB sweep attribute set

**Typical Pattern**: `ak.sweep_multi_gpu_params.v5_trial_0009_563d62`

**Use Case**: Track success rate and volume of hyperparameter exploration separate from production CI runs.

### Resource Usage (24h Window)

Tracks recent compute efficiency:
- Training duration helps estimate costs
- Total GPU hours enables cost tracking
- Assumes 1 GPU per run (configurable)

## Dependencies

- `wandb` - Official WandB Python client

## Implementation Notes

**Performance Optimizations**:
- Uses WandB API server-side filtering to avoid fetching 26k+ historical runs
- Single API call fetches all runs once, then categorizes efficiently
- Focuses on 24-hour window for actionable monitoring data
- Timeout protection (120s) prevents indefinite hangs

**Data Collection**:
- Runs categorized by type: push-to-main, sweep, regular
- Runs filtered by state (running, finished, failed, crashed)
- Uses `created_at` filters with ISO timestamps for time-based queries
- GPU hours assumes 1 GPU per run

**Crashed Run Support**:
- Handles WandB API quirk where crashed runs have `_json_dict` as JSON string
- Automatically parses JSON string to extract summary metrics
- Enables tracking performance metrics from failed CI runs

**Metric Naming**:
- Push-to-main metrics use dotted notation: `wandb.push_to_main.overview.steps_per_second`
- Sweep metrics follow same pattern: `wandb.sweep.runs_total_24h`
- Datadog automatically creates metric hierarchies from dot notation

## Troubleshooting

### No metrics collected

- Verify `WANDB_API_KEY` is valid
- Check entity/project names are correct: `metta-research/metta`
- Ensure project has runs logged to it
- Check AWS Secrets Manager permissions (IRSA for Kubernetes)

### Missing push-to-main metrics

- Ensure CI runs follow naming pattern `github.sky.*`
- Check that runs log SPS metrics to WandB summary
- Verify runs reach "finished" or "crashed" state (running runs don't have final metrics)
- All 5 SPS metrics should be present in run summary

### Missing sweep metrics

- Ensure sweep runs have `.sweep` in the name or WandB sweep attribute
- Check that sweep runs are completing successfully
- Verify runs are within the 24-hour window

### Crashed run metrics not appearing

The collector properly handles crashed runs (common for CI). If metrics still missing:
- Check that run summary exists (run must progress beyond initialization)
- Verify `_json_dict` field contains data
- Look for JSON parsing errors in logs

### Authentication errors

- Regenerate API key if expired
- Verify AWS Secrets Manager permissions
- Check IAM role has `secretsmanager:GetSecretValue`
- For IRSA issues, verify service account annotation

## Related Documentation

- [WandB Python API](https://docs.wandb.ai/ref/python/public-api/)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Metric Conventions](../../docs/METRIC_CONVENTIONS.md)
