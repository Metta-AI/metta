# WandB Collector

Collects training run metrics from Weights & Biases (wandb.ai).

## Metrics Collected

**Run Status Metrics (4 metrics)**:
- `wandb.runs.active` - Currently running experiments
- `wandb.runs.completed_7d` - Completed runs in last 7 days
- `wandb.runs.failed_7d` - Failed/crashed runs in last 7 days
- `wandb.runs.total` - Total runs in project

**Performance Metrics (3 metrics)**:
- `wandb.metrics.best_accuracy` - Best accuracy across all runs
- `wandb.metrics.latest_loss` - Most recent loss value
- `wandb.metrics.avg_accuracy_7d` - Average accuracy from recent runs

**Resource Usage Metrics (3 metrics)**:
- `wandb.training.avg_duration_hours` - Average training duration
- `wandb.training.gpu_utilization_avg` - Average GPU utilization
- `wandb.training.total_gpu_hours_7d` - Total GPU hours in last 7 days

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

### Run Status Tracking

Tracks the health of your training pipeline:
- Active runs indicate ongoing training
- Completed runs show successful training velocity
- Failed runs highlight training stability issues

### Performance Metrics

Monitors model quality:
- Best accuracy tracks your best model performance
- Latest loss shows recent training progress
- Average accuracy indicates consistent performance

### Resource Usage

Tracks compute efficiency:
- Training duration helps estimate costs
- GPU utilization shows resource efficiency
- Total GPU hours enables cost tracking

## Dependencies

- `wandb` - Official WandB Python client

## Implementation Notes

- Uses WandB Public API for querying runs
- Runs are filtered by state (running, finished, failed, crashed)
- Time-based metrics use last 7 days window
- GPU hours assumes 1 GPU per run (configurable in production)
- Performance metrics extracted from run summary
- Handles missing metrics gracefully (returns None)

## Troubleshooting

### No metrics collected

- Verify `WANDB_API_KEY` is valid
- Check entity/project names are correct
- Ensure project has runs logged to it

### Missing performance metrics

- Ensure training code logs `accuracy` and `loss` to WandB
- Check run summaries contain expected metrics
- Verify metric naming matches (case-sensitive)

### Authentication errors

- Regenerate API key if expired
- Verify AWS Secrets Manager permissions
- Check IAM role has `secretsmanager:GetSecretValue`

## Related Documentation

- [WandB Python API](https://docs.wandb.ai/ref/python/public-api/)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
