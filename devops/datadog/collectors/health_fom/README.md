# Health FoM Collector

Calculates normalized health scores (0.0-1.0) from other collectors' metrics.

## Metrics

All metrics are normalized to 0.0-1.0 scale:
- 1.0 = Healthy
- 0.7-1.0 = Good
- 0.3-0.7 = Warning
- 0.0-0.3 = Critical

| Metric | Type | Source Metrics | Formula |
|--------|------|----------------|---------|
| `health.ci.tests_passing.fom` | gauge | `github.ci.tests{branch:main}` | Binary: 1.0 if passing, 0.0 if failing |
| `health.ci.failing_workflows.fom` | gauge | `github.ci.runs{status:failed,timeframe:7d}` | `max(1.0 - count/5, 0.0)` |
| `health.ci.hotfix_count.fom` | gauge | `github.commits.special{type:hotfix,timeframe:7d}` | `max(1.0 - count/10, 0.0)` |
| `health.ci.revert_count.fom` | gauge | `github.commits.special{type:revert,timeframe:7d}` | `max(1.0 - count/2, 0.0)` |
| `health.ci.duration_p90.fom` | gauge | `github.ci.duration_minutes{metric:p90}` | `clamp(1.0 - (duration-3)/(10-3))` |
| `health.ci.stale_prs.fom` | gauge | `github.prs.stale{threshold:14d}` | `max(1.0 - count/50, 0.0)` |
| `health.ci.pr_cycle_time.fom` | gauge | `avg(github.pr.time_to_merge_hours{*})` | `clamp(1.0 - (hours-24)/(72-24))` |
| `health.training.run_success.fom` | gauge | `wandb.runs{run_type:train,state:finished,timeframe:7d}` | `min(count/7, 1.0)` |
| `health.training.run_failures.fom` | gauge | `wandb.runs{run_type:train,state:failed,timeframe:7d}` | `max(1.0 - count/3, 0.0)` |

**Total**: 9 FoM metrics (7 CI, 2 Training)

## Configuration

```bash
# Secrets (AWS Secrets Manager)
datadog/api-key    # For querying metrics
datadog/app-key    # For querying metrics
```

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py health_fom --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py health_fom --push
```

## Dashboard Queries

```python
# Overall CI health (average of all CI FoMs)
avg:health.ci.*.fom{}

# Training health
avg:health.training.*.fom{}

# Critical issues (FoM < 0.3)
health.*.fom{*} < 0.3
```

## Notes

- Runs last in collector sequence (depends on other collectors' metrics)
- Queries Datadog API with 4-hour lookback window
- All formulas are defined in collector code
