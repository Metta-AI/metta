# GitHub Metrics Missing Prefix

## Problem

**Status**: 🔧 **IN PROGRESS** - Need to add `github.` prefix (2025-10-24)

The GitHub collector metrics ARE in Datadog, but they're **missing the `github.` prefix**.

**Current**: `ci.*`, `prs.*`, `commits.*`, `branches.*`, `code.*`, `developers.*`
**Desired**: `github.ci.*`, `github.prs.*`, `github.commits.*`, etc.

**Decision**: Add `github.` prefix for consistency with all other collectors (wandb.*, asana.*, ec2.*, etc.)

## Evidence

### Discovery: Metrics Exist Without Prefix!

Using the new `list_datadog_metrics.py` helper script, discovered metrics ARE in Datadog:

```bash
$ uv run python devops/datadog/scripts/list_datadog_metrics.py --summary

Found 769 metrics total, including:
  ✅ ci.*: 10 metrics          # Should be github.ci.*
  ✅ prs.*: 4 metrics           # Should be github.prs.*
  ✅ commits.*: 5 metrics       # Should be github.commits.*
  ✅ branches.*: 1 metrics      # Should be github.branches.*
  ✅ code.*: 3 metrics          # Should be github.code.*
  ✅ developers.*: 1 metrics    # Should be github.developers.*

  ✅ wandb.*: 23 metrics        # Correct ✓
  ✅ asana.*: 12 metrics        # Correct ✓
  ✅ ec2.*: 20 metrics          # Correct ✓
```

**Actual metrics in Datadog:**
```
ci.tests_passing_on_main          # Expected: github.ci.tests_passing_on_main
ci.failed_workflows_7d            # Expected: github.ci.failed_workflows_7d
prs.open                          # Expected: github.prs.open
prs.stale_count_14d               # Expected: github.prs.stale_count_14d
commits.hotfix                    # Expected: github.commits.hotfix
... (24 more)
```

**Total**: 28 GitHub metrics exist, all missing the `github.` prefix!

### Dashboard Impact

The **System Health Rollup dashboard** (ID: `h3w-ibt-gkv`) shows "no data" because:
- It queries `health.ci.*` metrics
- Health FoM collector queries `github.ci.*` source metrics
- GitHub collector isn't emitting any metrics
- Therefore, no data propagates to the dashboard

**URL**: https://app.datadoghq.com/dashboard/h3w-ibt-gkv/system-health-rollup

## What's Working

### ✅ Other Collectors (Working)
- **Asana**: 14 metrics (e.g., `asana.projects.active`)
- **EC2**: 20 metrics (e.g., `ec2.instances.running`)
- **Skypilot**: 30 metrics
- **Kubernetes**: 15 metrics
- **WandB**: 23 metrics ✨ (including new push-to-main and sweep metrics)

**Total**: 169 metrics across 6/7 collectors

### ✅ WandB Metrics (Full List)
```
wandb.push_to_main.avg_duration_hours
wandb.push_to_main.overview.epoch_steps_per_second
wandb.push_to_main.overview.sps
wandb.push_to_main.overview.steps_per_second
wandb.push_to_main.runs_completed_24h
wandb.push_to_main.runs_failed_24h
wandb.push_to_main.success_rate_pct
wandb.push_to_main.timing_cumulative.sps
wandb.push_to_main.timing_per_epoch.sps
wandb.runs.active
wandb.runs.completed_24h
wandb.runs.completed_7d
wandb.runs.failed_24h
wandb.runs.failed_7d
wandb.runs.total
wandb.runs.total_recent
wandb.sweep.runs_active
wandb.sweep.runs_completed_24h
wandb.sweep.runs_failed_24h
wandb.sweep.runs_total_24h
wandb.sweep.success_rate_pct
wandb.training.avg_duration_hours
wandb.training.total_gpu_hours_7d
```

## Investigation Steps

### 1. Check CronJob Logs

```bash
# Check recent production jobs
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -10

# Check logs from most recent job
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=200
```

**Look for**:
- Does GitHub collector run?
- Any errors during collection?
- Does it successfully push metrics?

### 2. Test GitHub Collector Locally

```bash
# Run GitHub collector with verbose output
cd devops/datadog
uv run python scripts/run_collector.py github --verbose --push

# Check if metrics are emitted
uv run python scripts/run_collector.py github --verbose
```

**Expected**: Should see "Collected N metrics from github" and successful push

### 3. Check GitHub API Access

```bash
# Verify GitHub token works
uv run python -c "
from devops.datadog.utils.secrets import get_secret
token = get_secret('github/dashboard-token')
import requests
r = requests.get('https://api.github.com/user', headers={'Authorization': f'token {token}'})
print(f'Status: {r.status_code}')
print(f'User: {r.json().get(\"login\")}')
"
```

**Expected**: Status 200, user login shown

### 4. Verify Metrics Submission

```bash
# Use DatadogClient directly to test metric submission
uv run python -c "
from devops.datadog.utils.dashboard_client import get_datadog_credentials
from devops.datadog.utils.datadog_client import DatadogClient

api_key, app_key, site = get_datadog_credentials()
client = DatadogClient(api_key=api_key, app_key=app_key, site=site)

# Try pushing a test metric
metrics = {'github.test.metric': 1.0}
result = client.push_metrics(metrics)
print(f'Push result: {result}')
"
```

## Root Cause Hypotheses

1. **GitHub API Rate Limiting**: Token hitting rate limits
2. **Collector Disabled**: GitHub collector not running in CronJob
3. **Authentication Failure**: GitHub token invalid or expired
4. **Metric Submission Failure**: Datadog API rejecting metrics
5. **Collector Crash**: GitHub collector failing early, not pushing metrics

## Next Steps

1. Check CronJob logs to see if GitHub collector runs
2. Test GitHub collector locally to reproduce
3. Verify GitHub token has correct permissions
4. Check if metrics are being submitted to Datadog API
5. Fix identified issue and deploy

## Verification Commands

```bash
# Quick check: Do any github.* metrics exist?
uv run python -c "
from devops.datadog.utils.dashboard_client import get_datadog_credentials
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datetime import datetime, timedelta

config = Configuration()
api_key, app_key, site = get_datadog_credentials()
config.api_key['apiKeyAuth'] = api_key
config.api_key['appKeyAuth'] = app_key
config.server_variables['site'] = site

with ApiClient(config) as api_client:
    api = MetricsApi(api_client)
    from_time = int((datetime.now() - timedelta(hours=2)).timestamp())
    response = api.list_active_metrics(from_time)

    github_metrics = [m for m in response.metrics if m.startswith('github.')]
    print(f'GitHub metrics found: {len(github_metrics)}')
    for m in sorted(github_metrics)[:10]:
        print(f'  {m}')
"
```

## Implementation Plan

### 1. Investigate Metric Naming (WHERE is prefix stripped?)

**Compare collectors:**
- ✅ **WandB**: Returns `wandb.runs.completed_24h` → Datadog has `wandb.runs.completed_24h` ✓
- ❌ **GitHub**: Returns `ci.tests_passing_on_main` → Datadog has `ci.tests_passing_on_main` (missing `github.`)

**Questions to answer:**
- Does GitHub collector's `collect_metrics()` return metric names with or without `github.`?
- Does BaseCollector modify metric names during push?
- How does WandB collector ensure the `wandb.` prefix is preserved?

### 2. Fix GitHub Collector

**Option A**: Add prefix in metric definitions (if not present)
**Option B**: Add prefix in BaseCollector.push_metrics() (if being stripped)
**Option C**: Add prefix in collect_metrics() return dict

Need to check the code to determine which option is correct.

### 3. Update Health FoM Collector

Change queries from:
```python
self._query_metric("ci.tests_passing_on_main")
self._query_metric("ci.failed_workflows_7d")
```

To:
```python
self._query_metric("github.ci.tests_passing_on_main")
self._query_metric("github.ci.failed_workflows_7d")
```

### 4. Update Dashboards

Update any dashboard queries that reference:
- `ci.*` → `github.ci.*`
- `prs.*` → `github.prs.*`
- `commits.*` → `github.commits.*`
- etc.

### 5. Test Locally

```bash
# 1. Run GitHub collector
uv run python devops/datadog/scripts/run_collector.py github --push --verbose

# 2. Wait 60 seconds
sleep 60

# 3. Verify metrics with github. prefix
uv run python devops/datadog/scripts/list_datadog_metrics.py --prefix github --verbose
```

Expected: Should see 28 `github.*` metrics

### Helper Scripts Created

- `scripts/list_datadog_metrics.py` - List all metrics with filtering
- `scripts/test_metric_push.py` - Test metric push end-to-end
- `TESTING-metrics-push.md` - Systematic testing plan

## Related

- Health FoM collector depends on GitHub metrics with correct names
- System Health Rollup dashboard needs updated metric queries
- All other collectors working with correct prefixes (wandb.*, asana.*, ec2.*)
