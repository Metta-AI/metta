# CI/CD Metrics Documentation

## Overview

Comprehensive metrics tracking development velocity, code quality, and CI/CD efficiency for the Metta project.

All metrics are collected every 15 minutes via Kubernetes CronJob and submitted to Datadog as GAUGE values.

## Metric Categories

### Pull Request Metrics

Track PR velocity and merge patterns to understand code review efficiency.

| Metric Key | Description | Unit | Type |
|------------|-------------|------|------|
| `prs.open` | Currently open pull requests | count | GAUGE |
| `prs.merged_7d` | PRs merged in last 7 days | count | GAUGE |
| `prs.closed_without_merge_7d` | PRs closed without merge in last 7 days | count | GAUGE |
| `prs.avg_time_to_merge_hours` | Average time from PR creation to merge | hours | GAUGE |

**Use Cases**:
- Track PR backlog growth (`prs.open` trending up)
- Measure team velocity (`prs.merged_7d`)
- Identify abandoned PRs (`prs.closed_without_merge_7d`)
- Optimize review process (`prs.avg_time_to_merge_hours`)

**Example Queries**:
```
# PR merge rate
avg:prs.merged_7d{env:production}

# Time to merge trend
avg:prs.avg_time_to_merge_hours{env:production}

# PR closure ratio (merged vs abandoned)
avg:prs.merged_7d{env:production} / avg:prs.closed_without_merge_7d{env:production}
```

### Branch Metrics

Monitor branch proliferation and cleanup patterns.

| Metric Key | Description | Unit | Type |
|------------|-------------|------|------|
| `branches.active` | Active branches (excluding main/master) | count | GAUGE |

**Use Cases**:
- Detect branch sprawl (high `branches.active`)
- Encourage branch cleanup after merge
- Identify stale development

**Example Queries**:
```
# Active branches count
avg:branches.active{env:production}

# Branch growth rate (derivative)
derivative(avg:branches.active{env:production})
```

### Commit & Code Change Metrics

Track development activity and code churn.

| Metric Key | Description | Unit | Type |
|------------|-------------|------|------|
| `commits.total_7d` | Total commits to main in last 7 days | count | GAUGE |
| `commits.hotfix` | Hotfix commits in last 7 days | count | GAUGE |
| `commits.reverts` | Revert commits in last 7 days | count | GAUGE |
| `code.lines_added_7d` | Lines of code added in last 7 days | lines | GAUGE |
| `code.lines_deleted_7d` | Lines of code deleted in last 7 days | lines | GAUGE |
| `code.files_changed_7d` | Unique files modified in last 7 days | count | GAUGE |

**Use Cases**:
- Measure development velocity (`commits.total_7d`)
- Track code quality issues (`commits.hotfix`, `commits.reverts`)
- Monitor code churn (`code.lines_added_7d` + `code.lines_deleted_7d`)
- Identify refactoring efforts (high deletions, low additions)

**Example Queries**:
```
# Net code change (additions - deletions)
avg:code.lines_added_7d{env:production} - avg:code.lines_deleted_7d{env:production}

# Code churn (total changes)
avg:code.lines_added_7d{env:production} + avg:code.lines_deleted_7d{env:production}

# Hotfix rate (% of total commits)
(avg:commits.hotfix{env:production} / avg:commits.total_7d{env:production}) * 100
```

**Note**: `code.lines_*` and `code.files_changed_7d` metrics currently return 0 because the GitHub Commits API doesn't include file stats by default. To fix this, we would need to fetch each commit individually with `GET /repos/{owner}/{repo}/commits/{sha}` which would be expensive. Consider using the Repository Statistics API (`/repos/{owner}/{repo}/stats/code_frequency`) instead, which provides weekly aggregates.

### CI/CD Runtime Metrics

Monitor GitHub Actions usage, cost, and efficiency.

| Metric Key | Description | Unit | Type |
|------------|-------------|------|------|
| `ci.tests_passing_on_main` | Main branch CI test status (1=passing, 0=failing) | boolean | GAUGE |
| `ci.workflow_runs_7d` | Total workflow runs in last 7 days | count | GAUGE |
| `ci.failed_workflows_7d` | Failed workflow runs in last 7 days | count | GAUGE |
| `ci.avg_workflow_duration_minutes` | Average workflow run duration | minutes | GAUGE |

**Use Cases**:
- Track CI health (`ci.tests_passing_on_main`)
- Monitor CI usage (`ci.workflow_runs_7d`)
- Identify flaky tests (`ci.failed_workflows_7d`)
- Optimize workflow performance (`ci.avg_workflow_duration_minutes`)

**Example Queries**:
```
# CI failure rate
(avg:ci.failed_workflows_7d{env:production} / avg:ci.workflow_runs_7d{env:production}) * 100

# Workflow duration trend
avg:ci.avg_workflow_duration_minutes{env:production}

# CI reliability (uptime percentage)
avg:ci.tests_passing_on_main{env:production} * 100
```

**Note**: GitHub Actions billing minutes are not exposed via the API. To track compute costs, you would need to use the `GET /repos/{owner}/{repo}/actions/runs/{run_id}/timing` endpoint and calculate billable minutes based on runner types, but this requires fetching each run individually.

### Developer Activity Metrics

Track team engagement and individual productivity patterns.

| Metric Key | Description | Unit | Type |
|------------|-------------|------|------|
| `developers.active_7d` | Unique developers with commits in last 7 days | count | GAUGE |
| `commits.per_developer_7d` | Average commits per developer | count | GAUGE |

**Use Cases**:
- Measure team engagement (`developers.active_7d`)
- Identify productivity patterns (`commits.per_developer_7d`)
- Detect team capacity changes

**Example Queries**:
```
# Active developer count
avg:developers.active_7d{env:production}

# Individual productivity
avg:commits.per_developer_7d{env:production}

# Team velocity (commits * developers)
avg:commits.total_7d{env:production} * avg:developers.active_7d{env:production}
```

## Implementation Details

### Data Collection

**Location**: `softmax/src/softmax/dashboard/metrics.py`

All metrics use the `@system_health_metric` decorator for automatic registration:

```python
@system_health_metric(metric_key="prs.open")
def get_open_prs_count() -> int:
    """Count of currently open pull requests."""
    # Implementation
```

### GitHub API Dependencies

**Library**: `gitta` (internal GitHub API wrapper)

**Functions Used**:
- `get_pull_requests()` - List/filter PRs by state and date
- `get_branches()` - List repository branches
- `get_commits()` - List commits with filtering
- `list_all_workflow_runs()` - List GitHub Actions runs
- `get_workflow_run_jobs()` - Get jobs for specific runs

**Authentication**: GitHub token stored in AWS Secrets Manager (`github/dashboard-token`)

**Rate Limits**:
- Authenticated: 5,000 requests/hour
- Current usage: ~15 API calls every 15 minutes = ~60 calls/hour
- Well within limits

### Data Freshness

- **Collection Frequency**: Every 15 minutes (Kubernetes CronJob)
- **Lookback Window**: 7 days for most metrics
- **Point-in-time Metrics**: `prs.open`, `branches.active`, `ci.tests_passing_on_main`

### Error Handling

All metric collectors:
1. Wrap API calls in try/except blocks
2. Log errors with context
3. Return sensible defaults (0 for counts, None for averages)
4. Never crash the collection process

Example:
```python
try:
    prs = get_pull_requests(...)
    return len(prs)
except Exception as e:
    logger.error(f"Failed to get open PRs: {e}")
    return 0
```

## Dashboard Visualization

### Recommended Widgets

**PR Velocity Dashboard**:
- Timeseries: `prs.merged_7d` (line chart)
- Query Value: `prs.open` (current backlog)
- Query Value: `prs.avg_time_to_merge_hours` (review efficiency)
- Timeseries: `prs.closed_without_merge_7d` (abandoned PRs)

**Code Quality Dashboard**:
- Timeseries: `commits.hotfix` + `commits.reverts` (stacked area)
- Query Value: Hotfix rate percentage
- Timeseries: `code.lines_added_7d` vs `code.lines_deleted_7d` (dual axis)

**CI Efficiency Dashboard**:
- Timeseries: `ci.workflow_runs_7d` (bar chart)
- Timeseries: `ci.failed_workflows_7d` (overlay on runs)
- Query Value: `ci.avg_workflow_duration_minutes` (with threshold alert)
- Query Value: `ci.tests_passing_on_main` (boolean indicator)

**Team Velocity Dashboard**:
- Query Value: `developers.active_7d` (team size)
- Timeseries: `commits.total_7d` (daily commits)
- Query Value: `commits.per_developer_7d` (productivity)
- Timeseries: `branches.active` (work in progress)

### Sample Jsonnet Components

Create components in `devops/datadog/components/development.libsonnet`:

```jsonnet
local widgets = import '../lib/widgets.libsonnet';

{
  prVelocityWidget():: widgets.timeseries(
    title='PR Merge Velocity',
    queries=[{
      query: 'avg:prs.merged_7d{env:production}',
      display_type: 'line',
    }],
  ),

  ciFailureRateWidget():: widgets.query_value(
    title='CI Failure Rate (7d)',
    query='(avg:ci.failed_workflows_7d{env:production} / avg:ci.workflow_runs_7d{env:production}) * 100',
    unit: '%',
  ),

  codeChurnWidget():: widgets.timeseries(
    title='Code Churn (Lines Changed)',
    queries=[
      {
        query: 'avg:code.lines_added_7d{env:production}',
        display_type: 'area',
        style: {palette: 'green'},
      },
      {
        query: 'avg:code.lines_deleted_7d{env:production}',
        display_type: 'area',
        style: {palette: 'red'},
      },
    ],
  ),
}
```

## Limitations and Future Improvements

### Current Limitations

1. **Code Stats Limited**:
   - `code.lines_*` and `code.files_changed_7d` return 0
   - GitHub Commits API doesn't include stats by default
   - Would require individual commit fetches (expensive)

2. **No Billing Minutes**:
   - GitHub doesn't expose billable minutes via API
   - Can calculate from workflow timing, but requires per-run fetches
   - Would add significant API overhead

3. **7-Day Window Only**:
   - All historical metrics use 7-day lookback
   - Cannot track longer trends without Datadog aggregation

4. **No Per-Workflow Breakdown**:
   - CI metrics aggregate all workflows
   - Cannot isolate specific workflow performance

### Future Enhancements

1. **Enhanced Code Stats**:
   - Use Repository Statistics API (`/stats/code_frequency`)
   - Provides weekly additions/deletions for past year
   - Cached by GitHub, more efficient

2. **Per-Developer Metrics**:
   - Track commits per author (top contributors)
   - PR reviews per developer
   - Code ownership patterns

3. **Workflow-Specific Metrics**:
   - Break down by workflow name (checks, release, etc.)
   - Track job-level timing
   - Identify slow jobs

4. **PR Review Metrics**:
   - Time to first review
   - Number of review cycles
   - Reviewer participation

5. **Cost Tracking**:
   - Calculate GitHub Actions billing (runner minutes)
   - Track cost per workflow type
   - Cost per developer/team

6. **Advanced CI Metrics**:
   - Test flakiness rate (repeated failures on same commit)
   - Cache hit rates
   - Artifact upload/download sizes

## Testing

### Local Testing

Test metric collection without pushing to Datadog:

```bash
# Collect all metrics (dry run)
metta softmax-system-health report

# Expected output:
Metrics:
{
  "branches.active": 1160.0,
  "ci.avg_workflow_duration_minutes": 3.10,
  "ci.failed_workflows_7d": 38.0,
  "ci.tests_passing_on_main": 1.0,
  "ci.workflow_runs_7d": 1000.0,
  "code.files_changed_7d": 0.0,
  "code.lines_added_7d": 0.0,
  "code.lines_deleted_7d": 0.0,
  "commits.hotfix": 2.0,
  "commits.per_developer_7d": 5.06,
  "commits.reverts": 1.0,
  "commits.total_7d": 91.0,
  "developers.active_7d": 18.0,
  "prs.avg_time_to_merge_hours": 54.99,
  "prs.closed_without_merge_7d": 30.0,
  "prs.merged_7d": 97.0,
  "prs.open": 127.0
}
Skipping Datadog push
```

### Production Deployment

The metrics are automatically collected every 15 minutes via Kubernetes CronJob:

```yaml
# devops/charts/dashboard-cronjob/values.yaml
schedule: "*/15 * * * *"
command: ["uv", "run", "python", "-m", "softmax.dashboard.report", "--push"]
```

Check CronJob status:
```bash
kubectl get cronjob dashboard-cronjob -n production
kubectl logs -n production -l app=dashboard-cronjob --tail=100
```

## Troubleshooting

### Metrics Returning Zero

**Symptoms**: Specific metrics consistently return 0

**Causes**:
1. GitHub API rate limiting
2. Authentication token expired/invalid
3. Repository name/org misconfigured
4. API endpoint changed

**Debug**:
```bash
# Check logs for errors
metta softmax-system-health report 2>&1 | grep -i error

# Verify GitHub token
python -c "from softmax.aws.secrets_manager import get_secretsmanager_secret; print(get_secretsmanager_secret('github/dashboard-token')[:10])"

# Test GitHub API access
gh api repos/softmax-research/metta/pulls --jq 'length'
```

### High Latency

**Symptoms**: Metric collection takes > 30 seconds

**Causes**:
1. Too many API calls
2. Large repository with many PRs/commits
3. Network latency to GitHub

**Solutions**:
- Reduce `per_page` parameter to limit results
- Add caching layer for expensive queries
- Parallelize API calls where possible

### Missing Metrics

**Symptoms**: Some metrics not appearing in Datadog

**Causes**:
1. Metric value is `None` (filtered out)
2. Collection error (check logs)
3. Datadog API key incorrect

**Debug**:
```python
# Test metric registration
from softmax.dashboard.registry import get_system_health_metrics
print(get_system_health_metrics().keys())

# Test specific metric
from softmax.dashboard.metrics import get_open_prs_count
print(get_open_prs_count())
```

## Related Documentation

- [Datadog Integration Analysis](./DATADOG_INTEGRATION_ANALYSIS.md) - Overall architecture
- [Widget Reference](./DATADOG_WIDGET_REFERENCE.md) - Dashboard widgets
- [Quick Start](./QUICK_START.md) - Getting started with dashboards
- [softmax/dashboard/README.md](../../../softmax/src/softmax/dashboard/README.md) - Metric submission tool
