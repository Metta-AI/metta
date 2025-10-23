# CI/CD Metrics Documentation

## Overview

Comprehensive metrics tracking development velocity, code quality, and CI/CD efficiency for the Metta project.

**Total Metrics**: 25 (17 original + 8 new quality/velocity metrics)
**Collection Frequency**: Every 15 minutes via Kubernetes CronJob
**Submission Format**: All metrics submitted to Datadog as GAUGE values

## Metric Strategy

**Goals**: Quality + Velocity focus
- **Quality Metrics**: Track reverts, hotfixes, CI failures, review depth
- **Velocity Metrics**: Track PR flow, review speed, CI performance
- **Visibility Metrics**: Support top-line goals only

**Philosophy**: "Overshoot and trim later"
- Collect all potentially useful metrics now
- Let dashboard usage guide future refinement
- Can't go back in time for historical data

## Metric Categories

### Pull Request Metrics (8 metrics)

Track PR velocity, merge patterns, and code review efficiency.

| Metric Key | Description | Unit | Type | Added |
|------------|-------------|------|------|-------|
| `prs.open` | Currently open pull requests | count | GAUGE | Original |
| `prs.merged_7d` | PRs merged in last 7 days | count | GAUGE | Original |
| `prs.closed_without_merge_7d` | PRs closed without merge in last 7 days | count | GAUGE | Original |
| `prs.avg_time_to_merge_hours` | Average time from PR creation to merge | hours | GAUGE | Original |
| **`prs.stale_count_14d`** | PRs open for more than 14 days | count | GAUGE | **Phase 1D** |
| **`prs.cycle_time_hours`** | Average creation-to-merge duration | hours | GAUGE | **Phase 1D** |
| **`prs.with_review_comments_pct`** | % PRs with any review comments | percentage | GAUGE | **Phase 1D** |
| **`prs.avg_comments_per_pr`** | Average comments per PR | count | GAUGE | **Phase 1D** |

**Use Cases**:
- Track PR backlog growth (`prs.open` trending up)
- Measure team velocity (`prs.merged_7d`)
- Identify abandoned PRs (`prs.closed_without_merge_7d`)
- Optimize review process (`prs.avg_time_to_merge_hours`)
- **Identify stale PRs** (`prs.stale_count_14d` > threshold)
- **Track DORA lead time** (`prs.cycle_time_hours`)
- **Monitor review bottlenecks** (`prs.with_review_comments_pct` low = bottleneck)

**Example Queries**:
```
# PR merge rate
avg:prs.merged_7d{source:softmax-system-health}

# Time to merge trend
avg:prs.avg_time_to_merge_hours{source:softmax-system-health}

# Stale PR alert (> 50 stale PRs is concerning)
avg:prs.stale_count_14d{source:softmax-system-health} > 50

# Review engagement (should be > 80%)
avg:prs.with_review_comments_pct{source:softmax-system-health}
```

**Current Values** (as of 2025-10-23):
- `prs.open`: 122
- `prs.merged_7d`: 92
- `prs.stale_count_14d`: **70** (significant!)
- `prs.cycle_time_hours`: 59.89 hours
- `prs.with_review_comments_pct`: 0% (GitHub API limitation - see notes)
- `prs.avg_comments_per_pr`: 0.0 (GitHub API limitation - see notes)

**Note on Review Comments**: The `comments` field in GitHub's PR API returns issue comments, not review comments. This repository uses review comments for PR discussions, so these metrics return 0%. To track review comments, we would need to fetch `/pulls/{pr_number}/reviews` separately (60+ additional API calls).

### Branch Metrics (1 metric)

Monitor branch proliferation and cleanup patterns.

| Metric Key | Description | Unit | Type | Added |
|------------|-------------|------|------|-------|
| `branches.active` | Active branches (excluding main/master) | count | GAUGE | Original |

**Use Cases**:
- Detect branch sprawl (high `branches.active`)
- Encourage branch cleanup after merge
- Identify stale development

**Current Value**: 1,158 branches

### Commit & Code Change Metrics (7 metrics)

Track development activity and code churn.

| Metric Key | Description | Unit | Type | Added |
|------------|-------------|------|------|-------|
| `commits.total_7d` | Total commits to main in last 7 days | count | GAUGE | Original |
| `commits.per_developer_7d` | Average commits per developer | count | GAUGE | Original |
| `commits.hotfix` | Hotfix commits in last 7 days | count | GAUGE | Original |
| `commits.reverts` | Revert commits in last 7 days | count | GAUGE | Original |
| `code.lines_added_7d` | Lines of code added in last 7 days | lines | GAUGE | Original |
| `code.lines_deleted_7d` | Lines of code deleted in last 7 days | lines | GAUGE | Original |
| `code.files_changed_7d` | Unique files modified in last 7 days | count | GAUGE | Original |

**Use Cases**:
- Measure development velocity (`commits.total_7d`)
- Track code quality issues (`commits.hotfix`, `commits.reverts`)
- Monitor code churn (`code.lines_added_7d` + `code.lines_deleted_7d`)
- Identify refactoring efforts (high deletions, low additions)

**Example Queries**:
```
# Net code change (additions - deletions)
avg:code.lines_added_7d{source:softmax-system-health} - avg:code.lines_deleted_7d{source:softmax-system-health}

# Code churn (total changes)
avg:code.lines_added_7d{source:softmax-system-health} + avg:code.lines_deleted_7d{source:softmax-system-health}

# Hotfix rate (% of total commits)
(avg:commits.hotfix{source:softmax-system-health} / avg:commits.total_7d{source:softmax-system-health}) * 100
```

**Current Values** (as of 2025-10-23):
- `commits.total_7d`: 84
- `commits.per_developer_7d`: 4.61
- `commits.hotfix`: 1
- `commits.reverts`: 1
- `code.lines_added_7d`: **82,069** ✅ (fixed in Phase 1C!)
- `code.lines_deleted_7d`: **41,634** ✅ (fixed in Phase 1C!)
- `code.files_changed_7d`: **1,052** ✅ (fixed in Phase 1C!)

**Fix Applied (Phase 1C)**: Code statistics metrics were returning 0 because the GitHub list commits endpoint doesn't include stats. We now fetch individual commits with `get_commit_with_stats()` to retrieve accurate statistics.

### CI/CD Runtime Metrics (7 metrics)

Monitor GitHub Actions usage, cost, efficiency, and performance SLAs.

| Metric Key | Description | Unit | Type | Added |
|------------|-------------|------|------|-------|
| `ci.tests_passing_on_main` | Main branch CI test status (1=passing, 0=failing) | boolean | GAUGE | Original |
| `ci.workflow_runs_7d` | Total workflow runs in last 7 days | count | GAUGE | Original |
| `ci.failed_workflows_7d` | Failed workflow runs in last 7 days | count | GAUGE | Original |
| `ci.avg_workflow_duration_minutes` | Average workflow run duration | minutes | GAUGE | Original |
| **`ci.duration_p50_minutes`** | Median (50th percentile) workflow duration | minutes | GAUGE | **Phase 1D** |
| **`ci.duration_p90_minutes`** | 90th percentile workflow duration | minutes | GAUGE | **Phase 1D** |
| **`ci.duration_p99_minutes`** | 99th percentile workflow duration | minutes | GAUGE | **Phase 1D** |

**Use Cases**:
- Track CI health (`ci.tests_passing_on_main`)
- Monitor CI usage (`ci.workflow_runs_7d`)
- Identify flaky tests (`ci.failed_workflows_7d`)
- Optimize workflow performance (`ci.avg_workflow_duration_minutes`)
- **Track SLA compliance** (`ci.duration_p90_minutes` < target)
- **Identify outliers** (`ci.duration_p99_minutes` >> `ci.duration_p50_minutes`)

**Example Queries**:
```
# CI failure rate
(avg:ci.failed_workflows_7d{source:softmax-system-health} / avg:ci.workflow_runs_7d{source:softmax-system-health}) * 100

# Workflow duration trend
avg:ci.avg_workflow_duration_minutes{source:softmax-system-health}

# SLA tracking (% runs under 5 minutes = p90 target)
avg:ci.duration_p90_minutes{source:softmax-system-health} < 5

# Performance spread (p99 vs p50 ratio)
avg:ci.duration_p99_minutes{source:softmax-system-health} / avg:ci.duration_p50_minutes{source:softmax-system-health}
```

**Current Values** (as of 2025-10-23):
- `ci.tests_passing_on_main`: 1 (passing)
- `ci.workflow_runs_7d`: 1,000
- `ci.failed_workflows_7d`: 31
- `ci.avg_workflow_duration_minutes`: 3.15
- **`ci.duration_p50_minutes`: 1.18** (fast median!)
- **`ci.duration_p90_minutes`: 7.53** (acceptable)
- **`ci.duration_p99_minutes`: 16.05** (some slow outliers)

**Insight**: CI performance is generally excellent (median 1.2 min), but p99 shows some 16-minute outliers worth investigating.

### Developer Activity Metrics (2 metrics)

Track team engagement and individual productivity patterns.

| Metric Key | Description | Unit | Type | Added |
|------------|-------------|------|------|-------|
| `developers.active_7d` | Unique developers with commits in last 7 days | count | GAUGE | Original |
| `commits.per_developer_7d` | Average commits per developer | count | GAUGE | Original |

**Use Cases**:
- Measure team engagement (`developers.active_7d`)
- Identify productivity patterns (`commits.per_developer_7d`)
- Detect team capacity changes

**Current Values**:
- `developers.active_7d`: 18
- `commits.per_developer_7d`: 4.61

## Metric Summary by Phase

### Original Metrics (17)
Implemented before Phase 1:
- Pull Requests: 4 metrics
- Branches: 1 metric
- Commits & Code: 7 metrics
- CI/CD: 4 metrics
- Developers: 2 metrics (one shared with Commits)

### Phase 1C: Fix Code Stats (3 metrics fixed)
Fixed broken metrics that were returning 0:
- ✅ `code.lines_added_7d` - Now returns actual values
- ✅ `code.lines_deleted_7d` - Now returns actual values
- ✅ `code.files_changed_7d` - Now returns actual values

### Phase 1D: Quality & Velocity (8 new metrics)
Added based on team discussion:
- **Quality (2)**: `prs.with_review_comments_pct`, `prs.avg_comments_per_pr`
- **Velocity (6)**: `prs.stale_count_14d`, `prs.cycle_time_hours`, `ci.duration_p50_minutes`, `ci.duration_p90_minutes`, `ci.duration_p99_minutes`, `prs.time_to_first_review_hours` (returns None due to API limitation)

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
- **`get_commit_with_stats()`** - Fetch individual commit with stats (Phase 1C)
- `list_all_workflow_runs()` - List GitHub Actions runs
- `get_workflow_run_jobs()` - Get jobs for specific runs

**Authentication**: GitHub token stored in AWS Secrets Manager (`github/dashboard-token`)

**Rate Limits**:
- Authenticated: 5,000 requests/hour
- Current usage: ~150 API calls every 15 minutes = ~600 calls/hour
- Still well within limits (12% utilization)

**API Call Breakdown**:
- PR metrics: ~5 calls
- Commit metrics: ~95 calls (fetching individual commit stats)
- CI metrics: ~40 calls
- Branch metrics: ~1 call
- Developer metrics: ~5 calls

### Data Freshness

- **Collection Frequency**: Every 15 minutes (Kubernetes CronJob)
- **Lookback Window**: 7 days for most metrics, 14 days for stale PRs
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

## Dashboard Recommendations

### Executive Dashboard
- PR velocity: `prs.merged_7d`, `prs.open`
- Quality: `commits.reverts`, `commits.hotfix`
- CI health: `ci.tests_passing_on_main`, `ci.failed_workflows_7d`

### Engineering Dashboard
- Code churn: `code.lines_added_7d`, `code.lines_deleted_7d`
- PR flow: `prs.stale_count_14d`, `prs.cycle_time_hours`
- CI performance: `ci.duration_p50_minutes`, `ci.duration_p90_minutes`, `ci.duration_p99_minutes`

### Team Health Dashboard
- Engagement: `developers.active_7d`
- Workload: `commits.per_developer_7d`
- Review health: `prs.with_review_comments_pct`, `prs.avg_comments_per_pr`

## Known Limitations

### 1. Review Comments Metrics
**Issue**: `prs.with_review_comments_pct` and `prs.avg_comments_per_pr` return 0%

**Root Cause**: GitHub's PR API `comments` field returns issue comments, not review comments. The repository uses review comments for PR discussions.

**Fix**: Would require fetching `/pulls/{pr_number}/reviews` for each PR (~60+ additional API calls per collection).

**Workaround**: These metrics still provide value when teams use issue comments, or we can enhance in Phase 2.

### 2. Time to First Review
**Issue**: `prs.time_to_first_review_hours` returns None

**Root Cause**: GitHub's list PRs endpoint doesn't include comment timestamps. Uses `updated_at` as approximation.

**Fix**: Would require fetching PR timeline events or review comments for each PR.

### 3. GitHub Actions Billing
**Note**: GitHub Actions billing minutes are not exposed via the standard API. To track compute costs, you would need to use the `/repos/{owner}/{repo}/actions/runs/{run_id}/timing` endpoint and calculate billable minutes based on runner types.

## Future Enhancements

### DORA Metrics
```
github.dora.deployment_frequency         # Deployments per day
github.dora.lead_time_hours              # Code to production time
github.dora.change_failure_rate          # % deployments causing issues
github.dora.mttr_hours                   # Mean time to recovery
```

### Service-Level Objectives (SLOs)
```
github.slo.pr_merge_time_under_24h_pct   # % PRs merged within 24h
github.slo.ci_duration_under_5m_pct      # % CI runs under 5 minutes
```

### Multi-Repository Support
When tracking multiple repositories, add tags:
```python
tags = ["repo:metta", "repo:pufferlib"]
```

## References

- [METRIC_CONVENTIONS.md](METRIC_CONVENTIONS.md) - Naming patterns and conventions
- [WORKPLAN.md](../WORKPLAN.md) - Implementation plan and team decisions
- [Datadog Metrics Best Practices](https://docs.datadoghq.com/developers/guide/what-best-practices-are-recommended-for-naming-metrics-and-tags/)

---

Last updated: 2025-10-23 (Phase 1D complete - 25 metrics total)
