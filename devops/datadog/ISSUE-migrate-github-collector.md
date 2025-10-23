# Migrate GitHub Metrics to Modular Collector Architecture

**Status**: 📋 Planned
**Priority**: High
**Created**: 2025-10-22
**Branch**: `robb/1022-datadog`

## Overview

We have successfully implemented 17 enhanced GitHub metrics that are currently collecting and pushing to Datadog. However, they live in `softmax/src/softmax/dashboard/metrics.py` and need to be migrated to the new modular collector architecture in `devops/datadog/`.

## Current Status

### ✅ Working
- **17 GitHub metrics** implemented and tested
- Successfully pushing to Datadog via `metta softmax-system-health report --push`
- Running on CronJob every 15 minutes (deployed as `dashboard-cronjob` chart)
- All metrics except code stats are collecting correctly

### 📋 Metrics Inventory

**Pull Requests (4 metrics)**
- `github.prs.open`: 128
- `github.prs.merged_7d`: 97
- `github.prs.closed_without_merge_7d`: 29
- `github.prs.avg_time_to_merge_hours`: 55.0

**Branches (1 metric)**
- `github.branches.active`: 1162

**Commits (4 metrics)**
- `github.commits.total_7d`: 91
- `github.commits.per_developer_7d`: 5.06
- `github.commits.hotfix`: 2
- `github.commits.reverts`: 1

**CI/CD (4 metrics + 6 duration stats)**
- `github.ci.workflow_runs_7d`: 1000
- `github.ci.failed_workflows_7d`: 37
- `github.ci.avg_workflow_duration_minutes`: 3.06
- `github.ci.tests_passing_on_main`: 1
- **New**: `github.ci.workflow_duration_minutes.min` (planned)
- **New**: `github.ci.workflow_duration_minutes.p50` (planned)
- **New**: `github.ci.workflow_duration_minutes.p75` (planned)
- **New**: `github.ci.workflow_duration_minutes.p90` (planned)
- **New**: `github.ci.workflow_duration_minutes.p95` (planned)
- **New**: `github.ci.workflow_duration_minutes.max` (planned)

**Developers (1 metric)**
- `github.developers.active_7d`: 18

**Code Stats (3 metrics - ⚠️ BROKEN)**
- `github.code.lines_added_7d`: 0 (returns 0)
- `github.code.lines_deleted_7d`: 0 (returns 0)
- `github.code.files_changed_7d`: 0 (returns 0)

**Total**: 17 implemented (14 working, 3 broken) + 6 planned duration stats = 23 metrics

## Problems to Fix

### 1. Code Statistics Metrics Return 0

**Root Cause**: The GitHub `/repos/{owner}/{repo}/commits` endpoint doesn't include `stats` field by default. We're trying to read `commit.get("stats", {})` but it doesn't exist in the list response.

**Investigation Results**:
- `/stats/code_frequency` - Returns 202 "Computing" (unreliable for real-time)
- `/stats/commit_activity` - Also returns 202 (unreliable)
- Individual commit fetch (`/commits/{sha}`) - ✅ **DOES include stats**

**Test Results**:
```json
{
  "stats": {
    "total": 180,
    "additions": 121,
    "deletions": 59
  }
}
```

**Fix Options**:

**Option A: Individual Commit Fetches** (Recommended)
- Fetch each commit individually to get stats
- Cost: 91 API calls for 91 commits (well within 5000/hour limit)
- Trade-off: More API calls, but guaranteed to work
- Implementation: Add helper function `get_commit_with_stats(repo, sha)` to gitta

**Option B: PR-Based Stats** (Alternative)
- Calculate from merged PRs instead of commits
- More meaningful (measures reviewed/merged changes)
- Fewer API calls

**Option C: Leave As-Is**
- Document as known limitation
- These are nice-to-have metrics, not critical

### 2. CI Workflow Duration Needs Distribution Statistics

**Current**: Only `ci.avg_workflow_duration_minutes` (3.06)

**Problem**: Average alone doesn't tell the full story:
- Hides outliers (very slow builds)
- Doesn't show consistency/reliability
- Can't identify p95/p99 performance targets
- Misses distribution shape (bimodal, long tail, etc.)

**Proposed Enhancement**: Add 6 percentile metrics:
- `min` - Fastest workflow
- `p50` - Median (typical build time, better than mean)
- `p75` - 75th percentile
- `p90` - 90th percentile
- `p95` - SLA target (95% of builds should be faster)
- `max` - Slowest workflow (outliers)

**Benefits**:
- Understand distribution of CI times
- Set SLAs: p95 < 10 minutes = 95% of builds fast
- Track reliability: p95 - p50 = consistency measure
- Identify outliers: max shows worst-case scenarios
- Better KPI: Median (p50) better than mean for skewed distributions

**Implementation**: No additional API cost - same data, different aggregations

### 3. Code Consolidation Needed

**Current Location** (temporary):
```
softmax/src/softmax/dashboard/
├── metrics.py          # All GitHub metrics
└── report.py           # CLI runner
```

**Target Location** (modular architecture):
```
devops/datadog/
├── collectors/
│   └── github/
│       ├── collector.py    # BaseCollector subclass
│       ├── metrics.py      # Metric definitions with @metric decorator
│       └── README.md       # ✅ Already exists with full documentation
├── common/
│   ├── base.py            # BaseCollector abstract class
│   ├── decorators.py      # @metric decorator
│   ├── datadog_client.py  # Datadog submission
│   └── secrets.py         # AWS Secrets Manager
└── scripts/
    └── run_collector.py   # CLI: python -m devops.datadog.scripts.run_collector github
```

**Benefits**:
- Consistent structure for all collectors
- Reusable base class and utilities
- Clear separation of concerns
- Easy to add new collectors (Skypilot, WandB, EC2, Asana)

## Tasks

### Phase 1: Fix Code Stats & Add CI Distribution
- [ ] **Decision**: Choose code stats fix approach (Option A, B, or C)
- [ ] If Option A: Add `get_commit_with_stats()` to gitta library
- [ ] Update `get_lines_added_7d()` to fetch individual commit stats
- [ ] Update `get_lines_deleted_7d()` to fetch individual commit stats
- [ ] Update `get_files_changed_7d()` to count unique files across commits
- [ ] Implement `_get_workflow_durations_7d()` helper function
- [ ] Add 6 new CI duration percentile metrics (min, p50, p75, p90, p95, max)
- [ ] Test metrics collection with `metta softmax-system-health report`
- [ ] Verify stats are non-zero and percentiles are ordered correctly
- [ ] Push to Datadog and verify: `metta softmax-system-health report --push`

### Phase 2: Architecture Migration
- [ ] Create `devops/datadog/common/base.py` with `BaseCollector` abstract class
- [ ] Create `devops/datadog/common/decorators.py` with `@metric` decorator
- [ ] Create `devops/datadog/common/datadog_client.py` for metric submission
- [ ] Create `devops/datadog/common/secrets.py` for AWS Secrets Manager access
- [ ] Create `devops/datadog/collectors/github/collector.py`
  - Implement `GitHubCollector(BaseCollector)`
  - Move authentication logic
  - Implement `collect_metrics()` method
- [ ] Move metrics from `softmax/dashboard/metrics.py` to `devops/datadog/collectors/github/metrics.py`
  - Keep same metric keys (no breaking changes)
  - Use `@metric` decorator for automatic registration
  - Group by category (PRs, commits, CI/CD, etc.)
- [ ] Create `devops/datadog/scripts/run_collector.py`
  - CLI: `python -m devops.datadog.scripts.run_collector github [--push]`
  - Support dry-run mode (default)
  - Support `--push` flag to submit to Datadog
- [ ] Update tests to use new location

### Phase 3: Deployment Migration
- [ ] Update `devops/charts/dashboard-cronjob/values.yaml`
  - Change command to use new collector script
  - Keep same schedule (15 minutes)
  - Keep same secrets/IAM role
- [ ] Test locally: `python -m devops.datadog.scripts.run_collector github --push`
- [ ] Deploy updated CronJob to staging/production
- [ ] Verify metrics still flowing to Datadog (monitor for 1 hour)
- [ ] Remove old code from `softmax/dashboard/` once verified

### Phase 4: Documentation Updates
- [ ] Update `devops/datadog/collectors/github/README.md` with migration completion
- [ ] Update `devops/datadog/docs/COLLECTORS_ARCHITECTURE.md` to reflect GitHub as implemented
- [ ] Add migration guide for future collectors
- [ ] Document testing pattern: `run_collector <name> [--push]`

## Testing Checklist

### Local Testing
```bash
# Test metric collection (dry run)
python -m devops.datadog.scripts.run_collector github

# Expected output: 23 metrics (17 existing + 6 new duration stats)
# Verify code stats are non-zero
# Verify percentiles: min < p50 < p75 < p90 < p95 < max

# Test with Datadog push
python -m devops.datadog.scripts.run_collector github --push

# Verify metrics in Datadog (check last 5 minutes)
```

### Production Verification
```bash
# Check CronJob status
kubectl get cronjobs -n monitoring

# Check recent job runs
kubectl get jobs -n monitoring | grep github-collector

# Check logs
kubectl logs -n monitoring -l collector=github --tail=100

# Monitor Datadog for 1 hour
# Verify metrics update every 15 minutes
# Verify all 23 metrics present
```

## Success Criteria

- [ ] All 23 GitHub metrics collecting successfully
- [ ] Code stats metrics return non-zero values
- [ ] CI duration metrics show proper distribution (min < p50 < p95 < max)
- [ ] Metrics pushed to Datadog every 15 minutes
- [ ] Code moved to `devops/datadog/collectors/github/`
- [ ] Reusable base architecture in place for future collectors
- [ ] Old code removed from `softmax/dashboard/`
- [ ] Documentation updated
- [ ] No breaking changes (same metric keys, same schedule)

## Related Documentation

- [Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector Guide](docs/ADDING_NEW_COLLECTOR.md)
- [GitHub Collector README](collectors/github/README.md)
- [CI/CD Metrics Catalog](docs/CI_CD_METRICS.md)
- [Helm CronJob Conventions](docs/HELM_CRONJOB_CONVENTIONS.md)
- [GitHub Stats Collection Issue](ISSUE-github-stats-collection.md)

## Future Work

After GitHub collector migration is complete, implement additional collectors:

1. **Skypilot** (High Priority) - Job orchestration, compute costs
2. **EC2** (High Priority) - Instance tracking, cost monitoring
3. **WandB** (Medium Priority) - Training runs, GPU utilization
4. **Asana** (Low Priority) - Task tracking, project velocity

**Total Planned Metrics**: 128+ (23 GitHub, 105+ from other collectors)

## Notes

- Current implementation is on branch `robb/1022-datadog` (not merged to main yet)
- Helm chart location confirmed as standard: `devops/charts/` is correct
- All collectors will run at different frequencies based on data freshness needs
- Testing pattern established: `[collector_command] [--push]` for all collectors
- CI duration distribution is critical for KPI tracking
