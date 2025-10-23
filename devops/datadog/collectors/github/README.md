# GitHub Collector

Collects development metrics from GitHub API for the Metta repository.

**Status**: ✅ **Implemented** (currently in `softmax/dashboard/metrics.py`, to be migrated)

## Metrics Collected

### Pull Request Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.prs.open` | Currently open pull requests | count | ✅ Implemented |
| `github.prs.merged_7d` | PRs merged in last 7 days | count | ✅ Implemented |
| `github.prs.closed_without_merge_7d` | PRs closed without merge in last 7 days | count | ✅ Implemented |
| `github.prs.avg_time_to_merge_hours` | Average time from PR creation to merge | hours | ✅ Implemented |
| `github.prs.stale` | PRs open for >30 days | count | 📋 Planned |
| `github.prs.draft` | Draft PRs | count | 📋 Planned |

### Branch Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.branches.active` | Active branches (excluding main/master) | count | ✅ Implemented |
| `github.branches.stale` | Branches with no commits in >30 days | count | 📋 Planned |
| `github.branches.merged_7d` | Branches merged in last 7 days | count | 📋 Planned |

### Commit Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.commits.total_7d` | Total commits to main in last 7 days | count | ✅ Implemented |
| `github.commits.hotfix` | Hotfix commits in last 7 days | count | ✅ Implemented |
| `github.commits.reverts` | Revert commits in last 7 days | count | ✅ Implemented |
| `github.commits.per_developer_7d` | Average commits per developer | count | ✅ Implemented |

### Code Change Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.code.lines_added_7d` | Lines of code added in last 7 days | lines | ⚠️ Returns 0 (needs fix) |
| `github.code.lines_deleted_7d` | Lines of code deleted in last 7 days | lines | ⚠️ Returns 0 (needs fix) |
| `github.code.files_changed_7d` | Unique files modified in last 7 days | count | ⚠️ Returns 0 (needs fix) |
| `github.code.churn_7d` | Total lines changed (adds + deletes) | lines | 📋 Planned |

### CI/CD Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.ci.tests_passing_on_main` | Main branch test status (1=passing, 0=failing) | boolean | ✅ Implemented |
| `github.ci.workflow_runs_7d` | Total workflow runs in last 7 days | count | ✅ Implemented |
| `github.ci.failed_workflows_7d` | Failed workflow runs in last 7 days | count | ✅ Implemented |
| `github.ci.avg_workflow_duration_minutes` | Average workflow run duration | minutes | ✅ Implemented |
| `github.ci.success_rate_7d` | Workflow success rate (%) | percent | 📋 Planned |
| `github.ci.flaky_tests` | Tests that failed then passed on same commit | count | 📋 Planned |

### Developer Activity Metrics

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.developers.active_7d` | Unique developers with commits in last 7 days | count | ✅ Implemented |
| `github.developers.top_contributor_commits` | Commits by top contributor | count | 📋 Planned |
| `github.reviews.count_7d` | PR reviews in last 7 days | count | 📋 Planned |
| `github.reviews.avg_time_to_first_review_hours` | Avg time to first review | hours | 📋 Planned |

### Repository Health

| Metric | Description | Unit | Status |
|--------|-------------|------|--------|
| `github.issues.open` | Currently open issues | count | 📋 Planned |
| `github.issues.closed_7d` | Issues closed in last 7 days | count | 📋 Planned |
| `github.issues.stale` | Issues open for >60 days | count | 📋 Planned |

## Configuration

### Required Secrets (AWS Secrets Manager)

- `github/dashboard-token` - GitHub personal access token with repo access

### Environment Variables

- `GITHUB_ORG` - GitHub organization (default: "softmax-research")
- `GITHUB_REPO` - Repository name (default: "metta")

### Collection Schedule

**Current**: Every 15 minutes (`*/15 * * * *`)

## Known Issues

1. **Code change metrics return 0**: GitHub Commits API doesn't include file stats by default
   - **Solution**: Use Repository Statistics API (`/stats/code_frequency`) for weekly aggregates
   - **Tracking**: Issue TBD

## API Rate Limits

- **Authenticated**: 5,000 requests/hour
- **Current usage**: ~15 API calls per run = ~60 calls/hour
- **Headroom**: Well within limits

## Migration Notes

Currently implemented in `softmax/src/softmax/dashboard/metrics.py`. Migration plan:

1. Create `collectors/github/collector.py` (BaseCollector subclass)
2. Move metrics to `collectors/github/metrics.py`
3. Update `devops/charts/dashboard-cronjob/values.yaml` → `devops/charts/datadog-collectors/values.yaml`
4. Test locally
5. Deploy new structure
6. Verify metrics still flowing
7. Remove old `softmax/dashboard` code

## Related Documentation

- [CI/CD Metrics Catalog](../../docs/CI_CD_METRICS.md) - Detailed metric documentation
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md) - System design
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md) - Implementation guide

## Maintenance

- **Owner**: DevOps Team
- **API Version**: GitHub REST API v3
- **Dependencies**: `gitta`, `httpx`, `boto3`
- **Last Updated**: 2025-10-22
