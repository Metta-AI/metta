# Datadog Metric Naming Conventions

This document defines the naming conventions and patterns for all Datadog metrics in the Metta project.

## Naming Pattern

**Format**: `{service}.{category}.{metric_name}`

### Examples

```
github.prs.open
github.prs.merged_7d
github.commits.total_7d
github.ci.workflow_runs_7d
github.code.lines_added_7d
```

## Pattern Breakdown

### Service Component

The first component identifies the source service or system:

- `github` - GitHub repository metrics
- `aws` - AWS infrastructure metrics (future)
- `datadog` - Datadog meta-metrics (future)
- `system` - System/host metrics (future)

### Category Component

The second component groups related metrics by functional area:

**GitHub Categories:**
- `prs` - Pull request metrics
- `commits` - Commit and code change metrics
- `ci` - CI/CD pipeline metrics
- `branches` - Branch management metrics
- `developers` - Developer activity metrics
- `code` - Code statistics (lines, files)
- `quality` - Code quality metrics (future)
- `reviews` - Code review metrics (future)

### Metric Name Component

The third component describes the specific metric:

**Naming Guidelines:**
- Use descriptive names that clearly indicate what is measured
- Include time windows when relevant (e.g., `_7d`, `_30d`)
- Use units in the name when helpful (e.g., `_hours`, `_minutes`, `_pct`)
- Use snake_case for readability

**Common Suffixes:**
- `_7d`, `_30d` - Time window (7 days, 30 days)
- `_count` - Count of items
- `_pct` - Percentage value
- `_hours`, `_minutes` - Duration metrics
- `_p50`, `_p90`, `_p99` - Percentile values
- `_avg`, `_min`, `_max` - Statistical aggregations

## Metric Types

### Gauge
Metrics that represent a point-in-time value:
- `github.prs.open` - Current count
- `github.branches.active` - Current count
- `github.ci.tests_passing_on_main` - Binary status (0 or 1)

### Count/Rate
Metrics that represent cumulative values or rates over time:
- `github.prs.merged_7d` - Count over 7 days
- `github.commits.total_7d` - Count over 7 days
- `github.ci.workflow_runs_7d` - Count over 7 days

### Histogram/Distribution
Metrics with percentile calculations:
- `github.ci.duration_p50_minutes` - Median
- `github.ci.duration_p90_minutes` - 90th percentile
- `github.ci.duration_p99_minutes` - 99th percentile

## Time Windows

Standardize on these time windows for consistency:

- `_7d` - Last 7 days (weekly view)
- `_30d` - Last 30 days (monthly view)
- `_14d` - Last 14 days (bi-weekly view, used for stale PRs)

**Rationale:** These align with common review cycles and sprint durations.

## Tags Strategy

### Current Approach
We use hierarchical metric names without tags initially. This keeps the system simple and intuitive.

### Future Tag Strategy
Tags can be added later without breaking existing metrics:

```python
# Potential future tags
tags = [
    "repo:metta",
    "org:metta-ai",
    "environment:production",
    "category:pull_requests",
    "team:core"
]
```

**When to add tags:**
- Multi-repo support needed
- Multiple environments (dev, staging, prod)
- Team-based filtering required
- Cross-cutting concerns (all CI metrics, all quality metrics)

## Design Decisions

### Why Hierarchical Names?

1. **Intuitive**: Easy to understand and navigate
2. **Discoverable**: Can browse metrics by prefix
3. **Flexible**: Works well for hundreds or thousands of metrics
4. **Compatible**: Standard approach used by many systems
5. **Queryable**: Easy to filter with wildcards (`github.prs.*`)

### Why Not Tags Initially?

1. **Simplicity**: Don't need complexity until we need it
2. **Incremental**: Can add tags later without breaking changes
3. **Cost**: Each unique tag combination can increase cardinality
4. **Learning Curve**: Hierarchical is easier for new users

### When Would We Add Tags?

- Multiple repositories need tracking
- Multiple environments (dev/staging/prod) deployed
- Cross-service queries become common
- Team-based filtering needed

## Validation Rules

Metrics should follow these rules:

1. **Length**: Keep total length under 200 characters
2. **Characters**: Use only lowercase letters, numbers, underscores, and dots
3. **Uniqueness**: Each metric name must be globally unique
4. **Stability**: Don't rename metrics without migration plan
5. **Documentation**: Every metric must have a description and rationale

## Anti-Patterns

**❌ Avoid:**
- CamelCase or PascalCase: `github.PRs.Open`
- Redundant prefixes: `github.github_prs.open`
- Unclear abbreviations: `github.pr.cnt`
- Inconsistent time windows: mixing `_weekly` and `_7d`
- Very long names: `github.prs.merged_in_last_seven_days_count`

**✅ Prefer:**
- snake_case: `github.prs.open`
- Clear structure: `github.prs.open`
- Standard abbreviations: `prs`, `ci`, `pct`
- Standard suffixes: `_7d`, `_30d`
- Concise names: `github.prs.merged_7d`

## Migration Strategy

If we need to rename or restructure metrics:

1. **Add new metrics** alongside old ones
2. **Deprecation period**: Keep both for 30 days
3. **Update dashboards** to use new metrics
4. **Remove old metrics** after deprecation period
5. **Document changes** in changelog

## Examples by Category

### Pull Requests
```
github.prs.open                          # Currently open PRs
github.prs.merged_7d                     # Merged in last 7 days
github.prs.closed_without_merge_7d       # Closed without merge
github.prs.avg_time_to_merge_hours       # Average time from open to merge
github.prs.stale_count_14d               # Open for more than 14 days
github.prs.cycle_time_hours              # Creation to merge duration
github.prs.with_review_comments_pct      # % with any comments
github.prs.avg_comments_per_pr           # Average comment count
```

### CI/CD
```
github.ci.workflow_runs_7d               # Total runs in 7 days
github.ci.failed_workflows_7d            # Failed runs in 7 days
github.ci.avg_workflow_duration_minutes  # Average duration
github.ci.duration_p50_minutes           # Median duration
github.ci.duration_p90_minutes           # 90th percentile
github.ci.duration_p99_minutes           # 99th percentile
github.ci.tests_passing_on_main          # Binary: 1 = passing, 0 = failing
```

### Commits & Code
```
github.commits.total_7d                  # Total commits in 7 days
github.commits.per_developer_7d          # Average commits per developer
github.commits.hotfix                    # Hotfix commits (7 days)
github.commits.reverts                   # Revert commits (7 days)
github.code.lines_added_7d               # Lines added in 7 days
github.code.lines_deleted_7d             # Lines deleted in 7 days
github.code.files_changed_7d             # Unique files changed
```

### Developers & Branches
```
github.developers.active_7d              # Unique developers with commits
github.branches.active                   # Non-main branches
```

## Future Considerations

### Multi-Repository Support
When tracking multiple repositories:

**Option 1: Prefix per repo**
```
github_metta.prs.open
github_pufferlib.prs.open
```

**Option 2: Tags**
```
github.prs.open [repo:metta]
github.prs.open [repo:pufferlib]
```

**Recommendation**: Use tags when we reach 3+ repositories.

### Service-Level Objectives (SLOs)
```
github.slo.pr_merge_time_under_24h_pct   # % PRs merged within 24h
github.slo.ci_duration_under_5m_pct      # % CI runs under 5 minutes
```

### DORA Metrics
```
github.dora.deployment_frequency         # Deployments per day
github.dora.lead_time_hours              # Code to production time
github.dora.change_failure_rate          # % deployments causing issues
github.dora.mttr_hours                   # Mean time to recovery
```

## References

- [Datadog Metric Naming Best Practices](https://docs.datadoghq.com/developers/guide/what-best-practices-are-recommended-for-naming-metrics-and-tags/)
- [WORKPLAN.md](../WORKPLAN.md) - Team decisions and rationale
- [CI_CD_METRICS.md](CI_CD_METRICS.md) - Complete metric catalog

---

Last updated: 2025-10-23
