# Address PR #3384 Review Feedback

**Status**: 📋 Planned
**Priority**: High
**Created**: 2025-10-23
**Branch**: `robb/1022-datadog`
**PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)
**Reviewer**: @nishu-builder

## Review Comments Summary

From PR review on 2025-10-23:

> - definitely the general idea of multiple data collectors and a separate thing for dashboard management sound good ✅
> - interface for collectors looks solid ✅
> - i'd make sure that {service}.{category}.{metric_name} is the right topology for metrics -- that it lends itself nicely to the first ways we want to show things in the dashboard. datadog also has tags, beyond just metrics names. and i dont know how easy it is to split on metric names
> - i'd prefer typer apps over makefiles; makes composing them easier, which we're already doing a bunch of in other parts of the code
> - the actual metrics selected here don't feel super well-chosen but i'm guessing those parts may be ai generated

## Issues to Address

### 1. Metric Naming Topology & Tags Strategy ✅ **RESOLVED** (2025-10-23)

**Decision**: Keep hierarchical `{service}.{category}.{metric_name}` pattern
- Works well for hundreds/thousands of metrics
- Intuitive for both humans and AI agents
- Can add supplementary tags later if needed without breaking existing metrics
- No need to migrate or change architecture

**Current Pattern**:
```python
# Examples from current implementation
github.prs.open
github.prs.merged_7d
github.commits.total_7d
github.ci.workflow_runs_7d
```

**Questions**:
1. Should we use tags for categories instead?
   ```python
   # Alternative: Use tags
   github.metric_value  # tags: category:prs, name:open
   github.metric_value  # tags: category:commits, name:total_7d
   ```

2. How do we want to query in dashboards?
   - Group by service? (all GitHub metrics together)
   - Group by category? (all PR metrics across services)
   - Mix and match?

3. What's Datadog's best practice for metric organization?
   - Flat namespace with tags?
   - Hierarchical metric names?
   - Hybrid approach?

**Investigation Completed**:
- [x] Research Datadog metric naming best practices - Hierarchical works well
- [x] Validate approach with team - Decision to keep current pattern
- [x] Confirm no breaking changes needed

**Rationale**:
- Hierarchical naming is industry standard
- No practical limit on metric count
- Tags can be added later if needed
- Don't fix what isn't broken

### 2. Replace Makefile with Typer CLI ✅

**Concern**: Prefer typer apps over Makefiles for better composability

**Status**: ✅ **COMPLETED** (2025-10-23, commit 900b2278b1)

**Current Implementation** (was):
```makefile
# devops/datadog/Makefile
make build      # Build dashboards from Jsonnet
make push       # Push dashboards to Datadog
make list       # List dashboards
make pull       # Download dashboards
```

**Proposed Change**: Create a unified Typer CLI

```python
# devops/datadog/cli.py
import typer

app = typer.Typer()

@app.command()
def build():
    """Build all dashboards from Jsonnet sources."""
    ...

@app.command()
def push():
    """Push dashboards to Datadog."""
    ...

@app.command()
def list():
    """List all dashboards in Datadog."""
    ...

@app.command()
def pull():
    """Download dashboards from Datadog."""
    ...

# Usage:
# uv run python -m devops.datadog.cli build
# uv run python -m devops.datadog.cli push
# Or via metta:
# metta datadog build
# metta datadog push
```

**Benefits**:
- Consistent with rest of Metta codebase (already using Typer in tools/run.py)
- Better composability - can import and call from other Python code
- Type safety and validation
- Better help text and documentation
- Easier to test

**Tasks**:
- [x] Create `devops/datadog/cli.py` with Typer app
- [x] Migrate all Makefile commands to Typer commands
- [x] Add command for running collectors: `metta datadog collect <collector_name>`
- [x] Add to metta CLI or create `metta datadog` subcommand
- [x] Update documentation to use CLI instead of Make
- [x] Remove Makefile entirely (decided not to keep wrapper)

**Implementation**:
- Created comprehensive Typer CLI with all dashboard and collector commands
- Integrated as `metta datadog` subcommand in metta_cli.py
- All commands tested and working
- Documentation updated (README.md, STATUS.md, etc.)

### 3. Review and Improve Metric Selection 🚧 **IN PROGRESS** (2025-10-23)

**Strategy**: Overshoot and trim later based on dashboard usage
- Keep all existing 17 metrics (no removals)
- Add 8 new quality/velocity metrics
- Let real usage guide future refinement
- Can't go back in time to collect historical data

**Current Metrics** (17 implemented):
- `github.prs.open`, `github.prs.merged_7d`, `github.prs.closed_without_merge_7d`, `github.prs.avg_time_to_merge_hours`
- `github.branches.active`
- `github.commits.total_7d`, `github.commits.per_developer_7d`, `github.commits.hotfix`, `github.commits.reverts`
- `github.ci.workflow_runs_7d`, `github.ci.failed_workflows_7d`, `github.ci.avg_workflow_duration_minutes`, `github.ci.tests_passing_on_main`
- `github.developers.active_7d`
- `github.code.lines_added_7d`, `github.code.lines_deleted_7d`, `github.code.files_changed_7d` ✅ (fixed 2025-10-23)

**New Metrics Being Added** (8 total):

*Quality Metrics (2):*
- `github.prs.with_review_comments_pct` - % PRs with review discussion
- `github.prs.avg_comments_per_pr` - Depth of review

*Velocity Metrics (6):*
- `github.prs.time_to_first_review_hours` - Review responsiveness
- `github.prs.stale_count_14d` - PRs open >14 days
- `github.prs.cycle_time_hours` - Open to merge duration
- `github.ci.duration_p50_minutes` - Median CI time
- `github.ci.duration_p90_minutes` - 90th percentile
- `github.ci.duration_p99_minutes` - 99th percentile

**Team Goals Clarified**:
1. **Quality** (Primary) - reverts, hotfixes, CI failures, review depth
2. **Velocity** (Primary) - PR flow, review speed, CI performance
3. **Visibility** (Supporting) - only if supporting top-line goals

**Action Items**:
- [x] **Meeting with team**: Completed - goals and strategy aligned
- [x] **Define KPIs**: Quality + Velocity focus
- [ ] **Add missing high-value metrics**: In progress (Phase 1D)
- [ ] **Review after dashboard deployment**: Trim based on actual usage
- [ ] **Document metric rationale**: Will update after deployment

**Potential Improvements**:
```python
# Focus on actionable metrics
github.velocity.deployment_frequency  # How often we ship
github.quality.change_failure_rate    # How often deployments cause issues
github.quality.mttr_hours             # Mean time to resolve issues
github.reviews.avg_time_to_first_review_hours  # Collaboration health
github.ci.p95_duration_minutes        # CI performance SLA
github.prs.stale_count                # Process health (PRs open >7 days)
```

### 4. Metric Organization Best Practices

**Additional Considerations**:

**Cardinality Concerns**:
- Are we creating too many unique metric names?
- Would tags reduce cardinality?
- Cost implications of metric count

**Aggregation Patterns**:
- Which metrics should be gauge vs. count vs. distribution?
- Do we need histogram metrics for CI duration?
- Should we emit percentiles (p50, p95, p99) or raw values?

**Tags Strategy**:
```python
# Example: Using tags effectively
metric_name = "github.workflow.duration"
tags = [
    "workflow:tests",
    "branch:main",
    "result:success",
    "repo:metta"
]
```

**Tasks**:
- [ ] Define standard tags for all GitHub metrics
- [ ] Research Datadog metric type best practices (gauge/count/rate/distribution)
- [ ] Document when to use tags vs. metric name hierarchy
- [ ] Create tag taxonomy for consistent labeling

## Implementation Plan

### Phase 1: Investigation & Design (2-3 days)
- [ ] Research Datadog metric naming best practices
- [ ] Test current metric pattern in dashboard queries
- [ ] Prototype tag-based approach
- [ ] Meet with team to discuss metric selection
- [ ] Define KPIs and desired metrics
- [ ] Design Typer CLI structure

### Phase 2: Implementation (3-4 days)
- [ ] Refactor metric naming if needed (topology changes)
- [ ] Add/remove metrics based on team feedback
- [ ] Create Typer CLI for datadog commands
- [ ] Migrate Makefile commands to CLI
- [ ] Update collector interface for tags
- [ ] Add standard tags to all metrics

### Phase 3: Documentation & Testing (1-2 days)
- [ ] Update all documentation for new CLI
- [ ] Document metric naming conventions
- [ ] Document tag taxonomy
- [ ] Document metric rationale (why each exists)
- [ ] Test dashboard queries with new structure
- [ ] Update examples in README

### Phase 4: Review & Iterate (1 day)
- [ ] Request re-review from @nishu-builder
- [ ] Address any additional feedback
- [ ] Finalize before merge

## Success Criteria

- [ ] Metric naming pattern validated and documented
- [ ] Makefile replaced with Typer CLI
- [ ] Metrics reviewed and refined with team input
- [ ] All metrics have clear purpose/rationale documented
- [ ] Dashboard queries work smoothly with chosen topology
- [ ] Review feedback addressed and approved
- [ ] Documentation updated to reflect changes

## Questions for Team

1. **Metric Topology**: Do you prefer hierarchical metric names (`github.prs.open`) or flat names with tags (`github.metric` + `category:prs`)?

2. **CLI Tool**: Should the Datadog CLI be:
   - Part of `metta` command: `metta datadog <subcommand>`
   - Standalone: `uv run python -m devops.datadog.cli <subcommand>`
   - Both (metta as wrapper)?

3. **Metrics Priority**: Which metrics are most valuable?
   - Developer productivity (velocity, throughput)
   - Code quality (test coverage, reverts)
   - CI/CD health (duration, failure rate)
   - Team collaboration (review time, PR discussions)

4. **Dashboard Focus**: What dashboards do we actually need?
   - Weekly team report?
   - Real-time CI health?
   - Monthly trends?
   - Individual contributor stats?

## Related

- Main PR: [#3384](https://github.com/Metta-AI/metta/pull/3384)
- [ISSUE-migrate-github-collector.md](ISSUE-migrate-github-collector.md)
- [ISSUE-github-stats-collection.md](ISSUE-github-stats-collection.md)
- [Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)
- [CI/CD Metrics Catalog](docs/CI_CD_METRICS.md)

## Notes

- This addresses critical feedback before merging the PR
- Some changes (metric naming) may require updating existing dashboards
- Typer migration should be straightforward - we already use it elsewhere
- Metric selection requires team discussion - not a solo decision
