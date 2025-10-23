# Dashboard Implementation Workplan

## Overview

Concrete implementation plan for building all Datadog dashboards in the Metta observability system. This workplan focuses exclusively on dashboard creation, assuming collectors are implemented per the main [WORKPLAN.md](../WORKPLAN.md).

**Status**: Phase 1 complete - 3/3 dashboards deployed (GitHub ✅, Skypilot ✅, Health Rollup ✅)
**Last Updated**: 2025-10-23 (System Health Rollup dashboard deployed)
**Related Docs**: [DASHBOARD_DESIGN.md](DASHBOARD_DESIGN.md), [HEALTH_DASHBOARD_SPEC.md](HEALTH_DASHBOARD_SPEC.md)

---

## Dashboard Inventory

| Dashboard | Collector Dependency | Status | Priority | Time Spent | Dashboard ID |
|-----------|---------------------|---------|----------|-----------|--------------|
| GitHub CI/CD | GitHub collector | ✅ **Complete** | P0 | 2 hours | [7gy-9ub-2sq](https://app.datadoghq.com/dashboard/7gy-9ub-2sq/github-cicd-dashboard) |
| Skypilot Jobs | Skypilot collector | ✅ **Complete** | P1 | 1.5 hours | [wjp-n4n-dsf](https://app.datadoghq.com/dashboard/wjp-n4n-dsf/skypilot-jobs-dashboard) |
| System Health Rollup | FoM collector | ✅ **Complete** | P0 | 2 hours | [2mx-kfj-8pi](https://app.datadoghq.com/dashboard/2mx-kfj-8pi/system-health-rollup) |
| Training & WandB | WandB collector | Blocked | P1 | 6 hours (est) | - |
| Eval & Testing | Eval collector | Blocked | P2 | 4 hours (est) | - |
| Collector Health | All collectors | Not started | P2 | 3 hours (est) | - |

**Progress**: 3/6 complete (50%)
**Total Effort**: ~13 hours remaining (~2 days of focused work)

---

## Completed Dashboards

### ✅ System Health Rollup Dashboard (2025-10-23)

**Dashboard**: [System Health Rollup](https://app.datadoghq.com/dashboard/2mx-kfj-8pi/system-health-rollup)
**ID**: `2mx-kfj-8pi`
**Time Spent**: 2 hours (vs 8 hour estimate)
**Approach**: JSON-first with timeseries widgets

**What We Built**:
- **9 widgets** across 8 rows
- **2 note widgets**: Dashboard overview and future phases
- **7 timeseries widgets**: All CI FoM metrics with area charts and threshold markers
- All metrics normalized to [0.0, 1.0] Figure of Merit (FoM) scale
- Color coding: Green (0.7-1.0) = Healthy, Yellow (0.3-0.7) = Warning, Red (0.0-0.3) = Critical

**Key Features**:
- ✅ Figure of Merit (FoM) visualization for CI/CD health metrics
- ✅ Threshold markers showing target/warning/critical levels for each metric
- ✅ Area charts for trend visualization
- ✅ Consistent color palette across all widgets
- ✅ Note widgets explaining FoM scale and future phases
- ✅ 7-day rolling window for trend analysis

**Metrics Used** (7 CI FoM metrics from GitHub collector):
- `health.ci.tests_passing.fom` - Binary: all tests passing = 1.0
- `health.ci.failing_workflows.fom` - Fewer is better: 0→1.0, 5+→0.0
- `health.ci.hotfix_count.fom` - Fewer is better: 0→1.0, 10+→0.0
- `health.ci.revert_count.fom` - Fewer is better: 0→1.0, 2+→0.0
- `health.ci.duration_p90.fom` - Faster is better: 3min→1.0, 10min+→0.0
- `health.ci.stale_prs.fom` - Fewer is better: 0→1.0, 50+→0.0
- `health.ci.pr_cycle_time.fom` - Faster is better: 24h→1.0, 72h+→0.0

**Phase 1 Complete**:
- ✅ FoM collector implemented with 7 CI metric formulas
- ✅ Comprehensive unit tests (16 test cases, all passing)
- ✅ Dashboard deployed and accessible
- ✅ Bug fix: FoM clamping to [0.0, 1.0] range

**Future Phases**:
- Phase 2: Training metrics (9 FoMs) - requires WandB collector
- Phase 3: Eval metrics (5 FoMs) - requires Eval collector

**Learnings**:
1. **Note widget validation** - Note widgets don't support `title` property (API validation error)
2. **Reflow type** - Must use `reflow_type: "fixed"` when specifying layout properties
3. **FoM scaling** - Linear scaling formulas with proper clamping to [0.0, 1.0] range
4. **Test-driven development** - Unit tests caught clamping bug before deployment
5. **Efficient development** - 2 hours vs 8 hour estimate (75% faster than planned)

---

### ✅ Skypilot Jobs Dashboard (2025-10-23)

**Dashboard**: [Skypilot Jobs](https://app.datadoghq.com/dashboard/wjp-n4n-dsf/skypilot-jobs-dashboard)
**ID**: `wjp-n4n-dsf`
**Time Spent**: 1.5 hours (vs 4 hour estimate)
**Approach**: JSON-first (same as GitHub)

**What We Built**:
- **15 widgets** across 5 rows
- **4 query_value widgets**: Running Jobs, Queued Jobs, Failed (7d), Active Clusters
- **7 timeseries widgets**: Job status trends, Success rate, Runtime distribution, Runtime percentiles, GPU utilization, Spot vs On-demand, Regional distribution
- **2 additional query_value**: Total GPUs, Active Users
- **1 timeseries**: Jobs with Recoveries
- **1 note widget**: Recovery statistics explanation

**Key Features**:
- ✅ Runtime percentiles converted from seconds to hours (formula: `query / 3600`)
- ✅ Success rate calculated from formula: `100 * (succeeded / (succeeded + failed))`
- ✅ Alert threshold: p99 > 48h to detect stuck jobs
- ✅ Bar chart for runtime distribution buckets
- ✅ Area charts for GPU utilization and regional distribution
- ✅ Traffic light color palette consistent with GitHub dashboard

**Metrics Used** (30 total available, 24 deployed):
- Job status: `running`, `queued`, `failed`, `failed_7d`, `succeeded`, `cancelled`
- Runtime: `p50`, `p90`, `p99`, `min`, `max`, `avg` (all in seconds, converted to hours)
- Runtime buckets: `0_1h`, `1_4h`, `4_24h`, `over_24h`
- GPU counts: `l4_count`, `a10g_count`, `h100_count`, `total_count`
- Resources: `spot_jobs`, `ondemand_jobs`
- Reliability: `with_recoveries`
- Regional: `us_east_1`, `us_west_2`, `other`
- Team: `active_count`
- Clusters: `active`

**Not Yet Deployed**:
- Recovery count statistics (avg, max) - included in note widget for now

**Learnings**:
1. **Formula conversion** - Successfully converted seconds to hours using formula: `query / 3600`
2. **Bar charts** - Used `display_type: "bars"` for histogram visualization
3. **Area charts** - Used `display_type: "area"` for stacked resource views
4. **Even faster** - 1.5 hours vs 2 hours for GitHub (getting more efficient with JSON)

---

### ✅ GitHub CI/CD Dashboard (2025-10-23)

**Dashboard**: [GitHub CI/CD](https://app.datadoghq.com/dashboard/7gy-9ub-2sq/github-cicd-dashboard)
**ID**: `7gy-9ub-2sq`
**Time Spent**: 2 hours (vs 6 hour estimate)
**Approach**: JSON-first (not GUI as originally planned)

**What We Built**:
- **15 widgets** across 5 rows
- **4 query_value widgets**: Open PRs, PRs Merged (7d), Active Developers, Tests on Main
- **7 timeseries widgets**: PR cycle time, stale PRs, hotfixes, reverts, failed workflows, CI duration percentiles, workflow success rate
- **3 note widgets**: Coming soon placeholders for force merges and top contributors
- **1 calculated metric**: Success rate formula (100 * (total - failed) / total)

**Key Features**:
- ✅ Conditional formatting on Tests on Main (green/red traffic light)
- ✅ Threshold markers on all charts (targets and warnings)
- ✅ Multi-line timeseries for CI duration percentiles (p50, p90, p99)
- ✅ Traffic light color palette (red/yellow/green)

**Learnings**:
1. **JSON editing worked better than GUI** - Faster iteration, easier to duplicate widgets
2. **Note widgets**: Don't support `title`, `title_align`, or `title_size` properties (API validation error)
3. **Conditional formatting**: Required specific structure with `custom_fg_color` and `custom_bg_color`
4. **Template committed**: `devops/datadog/templates/github_cicd.json` in version control

**Metrics Used** (25 total available, 11 deployed):
- `github.prs.open`, `github.prs.merged_7d`, `github.prs.cycle_time_hours`, `github.prs.stale_count_14d`
- `github.developers.active_7d`
- `github.ci.tests_passing_on_main`, `github.ci.failed_workflows_7d`, `github.ci.workflow_runs_7d`
- `github.ci.duration_p50_minutes`, `github.ci.duration_p90_minutes`, `github.ci.duration_p99_minutes`
- `github.commits.hotfix`, `github.commits.reverts`, `github.commits.per_developer_7d`

**Not Yet Deployed**:
- Top contributors (needs collector extension to tag by author)
- Force merge tracking (metric not yet implemented)

---

## Implementation Strategy

### Approach: JSON First (Updated Based on Experience)

**Original Plan**: GUI first, JSON later
**Actual Approach**: JSON first - proved more efficient!

**Rationale for JSON-first**:
- ✅ **Faster iteration** - Edit JSON directly, push, review in browser
- ✅ **Easy duplication** - Copy/paste widget definitions
- ✅ **Version control from start** - All changes tracked in git
- ✅ **Consistent formatting** - Enforced structure
- ✅ **No manual export step** - JSON is the source of truth

**Updated Workflow**:
```bash
# 1. Create dashboard JSON template
cat > devops/datadog/templates/my_dashboard.json <<EOF
{
  "title": "My Dashboard",
  "widgets": [...]
}
EOF

# 2. Push to Datadog
uv run python devops/datadog/cli.py dashboard push devops/datadog/templates/my_dashboard.json

# 3. View in browser, iterate on JSON
# - Fix any validation errors
# - Adjust queries, formatting, layout

# 4. Commit final version
git add devops/datadog/templates/my_dashboard.json
git commit -m "feat: add my dashboard"

# 5. Future updates: Edit JSON, push, commit
```

**When to use GUI**:
- Exploring complex queries that need visual feedback
- Testing formulas and aggregations
- Quick prototyping before codifying in JSON

### Dashboard Creation Process

**For each dashboard:**

1. **Setup** (15 min)
   - Create new dashboard in Datadog
   - Set title and description
   - Configure time range defaults

2. **Build Layout** (2-4 hours)
   - Add widgets row by row
   - Configure queries for each widget
   - Set up conditional formatting
   - Add threshold markers

3. **Polish** (30 min)
   - Add notes/descriptions
   - Configure template variables
   - Set up links to other dashboards
   - Test on different screen sizes

4. **Export & Version Control** (15 min)
   - Export JSON
   - Commit to git with descriptive message
   - Document any manual steps

5. **Team Review** (30 min)
   - Demo to team
   - Gather feedback
   - Iterate if needed

**Total per dashboard**: 4-6 hours

---

## Phase 1: Foundation Dashboards (Week 1)

**Goal**: Build minimum viable dashboard suite with existing data
**Duration**: 5 days
**Prerequisites**: GitHub collector deployed ✅, Skypilot collector ready ✅

### Dashboard 1: GitHub CI/CD Dashboard ✅

**Priority**: P0 (Do this first!)
**Estimated Time**: 6 hours
**Actual Time**: 2 hours
**Status**: ✅ **Complete** (2025-10-23)
**Dashboard**: [a53-9nk-w6j](https://app.datadoghq.com/dashboard/a53-9nk-w6j/github-cicd-dashboard)

#### Why First?
- All 25 metrics already flowing from GitHub collector
- Most complete data set available
- High value for engineering team
- No blockers

#### Implementation Steps

**Day 1 Morning (3 hours):**

1. **Create Dashboard** (30 min)
   ```
   In Datadog UI:
   - New Dashboard → "GitHub CI/CD"
   - Layout: Grid
   - Time Range: Past 30 days
   ```

2. **Row 1: Current Status** (45 min)
   - Add 4 Query Value widgets:
     - Open PRs: `avg:github.prs.open{*}`
     - PRs Merged (7d): `sum:github.prs.merged_7d{*}`
     - Active Developers: `avg:github.developers.active_7d{*}`
     - Tests Passing: `avg:github.ci.tests_passing_on_main{*}`
       - Conditional: 1.0 = green "PASSING", 0.0 = red "FAILING"

3. **Row 2: PR Velocity** (45 min)
   - Left: Timeseries - PR Cycle Time
     - Query: `avg:github.prs.cycle_time_hours{*}`
     - Add horizontal marker at 48h (target)
   - Right: Timeseries - Stale PRs
     - Query: `avg:github.prs.stale_count_14d{*}`
     - Add horizontal marker at 20 (threshold)

4. **Row 3: Code Quality** (60 min)
   - 4 Timeseries widgets (2x2 grid):
     - Hotfixes: `sum:github.commits.hotfix{*}` (marker at 5)
     - Reverts: `sum:github.commits.reverts{*}` (marker at 1)
     - Failed Workflows: `sum:github.ci.failed_workflows_7d{*}` (marker at 2)
     - (Placeholder for Force Merges - coming in Phase 2)

**Day 1 Afternoon (3 hours):**

5. **Row 4: CI Performance** (90 min)
   - Left: Timeseries (multi-line) - CI Duration Percentiles
     - Query 1: `avg:github.ci.duration_p50_minutes{*}` (label: p50)
     - Query 2: `avg:github.ci.duration_p90_minutes{*}` (label: p90)
     - Query 3: `avg:github.ci.duration_p99_minutes{*}` (label: p99)
     - Add horizontal marker at 5 min (p90 target)
   - Right: Calculate success rate
     - Formula: `(total_runs - failed_runs) / total_runs * 100`
     - Display as percentage gauge

6. **Row 5: Developer Activity** (60 min)
   - Left: Timeseries - Commits per Developer
     - Query: `avg:github.commits.per_developer_7d{*}`
   - Right: Top List - Top Contributors
     - Query: `sum:github.commits.total_7d{*} by {author}`
     - (Note: May need to extend collector to tag by author)
     - Placeholder for now

7. **Polish & Export** (30 min)
   - Add dashboard description
   - Add note widgets with explanations
   - Test time range selector
   - Export: `metta datadog dashboard pull`
   - Commit: `git add devops/datadog/templates/github_cicd.json`

#### Success Criteria

- ✅ Dashboard displays all key GitHub metrics
- ✅ No query errors or missing data
- ✅ Conditional formatting works (green/yellow/red)
- ✅ Team can view and understand metrics
- ✅ JSON committed to version control

**All criteria met!** Dashboard successfully deployed.

#### What We Actually Did

**Different from Plan**:
- Used JSON editing instead of GUI (faster iteration)
- Built all widgets in JSON template first
- Pushed via CLI: `uv run python devops/datadog/cli.py dashboard push devops/datadog/templates/github_cicd.json`
- Fixed API validation errors (note widget properties)
- Took 2 hours instead of 6 hours (JSON approach more efficient)

**Issues Encountered & Resolved**:
1. ❌ Note widgets rejected with title properties
   - ✅ Fixed: Removed `title`, `title_align`, `title_size` from note definitions
   - ✅ Used markdown in content instead
2. ❌ Conditional formatting initially unclear
   - ✅ Fixed: Used correct structure with `custom_fg_color` and `custom_bg_color`

#### Blockers/Risks

- ~~**Low risk** - All metrics already available~~ ✅ Complete
- Minor: Top contributors by author needs collector extension
  - **Status**: Placeholder added, will extend collector in Phase 2

---

### Dashboard 2: Skypilot Jobs Dashboard ✅

**Priority**: P1
**Estimated Time**: 4 hours
**Actual Time**: 1.5 hours
**Status**: ✅ **Complete** (2025-10-23)
**Dashboard**: [ndg-4cn-2h2](https://app.datadoghq.com/dashboard/ndg-4cn-2h2/skypilot-jobs-dashboard)

#### Implementation Steps

**Day 2 Morning (2 hours):**

1. **Create Dashboard** (15 min)
   ```
   In Datadog UI:
   - New Dashboard → "Skypilot Jobs"
   - Layout: Grid
   - Time Range: Past 7 days
   ```

2. **Row 1: Job Status** (45 min)
   - Add 4 Query Value widgets:
     - Running Jobs: `avg:skypilot.jobs.running{*}`
     - Queued Jobs: `avg:skypilot.jobs.queued{*}`
     - Failed (7d): `sum:skypilot.jobs.failed_7d{*}`
     - Active Clusters: `avg:skypilot.clusters.active{*}`

3. **Row 2: Job Trends** (60 min)
   - Left: Stacked Area - Jobs Over Time
     - Metrics: running, queued, failed (stacked)
   - Right: Timeseries - Success Rate
     - Formula: `succeeded / (succeeded + failed) * 100`

**Day 2 Afternoon (2 hours):**

4. **Row 3: Cluster Health** (45 min)
   - Left: Timeseries - Active Clusters
     - Query: `avg:skypilot.clusters.active{*}`
   - Right: Placeholder for uptime (future metric)

5. **Row 4: Cost Analysis** (45 min)
   - Placeholder widgets (metrics not yet available)
   - Add notes: "Coming soon: cost tracking"

6. **Polish & Export** (30 min)
   - Add descriptions
   - Export and commit JSON

#### Success Criteria

- ✅ Dashboard shows all available Skypilot metrics
- ✅ Job status clearly visible
- ✅ Trends show historical data
- ✅ Placeholders documented for future metrics

---

### Dashboard 3: System Health Rollup (Partial)

**Priority**: P0
**Estimated Time**: 4 hours (partial implementation)
**Status**: Waiting for FoM collector
**Dependencies**: FoM collector with CI metrics only

#### Implementation Strategy

**Approach**: Build incrementally
1. Start with 7 CI FoM metrics (Phase 1)
2. Add training FoMs when WandB collector ready (Phase 2)
3. Add eval FoMs when eval collector ready (Phase 3)

#### Phase 1a: CI Metrics Only (This Week)

**Day 3 Morning (2 hours):**

1. **Create Dashboard** (15 min)
   ```
   In Datadog UI:
   - New Dashboard → "System Health Rollup"
   - Layout: Grid
   - Time Range: Past 7 days (fixed)
   - Description: "Executive dashboard - system health at a glance"
   ```

2. **Header: Health Scores** (30 min)
   - 3 Query Value widgets (big numbers):
     - CI/CD Health: Average of all CI FoM values
       - Query: `avg:health.ci.*.fom{*}`
       - Conditional: >0.7 green, 0.3-0.7 yellow, <0.3 red
     - Training Health: Placeholder "Coming Soon"
     - Eval Health: Placeholder "Coming Soon"

3. **Main Table: CI Metrics (7 rows)** (75 min)
   - Create Query Table widget
   - Add 7 queries (one per CI FoM metric):
     ```
     1. Tests Passing:     health.ci.tests_passing.fom
     2. Failing Workflows: health.ci.failing_workflows.fom
     3. Hotfix Count:      health.ci.hotfix_count.fom
     4. Revert Count:      health.ci.revert_count.fom
     5. CI Duration P90:   health.ci.duration_p90.fom
     6. Stale PRs:         health.ci.stale_prs.fom
     7. PR Cycle Time:     health.ci.pr_cycle_time.fom
     ```
   - Group by day (last 7 days)
   - Configure conditional formatting:
     - ≥ 0.7: Green background (#28A745)
     - ≥ 0.3: Yellow background (#FFC107)
     - < 0.3: Red background (#DC3545)
   - Set cell display to show FoM value

**Day 3 Afternoon (2 hours):**

4. **Add Placeholder Sections** (30 min)
   - Training section:
     - Gray boxes with "Waiting for WandB collector"
     - 9 placeholder rows
   - Eval section:
     - Gray boxes with "Waiting for Eval collector"
     - 5 placeholder rows

5. **Footer: Quick Links** (30 min)
   - Add Note widget with links:
     - → GitHub CI/CD Dashboard
     - → Skypilot Jobs Dashboard
     - → Collector Health Dashboard

6. **Polish & Export** (60 min)
   - Add dashboard description explaining FoM scale
   - Add legend note widget:
     ```
     FoM Scale:
     🟢 Green (0.7-1.0): Healthy
     🟡 Yellow (0.3-0.7): Warning
     🔴 Red (0.0-0.3): Critical
     ⚪ Gray: No data
     ```
   - Test that table refreshes correctly
   - Export and commit JSON

#### Success Criteria

- ✅ Dashboard loads with 7 CI metrics
- ✅ Color coding works correctly
- ✅ 7-day rolling window displays properly
- ✅ Placeholders clearly indicate future sections
- ✅ Links to other dashboards work

#### Blockers/Dependencies

- **BLOCKER**: Requires FoM collector to be deployed first
  - See [HEALTH_DASHBOARD_SPEC.md](HEALTH_DASHBOARD_SPEC.md) Phase 1
  - **Action**: Implement FoM collector for CI metrics (4 hours)
  - **Timeline**: Must complete before building this dashboard

---

## Phase 2: Training & Expanded Metrics (Week 2-3)

**Goal**: Add training visibility and complete CI coverage
**Duration**: 1-2 weeks
**Prerequisites**: WandB collector deployed, GitHub collector extended

### Task 1: Extend GitHub Collector (Day 4-5)

**Time**: 4 hours
**New Metrics Needed**:
- `github.ci.benchmarks_passing`
- `github.commits.force_merge_7d`
- `github.ci.timeout_cancellations_7d`
- `github.ci.flaky_checks_7d`
- `github.commits.total_7d by {author}` (for top contributors)

**Steps**:
1. Add metric functions to `collectors/github/metrics.py`
2. Test locally: `metta datadog collect github`
3. Deploy (automatic on next CronJob run)
4. Verify metrics flowing to Datadog

### Task 2: Implement WandB Collector (Day 6-10)

**Time**: 2-3 days
**See**: [ADDING_NEW_COLLECTOR.md](ADDING_NEW_COLLECTOR.md)
**See**: [WORKPLAN.md](../WORKPLAN.md) Section 4A

**Deliverables**:
- WandB collector deployed
- 15+ training metrics flowing
- Tests passing

### Task 3: Training & WandB Dashboard (Day 11-12)

**Priority**: P1
**Estimated Time**: 6 hours
**Dependencies**: WandB collector deployed

**Implementation** (similar to GitHub dashboard):
1. Create dashboard
2. Row 1: Training status (active, completed, failed, queued)
3. Row 2: Multigpu arena performance
4. Row 3: Multinode performance
5. Row 4: Resource utilization
6. Row 5: Jobs breakdown and recent runs
7. Export and commit

### Task 4: Update System Health Rollup (Day 13)

**Time**: 2 hours
**Add**: 9 training FoM metrics to rollup table

**Steps**:
1. Update FoM collector to include training metrics
2. Open rollup dashboard in UI
3. Replace placeholder rows with actual training FoM queries
4. Test color coding
5. Export and commit

### Task 5: Update GitHub Dashboard (Day 13)

**Time**: 1 hour
**Add**: 4 new GitHub metrics collected in Task 1

---

## Phase 3: Eval & Meta-Monitoring (Week 4-5)

**Goal**: Complete dashboard suite
**Duration**: 1 week
**Prerequisites**: Eval collector deployed

### Task 1: Implement Eval Collector (Day 14-18)

**Time**: 2-3 days
**See**: [ADDING_NEW_COLLECTOR.md](ADDING_NEW_COLLECTOR.md)

**Deliverables**:
- Eval collector deployed
- 5 eval metrics flowing

### Task 2: Eval & Testing Dashboard (Day 19)

**Priority**: P2
**Estimated Time**: 4 hours

**Implementation**:
1. Create dashboard
2. Row 1: Eval status (local/remote)
3. Row 2: Accuracy trends
4. Row 3: Performance metrics
5. Row 4: Recent evaluations table
6. Export and commit

### Task 3: Collector Health Dashboard (Day 20)

**Priority**: P2
**Estimated Time**: 3 hours
**Dependencies**: All collectors deployed

**Implementation**:
1. Create dashboard
2. Collector status table (all collectors)
3. Performance metrics (duration, errors)
4. Recent errors live feed
5. Export and commit

### Task 4: Final System Health Rollup Update (Day 20)

**Time**: 1 hour
**Add**: 5 eval FoM metrics

**Steps**:
1. Update FoM collector with eval metrics
2. Replace placeholder rows in rollup
3. Final polish and testing
4. Export and commit

---

## Dashboard Maintenance & Iteration

### Weekly Reviews (Ongoing)

**Every Friday** (30 min):
- Review dashboard usage in Datadog analytics
- Check for query errors or missing data
- Gather team feedback
- Document improvement ideas

### Monthly Updates (Ongoing)

**First Monday of month** (2 hours):
- Review and update threshold markers based on trends
- Add new metrics if available
- Remove metrics that aren't useful
- Update FoM formulas if needed

### Version Control Workflow

**For dashboard updates**:
```bash
# 1. Make changes in Datadog UI
# (easier than editing JSON)

# 2. Pull latest version
metta datadog dashboard pull

# 3. Review changes
git diff devops/datadog/templates/

# 4. Commit with descriptive message
git add devops/datadog/templates/*.json
git commit -m "feat(dashboards): add benchmark status to CI dashboard"

# 5. Push
git push
```

---

## Success Metrics

### Phase 1 Complete (End of Week 1)

- ✅ GitHub CI/CD dashboard deployed
- ✅ Skypilot Jobs dashboard deployed
- ✅ System Health Rollup deployed (CI metrics only)
- ✅ Team using dashboards daily
- ✅ All dashboards committed to git

**Measurement**:
- Dashboard views per day > 20
- Team mentions dashboards in standups
- No query errors in Datadog

### Phase 2 Complete (End of Week 3)

- ✅ Training & WandB dashboard deployed
- ✅ System Health Rollup updated with training FoMs
- ✅ GitHub dashboard shows all 29 metrics
- ✅ Training issues caught early via dashboard

**Measurement**:
- Training metrics visible and accurate
- Research team using training dashboard
- At least 1 issue detected via FoM alerts

### Phase 3 Complete (End of Week 5)

- ✅ All 6 dashboards deployed
- ✅ System Health Rollup complete (25 metrics)
- ✅ Collector Health monitoring in place
- ✅ Dashboard suite in daily use

**Measurement**:
- Full rollup table (no placeholders)
- All collectors monitored
- Dashboard suite standard in team workflow

---

## Risk Management

### Risk: FoM Collector Complexity

**Description**: FoM collector needs to query Datadog API for raw metrics

**Impact**: High - blocks System Health Rollup

**Mitigation**:
1. Build FoM collector in parallel with dashboards
2. Start with simple CI metrics (known to work)
3. Test extensively before deploying
4. Document query API usage clearly

**Owner**: DevOps

### Risk: WandB API Access

**Description**: May not have WandB API credentials or rate limits

**Impact**: Medium - blocks Training dashboard

**Mitigation**:
1. Check WandB access early (this week)
2. Request API key if needed
3. Understand rate limits
4. Have backup plan (manual metrics)

**Owner**: Research team lead

**Action Item**: Verify WandB access by end of Week 1

### Risk: Dashboard Complexity

**Description**: Dashboards may be too complex or slow

**Impact**: Low - poor UX but functional

**Mitigation**:
1. Start simple, add complexity iteratively
2. Test query performance
3. Use efficient aggregations
4. Monitor dashboard load times

**Owner**: Dashboard creator

### Risk: Changing Requirements

**Description**: Team may want different metrics or layouts

**Impact**: Low - rework but not blocked

**Mitigation**:
1. Get early feedback (after each dashboard)
2. Iterate quickly
3. Keep dashboards in version control (easy rollback)
4. Document design decisions

**Owner**: Team

---

## Open Questions & Decisions Needed

### Q1: Dashboard Ownership

**Question**: Who maintains each dashboard long-term?

**Options**:
- A) DevOps owns all dashboards
- B) Per-collector dashboards owned by team that owns that area
- C) Shared ownership with designated reviewers

**Recommendation**: Option B
- GitHub CI/CD → Engineering
- Training & WandB → Research
- Skypilot → DevOps
- Eval → QA/Research
- Collector Health → DevOps
- System Health Rollup → DevOps (synthesizes all)

**Decision Needed By**: End of Week 1

---

### Q2: Alert Configuration

**Question**: When to set up Datadog monitors/alerts for dashboard metrics?

**Options**:
- A) Now (Phase 1) - alongside dashboards
- B) After Phase 3 - once we understand baseline
- C) Per-dashboard as each is built

**Recommendation**: Option C
- Set up critical alerts as each dashboard is built
- Start conservative (only critical issues)
- Expand based on incidents

**Decision Needed By**: During Phase 1 implementation

---

### Q3: Multi-Environment Support

**Question**: Do we need separate dashboards for staging/production?

**Options**:
- A) Separate dashboards per environment
- B) One dashboard with template variables to filter
- C) Production only for now

**Recommendation**: Option C → Option B
- Start with production only (Phase 1-3)
- Add template variables in Phase 4
- Avoids duplication, easier to maintain

**Decision Needed By**: End of Phase 3

---

### Q4: Custom Time Ranges

**Question**: Should users be able to change time ranges on all dashboards?

**Options**:
- A) Fixed time ranges (consistent, predictable)
- B) User-selectable (flexible, personal preference)
- C) Fixed on Rollup, flexible on detail dashboards

**Recommendation**: Option C
- System Health Rollup: Fixed 7 days (consistency)
- Per-collector dashboards: User-selectable (flexibility)
- Default to sensible ranges (7-30 days)

**Decision**: Approved (apply during implementation)

---

## Resource Requirements

### People

- **Primary**: 1 engineer focused on dashboards (Week 1-5)
- **Supporting**:
  - DevOps for FoM collector implementation (4 hours)
  - Research for WandB collector review (2 hours)
  - Team for feedback sessions (30 min each)

### Tools

- **Datadog Account**: ✅ Have access
- **Dashboard Edit Permissions**: ✅ DevOps has access
- **Git Repository**: ✅ Ready
- **CLI Tools**: ✅ `metta datadog dashboard` commands ready

### Infrastructure

- **Collectors Running**:
  - ✅ GitHub collector deployed
  - ✅ Skypilot collector implemented
  - ⏳ FoM collector needed (Week 1)
  - ⏳ WandB collector needed (Week 2-3)
  - ⏳ Eval collector needed (Week 4-5)

---

## Next Steps

### Immediate Actions (This Week)

1. ~~**Review this workplan** with team~~ ✅ **Complete**
   - ~~Stakeholders: DevOps, Engineering, Research~~
   - ~~Format: 30-min meeting or async review~~
   - ~~Get buy-in on approach and timeline~~

2. **Verify WandB API access** ⏳ Next
   - Check if we have API keys
   - Understand rate limits
   - Identify potential blockers

3. **Implement FoM collector** (4 hours) ⏳ Needed for rollup
   - See [HEALTH_DASHBOARD_SPEC.md](HEALTH_DASHBOARD_SPEC.md) Phase 1
   - Start with 7 CI metrics
   - Deploy to production

4. ~~**Build GitHub CI/CD Dashboard** (6 hours)~~ ✅ **Complete** (2 hours)
   - ~~Follow steps in Phase 1~~
   - ✅ First complete dashboard deployed!
   - **Dashboard**: [a53-9nk-w6j](https://app.datadoghq.com/dashboard/a53-9nk-w6j/github-cicd-dashboard)

5. ~~**Build Skypilot Dashboard** (4 hours)~~ ✅ **Complete** (1.5 hours)
   - ~~Leverage existing collector (30 metrics available)~~
   - ✅ Second complete dashboard deployed!
   - **Dashboard**: [ndg-4cn-2h2](https://app.datadoghq.com/dashboard/ndg-4cn-2h2/skypilot-jobs-dashboard)

6. **Build partial System Health Rollup** (4 hours) ⏳ Blocked
   - Requires FoM collector (Task #3)
   - Start with CI FoMs only
   - Add placeholders for training/eval

### Week 2-3

7. Extend GitHub collector (4 hours)
8. Implement WandB collector (2-3 days)
9. Build Training dashboard (6 hours)
10. Update rollup with training FoMs (2 hours)

### Week 4-5

11. Implement Eval collector (2-3 days)
12. Build Eval dashboard (4 hours)
13. Build Collector Health dashboard (3 hours)
14. Complete rollup with eval FoMs (1 hour)

---

## Appendix: Quick Reference

### Dashboard URLs

```
✅ GitHub CI/CD:           https://app.datadoghq.com/dashboard/a53-9nk-w6j/github-cicd-dashboard
✅ Skypilot Jobs:          https://app.datadoghq.com/dashboard/ndg-4cn-2h2/skypilot-jobs-dashboard
   System Health Rollup:   (not yet created)
   Training & WandB:       (not yet created)
   Eval & Testing:         (not yet created)
   Collector Health:       (not yet created)
```

### Key Commands

```bash
# List dashboards
metta datadog dashboard list

# Pull all dashboards (export JSON)
metta datadog dashboard pull

# Pull specific dashboard
metta datadog dashboard export <dashboard-id>

# Push dashboards (from JSON)
metta datadog dashboard push

# Show git diff of dashboard changes
metta datadog dashboard diff
```

### File Locations

```
Dashboard Designs:      devops/datadog/docs/DASHBOARD_DESIGN.md
Dashboard JSONs:        devops/datadog/templates/*.json
FoM Spec:              devops/datadog/docs/HEALTH_DASHBOARD_SPEC.md
Collector Arch:        devops/datadog/docs/COLLECTORS_ARCHITECTURE.md
Main Workplan:         devops/datadog/WORKPLAN.md
```

---

**Last Updated**: 2025-10-23
**Owner**: DevOps team
**Status**: Ready to start Phase 1
**Next Review**: After Phase 1 completion
