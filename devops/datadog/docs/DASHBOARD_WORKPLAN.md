# Dashboard Implementation Workplan

## Overview

Concrete implementation plan for building all Datadog dashboards in the Metta observability system. This workplan focuses exclusively on dashboard creation, assuming collectors are implemented per the main [WORKPLAN.md](../WORKPLAN.md).

**Status**: Planning phase
**Last Updated**: 2025-10-23
**Related Docs**: [DASHBOARD_DESIGN.md](DASHBOARD_DESIGN.md), [HEALTH_DASHBOARD_SPEC.md](HEALTH_DASHBOARD_SPEC.md)

---

## Dashboard Inventory

| Dashboard | Collector Dependency | Status | Priority | Est. Time |
|-----------|---------------------|---------|----------|-----------|
| System Health Rollup | FoM collector | Not started | P0 | 8 hours |
| GitHub CI/CD | GitHub collector | ✅ Ready | P0 | 6 hours |
| Skypilot Jobs | Skypilot collector | ✅ Ready | P1 | 4 hours |
| Training & WandB | WandB collector | Blocked | P1 | 6 hours |
| Eval & Testing | Eval collector | Blocked | P2 | 4 hours |
| Collector Health | All collectors | Not started | P2 | 3 hours |

**Total Effort**: ~31 hours (4-5 days of focused work)

---

## Implementation Strategy

### Approach: GUI First, JSON Later

**Rationale**:
- Datadog UI is faster for prototyping
- Visual feedback while building
- Easier to iterate on design
- Export to JSON for version control

**Workflow**:
```bash
1. Build dashboard in Datadog UI
2. Test queries and visualizations
3. Get team feedback
4. Export: metta datadog dashboard pull
5. Commit JSON to git
6. Future updates via JSON editing
```

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

### Dashboard 1: GitHub CI/CD Dashboard

**Priority**: P0 (Do this first!)
**Estimated Time**: 6 hours
**Status**: Ready to build (all metrics available)

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

#### Blockers/Risks

- **Low risk** - All metrics already available
- Minor: Top contributors by author needs collector extension
  - **Mitigation**: Use placeholder widget, extend collector in Phase 2

---

### Dashboard 2: Skypilot Jobs Dashboard

**Priority**: P1
**Estimated Time**: 4 hours
**Status**: Ready to build (collector implemented yesterday)
**Dependencies**: Skypilot collector deployed

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

1. **Review this workplan** with team
   - Stakeholders: DevOps, Engineering, Research
   - Format: 30-min meeting or async review
   - Get buy-in on approach and timeline

2. **Verify WandB API access**
   - Check if we have API keys
   - Understand rate limits
   - Identify potential blockers

3. **Implement FoM collector** (4 hours)
   - See [HEALTH_DASHBOARD_SPEC.md](HEALTH_DASHBOARD_SPEC.md) Phase 1
   - Start with 7 CI metrics
   - Deploy to production

4. **Build GitHub CI/CD Dashboard** (6 hours)
   - Follow steps in Phase 1
   - First complete dashboard!

5. **Build Skypilot Dashboard** (4 hours)
   - Leverage existing collector
   - Second complete dashboard

6. **Build partial System Health Rollup** (4 hours)
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

### Dashboard URLs (After Creation)

```
System Health Rollup:   https://app.datadoghq.com/dashboard/system-health-rollup
GitHub CI/CD:           https://app.datadoghq.com/dashboard/github-cicd
Training & WandB:       https://app.datadoghq.com/dashboard/training-wandb
Skypilot Jobs:          https://app.datadoghq.com/dashboard/skypilot-jobs
Eval & Testing:         https://app.datadoghq.com/dashboard/eval-testing
Collector Health:       https://app.datadoghq.com/dashboard/collector-health
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
