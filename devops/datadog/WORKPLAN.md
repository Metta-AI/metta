# Datadog Observability System - Work Plan

**Current Status**: Phase 3B Complete - GitHub collector deployed and operational
**Branch**: `robb/1022-datadog`
**PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

---

## What We Built (Phases 1-3B Complete)

### ✅ Modular Collector Architecture
- **BaseCollector pattern** for consistent collector implementation
- **GitHub collector** deployed to production (Helm revision 16, monitoring namespace)
- **25 metrics** collected every 15 minutes via CronJob
- **Metrics flowing successfully** to Datadog for 24+ hours
- Framework ready for additional collectors (WandB, EC2, Skypilot, Asana)

### ✅ Deployment Infrastructure
- Helm chart for CronJob deployment (`devops/charts/dashboard-cronjob/`)
- AWS Secrets Manager integration for credentials
- IAM Roles for Service Accounts (IRSA) with proper trust policies
- Kubernetes deployment scripts and helpers

### ✅ Dashboard Management System
- Jsonnet-based dashboard configuration
- Modular widget library for reusable components
- Python scripts for dashboard operations (push, fetch, export, batch)
- Template dashboards ready to customize

### ✅ Comprehensive Documentation
- **Quick Start Guide** - Getting started in 10 minutes
- **Collectors Architecture** - System design and patterns
- **Adding New Collectors** - Step-by-step implementation guide
- **CI/CD Metrics Catalog** - Complete list of 25 GitHub metrics
- **Jsonnet Design** - Dashboard configuration patterns
- **Widget Reference** - Datadog widget types and usage

### ✅ Production Metrics (GitHub Collector)

**Pull Requests**: open, merged_7d, closed_without_merge_7d, avg_time_to_merge_hours, stale_count_14d, cycle_time_hours

**Commits**: total_7d, hotfix, reverts, per_developer_7d

**Branches**: active

**CI/CD**: tests_passing_on_main, workflow_runs_7d, failed_workflows_7d, avg_workflow_duration_minutes, duration_p50_minutes, duration_p90_minutes, duration_p99_minutes

**Developers**: active_7d

**Code Changes**: lines_added_7d, lines_deleted_7d, files_changed_7d

**Pull Request Quality**: with_review_comments_pct, avg_comments_per_pr

---

## Immediate Priorities (This Week)

### 1. Clean Up Old Code (Phase 3C) - 2 hours

**Goal**: Remove legacy code from `softmax/src/softmax/dashboard/`

**Files to Remove**:
```bash
softmax/src/softmax/dashboard/
├── metrics.py          # Old GitHub collector (27KB)
├── registry.py         # Old metric registry
├── report.py           # Old CLI entry point
├── __init__.py         # Empty init
└── README.md           # Old docs
```

**Steps**:
1. Verify new collector running 24+ hours without issues
2. Run `metta pytest` to ensure tests pass
3. Remove `softmax/src/softmax/dashboard/` directory
4. Search for any imports: `grep -r "softmax.dashboard" .`
5. Update any references in code/docs
6. Verify `metta pytest` still passes
7. Commit: "chore: remove legacy dashboard code"

**Dependencies**: ✅ New collector verified (deployed 2025-10-23)

---

### 2. Finalize Current PR (Phase 4B-4C) - 4 hours

**Goal**: Get current work merged to main

**Tasks**:
- [x] Documentation review (completed)
- [x] Run full test suite: `metta pytest` (all 1211 tests passing)
- [x] Update PR description with Phase 3B completion
- [ ] Request re-review from @nishu-builder
- [ ] Address any new feedback
- [ ] Merge to main

**Deliverable**: GitHub collector in production, foundation for additional collectors

---

## Short-term Goals (Next 2 Weeks)

### 3. Build Beautiful Dashboards - 8 hours

**Current State**:
```
devops/datadog/templates/
├── demo.json                      # 31KB demo dashboard
├── softmax_system_health.json     # 4.7KB basic health
├── softmax_pulse.json             # 450B minimal
└── policy_evaluator.json          # 4.5KB policy metrics
```

**Goal**: Create polished, useful dashboards using GitHub metrics

**Priority Dashboards**:

#### A. Engineering Velocity Dashboard (High Priority)
**Metrics**:
- PR throughput: `prs.merged_7d`, `prs.open`, `prs.stale_count_14d`
- Cycle time: `prs.cycle_time_hours`, `prs.avg_time_to_merge_hours`
- Developer activity: `developers.active_7d`, `commits.per_developer_7d`
- Branch health: `branches.active`

**Layout**:
```
┌──────────────────────────────────────────────────────┐
│ Engineering Velocity                          [Filters]│
├────────────────┬────────────────┬────────────────────┤
│ PRs Merged/7d  │ Open PRs       │ Stale PRs (>14d)   │
│ [Big Number]   │ [Big Number]   │ [Big Number]       │
├────────────────┴────────────────┴────────────────────┤
│ PR Cycle Time (hours)                                 │
│ [Timeseries: avg, p50, p90]                          │
├──────────────────────────────────────────────────────┤
│ Active Developers          │ Commit Rate             │
│ [Timeseries]              │ [Timeseries]            │
└────────────────────────────┴─────────────────────────┘
```

#### B. Code Quality Dashboard (High Priority)
**Metrics**:
- Quality signals: `commits.reverts`, `commits.hotfix`
- CI health: `ci.tests_passing_on_main`, `ci.failed_workflows_7d`
- CI performance: `ci.duration_p50_minutes`, `ci.duration_p90_minutes`, `ci.duration_p99_minutes`
- Review engagement: `prs.with_review_comments_pct`, `prs.avg_comments_per_pr`

**Layout**:
```
┌──────────────────────────────────────────────────────┐
│ Code Quality                              [Filters]   │
├────────────────┬────────────────┬────────────────────┤
│ Tests Passing  │ Reverts/7d     │ Hotfixes/7d        │
│ [Status]       │ [Big Number]   │ [Big Number]       │
├────────────────┴────────────────┴────────────────────┤
│ CI Duration Percentiles (minutes)                     │
│ [Timeseries: p50, p90, p99 with threshold markers]   │
├──────────────────────────────────────────────────────┤
│ Failed Workflows           │ Workflow Success Rate   │
│ [Timeseries]              │ [Gauge/Percentage]      │
└────────────────────────────┴─────────────────────────┘
```

#### C. System Health Overview (Medium Priority)
**Combines**: Engineering velocity + Quality + (future: infrastructure metrics)

**Implementation Steps**:
1. Export current dashboards: `metta datadog dashboard pull`
2. Design layouts in Figma/paper
3. Build incrementally:
   - Start with velocity dashboard
   - Add quality dashboard
   - Create overview combining both
4. Use Datadog's dashboard JSON API
5. Add threshold markers for SLOs
6. Test query performance
7. Get team feedback
8. Iterate

**References**:
- [Datadog Dashboard JSON](https://docs.datadoghq.com/dashboards/graphing_json/)
- [Widget Reference](devops/datadog/docs/DATADOG_WIDGET_REFERENCE.md)
- [Metric Catalog](devops/datadog/docs/CI_CD_METRICS.md)

---

## Medium-term Goals (Next Month)

### 4. Implement Additional Collectors - 3-4 days each

**Priority Order**:

#### A. WandB Collector (Highest Priority) - 3 days
**Why First**: Training run metrics are core to ML research

**Planned Metrics**:
```python
# Training Progress
wandb.runs.active                    # Currently running experiments
wandb.runs.completed_7d              # Completed runs in last week
wandb.runs.failed_7d                 # Failed runs in last week

# Model Performance
wandb.metrics.best_accuracy          # Best model accuracy across runs
wandb.metrics.latest_loss            # Latest training loss
wandb.training.gpu_utilization_pct   # GPU usage

# Resource Usage
wandb.training.duration_hours        # Training time per run
wandb.training.cost_estimate_usd     # Estimated cloud cost
```

**Implementation**:
1. Create `collectors/wandb/collector.py` (BaseCollector)
2. WandB API authentication via API key in Secrets Manager
3. Test locally: `metta datadog collect wandb --dry-run`
4. Deploy CronJob
5. Create WandB dashboard

**References**: `collectors/wandb/README.md`

#### B. EC2 Collector (High Priority) - 2 days
**Why**: Track compute costs and utilization

**Planned Metrics**:
```python
# Instance Metrics
ec2.instances.running               # Running instances
ec2.instances.stopped               # Stopped instances
ec2.costs.monthly_estimate_usd      # Cost projection

# GPU Instances
ec2.gpu.instances.running           # GPU instances
ec2.gpu.utilization_pct             # GPU usage
ec2.gpu.costs.daily_usd             # Daily GPU costs
```

#### C. Skypilot Collector (Medium Priority) - 2 days
**Why**: Training job orchestration metrics

**Planned Metrics**:
```python
# Job Status
skypilot.jobs.queued                # Waiting jobs
skypilot.jobs.running               # Active jobs
skypilot.jobs.failed_7d             # Failed jobs

# Resource Utilization
skypilot.cluster.nodes_active       # Active cluster nodes
skypilot.cluster.utilization_pct    # Cluster usage
```

#### D. Asana Collector (Lower Priority) - 2 days
**Why**: Project management insights

**Planned Metrics**:
```python
# Task Metrics
asana.tasks.open                    # Open tasks
asana.tasks.completed_7d            # Completed tasks
asana.tasks.overdue                 # Overdue tasks

# Sprint Metrics
asana.sprint.completion_rate_pct    # Sprint completion %
asana.sprint.velocity               # Story points/sprint
```

**Decision Point**: May skip Asana if team doesn't actively use it for tracking

---

### 5. Advanced Dashboard Features - 2 days

**Enhancements**:

#### A. Template Variables
Add filters to dashboards:
```json
"template_variables": [
  {
    "name": "repo",
    "prefix": "repo",
    "default": "metta"
  },
  {
    "name": "environment",
    "prefix": "env",
    "default": "production"
  }
]
```

#### B. SLO Tracking
Define and track Service Level Objectives:
- **CI Success Rate**: ≥ 95%
- **PR Cycle Time**: ≤ 48 hours (p90)
- **Tests Passing**: 100% on main
- **Stale PRs**: ≤ 20 PRs > 14 days old

#### C. Alerting
Set up Datadog monitors:
- Alert when CI fails on main
- Alert when PR cycle time > 72 hours
- Alert when stale PR count > 50
- Alert when hotfix/revert rate spikes

#### D. Anomaly Detection
Use Datadog's built-in anomaly detection:
- Unusual spike in failed CI runs
- Sudden drop in active developers
- PR merge rate anomalies

---

## Long-term Vision (Next Quarter)

### 6. Jsonnet Dashboard System (Optional) - 1 week

**Goal**: Composable, version-controlled dashboards

**Current Approach**: JSON files in `templates/`
**Proposed**: Jsonnet system (like Grafana's Grafonnet)

**Benefits**:
- Reusable widget components
- Mix-and-match widgets across dashboards
- Grid layouts with automatic positioning
- Version control for dashboard changes

**Files**:
- `devops/datadog/docs/JSONNET_DESIGN.md` - Design doc
- `devops/datadog/docs/JSONNET_PROTOTYPE.md` - Implementation guide

**Decision**: Defer until we have 5+ dashboards and see pain points

---

### 7. Multi-Environment Support - 2 days

**Goal**: Separate staging/production metrics

**Implementation**:
1. Add `environment` tag to all metrics
2. Deploy separate CronJobs for staging/production
3. Use template variables to filter dashboards
4. Different alert thresholds per environment

**Configuration**:
```yaml
# Helm values - production
datadog:
  env: "production"

# Helm values - staging
datadog:
  env: "staging"
```

---

### 8. Cross-Service Dashboards

**Goal**: Unified view across multiple data sources

**Example Dashboard**: "Training Pipeline Health"
- GitHub metrics (commits, PRs)
- WandB metrics (training runs, model performance)
- EC2 metrics (GPU utilization, costs)
- Skypilot metrics (job status)

**Layout**:
```
┌──────────────────────────────────────────────────────┐
│ Training Pipeline Health                              │
├────────────────────────────────────────────────────────┤
│ Code Activity          │ Training Runs                │
│ (GitHub)              │ (WandB)                      │
├────────────────────────┼──────────────────────────────┤
│ GPU Utilization       │ Daily Costs                  │
│ (EC2)                 │ (EC2 + Skypilot)             │
└────────────────────────┴──────────────────────────────┘
```

---

## Priority Matrix

| Task | Impact | Effort | Priority | Timeline |
|------|--------|--------|----------|----------|
| Clean up old code | Low | 2h | High | This week |
| Merge current PR | High | 4h | High | This week |
| Velocity dashboard | High | 4h | High | Next week |
| Quality dashboard | High | 4h | High | Next week |
| WandB collector | High | 3d | Medium | 2 weeks |
| EC2 collector | High | 2d | Medium | 2 weeks |
| SLO tracking | Medium | 1d | Medium | 3 weeks |
| Alerting setup | Medium | 1d | Medium | 3 weeks |
| Skypilot collector | Medium | 2d | Low | 1 month |
| Asana collector | Low | 2d | Low | TBD |
| Jsonnet system | Low | 1w | Low | TBD |

---

## Success Metrics

**Week 1** (Current PR merged):
- ✅ GitHub collector in production
- ✅ Clean codebase (old code removed)
- ✅ Foundation for additional collectors

**Week 2** (Dashboards):
- 📊 2 polished dashboards (Velocity + Quality)
- 📈 Team actively using dashboards
- 🎯 Clear SLO definitions

**Month 1** (More collectors):
- 🔌 2-3 additional collectors deployed (WandB, EC2)
- 🎨 5+ production dashboards
- 🔔 Basic alerting configured
- 📊 Cross-service insights

**Quarter 1** (Mature system):
- 🔌 All planned collectors deployed
- 🎨 Comprehensive dashboard suite
- 🤖 Automated anomaly detection
- 📊 Regular metric reviews in team meetings

---

## Open Questions

1. **Dashboard tool**: Stick with JSON or invest in Jsonnet system?
   - **Decision**: Start with JSON, migrate if we hit >5 dashboards

2. **Asana collector**: Is team using Asana enough to justify?
   - **Decision**: TBD - need to assess actual Asana usage

3. **Multi-repo support**: Just Metta or expand to other repos?
   - **Decision**: Start with Metta, expand if successful

4. **Alert fatigue**: How to balance alerting vs noise?
   - **Decision**: Start conservative, add alerts based on actual incidents

---

## Resources

**Documentation**:
- [Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
- [Metric Conventions](docs/METRIC_CONVENTIONS.md)
- [CI/CD Metrics](docs/CI_CD_METRICS.md)

**Tools**:
- CLI: `metta datadog --help`
- Dashboard JSON: [Datadog Docs](https://docs.datadoghq.com/dashboards/graphing_json/)

**Team**:
- Owner: DevOps team
- Stakeholders: Engineering, Research, SRE

---

**Last Updated**: 2025-10-23
**Next Review**: After current PR merge
