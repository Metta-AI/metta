# Datadog Observability System - Work Plan

**Current Status**: Phase 6 Complete - Production deployment with 7 collectors, 8 dashboards, and 123+ metrics

**Branch**: `robb/1022-datadog` **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384) **Last Updated**: 2025-10-24

---

## ✅ What We Built (Phases 1-6 Complete)

### 🔌 Seven Collectors Deployed (123+ metrics)

All collectors running in production via unified CronJob (every 15 minutes):

#### 1. GitHub Collector (28 metrics)

**Pull Requests**: open, merged_7d, closed_without_merge_7d, avg_time_to_merge_hours, stale_count_14d, cycle_time_hours, with_review_comments_pct, avg_comments_per_pr

**Commits**: total_7d, hotfix, reverts, per_developer_7d, force_merge_7d

**Branches**: active

**CI/CD**: tests_passing_on_main, benchmarks_passing, workflow_runs_7d, failed_workflows_7d, avg_workflow_duration_minutes, duration_p50_minutes, duration_p90_minutes, duration_p99_minutes, timeout_cancellations_7d, flaky_checks_7d

**Code Changes**: lines_added_7d, lines_deleted_7d, files_changed_7d

**Developers**: active_7d

#### 2. Skypilot Collector (30 metrics)

**Job Status**: queued, running, failed, failed_7d, succeeded, cancelled

**Runtime Distribution**: min, max, avg, p50, p90, p99 (for running jobs)

**Runtime Buckets**: 0-1h, 1-4h, 4-24h, over_24h

**Resource Utilization**: L4/A10G/H100 GPU counts, total GPUs, spot/on-demand jobs

**Reliability**: jobs with recoveries, recovery count (avg/max)

**Regional Distribution**: us-east-1, us-west-2, other

**Team Activity**: active user count

**Cluster Health**: active clusters

**Key Insights**: Runtime p99 = 14 days, 85% spot usage, 184 L4 GPUs across 24 users

#### 3. Asana Collector (14 metrics)

**Workspace Tasks**: total, open, completed_7d, completed_30d, overdue, due_today, due_this_week, no_due_date, unassigned, assigned

**Velocity**: completed_per_day_7d, completion_rate_pct

**Cycle Time**: avg_hours, p50_hours, p90_hours

**Team Activity**: active_7d

**Bugs Project**: triage_count, active_count, backlog_count, other_count, total_open, completed_7d, completed_30d, created_7d, avg_age_days, oldest_bug_days

**Features**: Section-based workflow tracking, aging statistics, velocity tracking

#### 4. EC2 Collector (19 metrics)

**Instances**: total, running, stopped, spot, on-demand, gpu_count, cpu_count, idle, avg_age_days, oldest_age_days

**EBS Volumes**: total, attached, unattached, size_gb

**EBS Snapshots**: total, size_gb

**Cost Estimates**: running_hourly_estimate, monthly_estimate, spot_savings_pct

#### 5. WandB Collector (10 metrics)

**Training Progress**: runs.active, runs.completed_7d, runs.failed_7d

**Model Performance**: metrics.best_accuracy, metrics.latest_loss

**Resource Usage**: training.duration_hours, training.cost_estimate_usd, training.gpu_utilization_pct, training.gpu_hours_7d, training.runs_per_user

#### 6. Kubernetes Collector (15 metrics)

**Resource Efficiency**: pods.total, pods.running, pods.pending, pods.failed, deployments.total, deployments.available, deployments.unavailable

**Pod Health**: restarts_total, restarts_7d, oomkilled_7d

**Resource Waste Tracking**: cpu_request_waste_cores, memory_request_waste_gb, gpu_allocation_efficiency_pct

**Node Health**: nodes.ready, nodes.not_ready

#### 7. Health FoM Collector (7 metrics)

**CI/CD Figure of Merit** (0.0-1.0 scale):

- health.ci.tests_passing.fom
- health.ci.benchmarks_passing.fom
- health.ci.workflow_success_rate.fom
- health.ci.duration.fom
- health.commits.quality.fom (reverts + hotfixes)
- health.prs.velocity.fom
- health.prs.quality.fom (review coverage)

**Features**: Reads raw metrics from Datadog, applies normalization formulas, emits 0.0-1.0 health scores

**Color Scale**: 1.0=green, 0.7-1.0=good, 0.3-0.7=warning, 0.0-0.3=critical

---

### 📊 Eight Production Dashboards

```
devops/datadog/templates/
├── github_cicd.json                      # 15KB - CI/CD metrics and workflow health
├── skypilot_jobs.json                    # 18KB - Job tracking and GPU utilization
├── asana.json                            # 17KB - Project tracking and bug workflow
├── ec2.json                              # 25KB - Infrastructure monitoring and costs
├── system_health_rollup.json             # 67KB - 7×7 FoM grid (65 widgets)
├── system_health_rollup_wildcard.json    # 20KB - 7×7 FoM grid (Vega-Lite)
├── policy_evaluator.json                 # 4.2KB - Policy evaluation metrics
└── demo.json                             # 29KB - Demo/reference dashboard
```

**Dashboard Features**:

- **System Health Rollup**: 7×7 grid showing 7 FoM metrics over 7 days with color coding
- **Wildcard Widget**: Vega-Lite version with interactive tooltips and custom visualizations
- **Collector-Specific**: Dedicated dashboards for GitHub, Skypilot, Asana, EC2
- **All deployed to Datadog**: Live and accessible

---

### 🏗️ Production Infrastructure

#### Deployment System

- **Helm Chart**: `devops/charts/dashboard-cronjob/` for unified CronJob deployment
- **Single CronJob**: Runs all 7 collectors sequentially every 15 minutes
- **Docker Image**: `softmax-dashboard:latest` with all collector dependencies
- **Kubernetes Namespace**: `monitoring`
- **Current Revision**: Helm revision 16+
- **Uptime**: 24+ hours of successful metric collection

#### Security & Credentials

- **AWS Secrets Manager**: All API keys and tokens stored securely
- **IAM Roles for Service Accounts** (IRSA): Proper trust policies configured
- **Kubernetes RBAC**: Service account with read permissions for pods/deployments
- **GitHub Token**: Stored in AWS Secrets Manager (`github/dashboard-token`)
- **Datadog Keys**: API and App keys in Secrets Manager
- **Asana Token**: Personal access token in Secrets Manager

#### CI/CD Pipeline

- **Automatic Builds**: Docker image built on push to main
- **Automatic Deployment**: Helm chart deployed to Kubernetes
- **Health Checks**: Metrics verification post-deployment
- **Rollback Capability**: Helm revision history maintained

#### Dashboard Management Tools

- **Python Scripts** (`devops/datadog/scripts/`):
  - `push_dashboard.py` - Deploy dashboards to Datadog
  - `fetch_dashboards.py` - Download dashboards from Datadog
  - `export_dashboard.py` - Export specific dashboard by ID
  - `batch_export.py` - Export multiple dashboards
  - `delete_dashboard.py` - Remove dashboard from Datadog
  - `list_metrics.py` - List all available metrics
  - `generate_health_grid.py` - Generate 65-widget grid dashboard
  - `generate_wildcard_fom_grid.py` - Generate Vega-Lite heatmap

- **CLI**: `metta datadog` command for all operations

---

### 📚 Comprehensive Documentation

**Implementation Guides**:

- `docs/QUICK_START.md` - Getting started in 10 minutes
- `docs/COLLECTORS_ARCHITECTURE.md` - System design and patterns
- `docs/ADDING_NEW_COLLECTOR.md` - Step-by-step implementation guide
- `docs/DEPLOYMENT_GUIDE.md` - Production deployment procedures
- `devops/k8s/BUILD_AND_DEPLOY.md` - Kubernetes deployment details

**Reference Documentation**:

- `docs/CI_CD_METRICS.md` - Complete GitHub metrics catalog
- `docs/METRIC_CONVENTIONS.md` - Naming and tagging standards
- `docs/DATADOG_WIDGET_REFERENCE.md` - Widget types and usage
- `docs/WILDCARD_WIDGET.md` - Vega-Lite custom visualizations
- `docs/DASHBOARD_WORKPLAN.md` - Dashboard design evolution
- `docs/HEALTH_DASHBOARD_SPEC.md` - FoM grid specifications

**Design Documentation**:

- `docs/JSONNET_DESIGN.md` - Jsonnet system design (future)
- `docs/JSONNET_PROTOTYPE.md` - Jsonnet implementation guide (future)
- `docs/HELM_CRONJOB_CONVENTIONS.md` - Helm chart patterns
- `docs/IMAGE_COLLECTOR_PLAN.md` - Rejected matplotlib approach

**Secrets Management**:

- `docs/SECRETS_SETUP.md` - AWS Secrets Manager setup guide
- `.env.sample` - Example environment configuration

---

### 🧹 Legacy Cleanup (Completed 2025-10-24)

- ✅ Removed `softmax/src/softmax/dashboard/` directory (5 files)
- ✅ Removed CLI registration for `metta softmax-system-health`
- ✅ Removed legacy dashboard templates (`softmax_pulse.json`, `softmax_system_health.json`)
- ✅ Updated WORKPLAN.md with current dashboard inventory
- ✅ All 24 old metrics verified migrated to new GitHub collector
- ✅ Production using new system successfully for 24+ hours

---

## 🎯 Immediate Priorities (This Week)

### 1. ✅ Clean Up Old Code - COMPLETED

**Status**: All legacy code removed, production verified (2025-10-24)

---

### 2. Finalize Current PR - 2 hours

**Goal**: Get current work merged to main

**Tasks**:

- [x] Documentation review
- [x] Run full test suite: `metta pytest` (all tests passing)
- [x] Update PR description
- [x] Clean up legacy code
- [ ] Request re-review from @nishu-builder
- [ ] Address any new feedback
- [ ] Merge to main

**Deliverable**: All 7 collectors in production, 8 dashboards deployed, clean codebase

---

### 3. Monitor Production Deployment - Ongoing

**Goal**: Ensure stable metric collection and dashboard availability

**Monitoring**:

- Check CronJob execution logs
- Verify metrics appearing in Datadog
- Monitor for failures or gaps in data
- Track resource usage (CPU/memory)

**Success Criteria**:

- 95%+ successful collector runs
- No metric gaps > 30 minutes
- All dashboards loading correctly

---

## 🚀 Short-term Goals (Next 2-4 Weeks)

### 4. Advanced Dashboard Features - 3 days

#### A. SLO Tracking

Define and track Service Level Objectives:

- **CI Success Rate**: ≥ 95%
- **PR Cycle Time**: ≤ 48 hours (p90)
- **Tests Passing on Main**: 100%
- **Stale PRs**: ≤ 20 PRs > 14 days old
- **Deployment Frequency**: Track via CI metrics

**Implementation**: Add SLO threshold markers to existing dashboards

#### B. Template Variables

Add filters to dashboards for better navigation:

```json
"template_variables": [
  {
    "name": "collector",
    "prefix": "source",
    "default": "github-collector"
  },
  {
    "name": "environment",
    "prefix": "env",
    "default": "production"
  }
]
```

#### C. Conditional Formatting

Add color-coded indicators based on thresholds:

- Green: Metric within SLO
- Yellow: Approaching threshold
- Red: SLO violation

---

### 5. Alerting Setup - 2 days

**Goal**: Proactive notification of system issues

**Critical Alerts**:

1. **CI Failure on Main**
   - Trigger: `ci.tests_passing_on_main == 0`
   - Severity: High
   - Notification: Slack + PagerDuty

2. **Collector Failure**
   - Trigger: No metrics from collector for 30+ minutes
   - Severity: Medium
   - Notification: Slack

3. **PR Cycle Time SLO Violation**
   - Trigger: `prs.cycle_time_hours.p90 > 72`
   - Severity: Low
   - Notification: Slack

4. **EC2 Cost Spike**
   - Trigger: `ec2.cost.monthly_estimate` > 20% increase
   - Severity: Medium
   - Notification: Slack + Email

**Implementation**:

- Use Datadog Monitors API
- Configure notification channels
- Set up escalation policies
- Test alert delivery

---

### 6. Anomaly Detection - 1 day

**Goal**: Automatic detection of unusual patterns

**Metrics to Monitor**:

- Sudden spike in failed CI runs
- Sudden drop in active developers
- PR merge rate anomalies
- Unusual EC2 cost patterns
- GPU utilization spikes/drops

**Implementation**:

- Enable Datadog's built-in anomaly detection
- Configure sensitivity levels
- Set up anomaly alerts
- Review anomaly detections weekly

---

## 📈 Medium-term Goals (Next Quarter)

### 7. Multi-Environment Support - 2 days

**Goal**: Separate staging/production metrics and dashboards

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
  service: "datadog-collectors"

# Helm values - staging
datadog:
  env: "staging"
  service: "datadog-collectors-staging"
```

**Benefits**:

- Isolate staging noise from production metrics
- Test new collectors in staging first
- Environment-specific SLOs and alerts

---

### 8. Cross-Service Correlation Dashboards - 3 days

**Goal**: Unified view across multiple data sources

**Example Dashboard**: "Training Pipeline Health"

```
┌──────────────────────────────────────────────────────┐
│ Training Pipeline Health                      [Today]│
├────────────────────────┬─────────────────────────────┤
│ Code Activity          │ Training Runs               │
│ • Commits: 45          │ • Active: 12                │
│ • PRs merged: 8        │ • Failed: 2                 │
│ (GitHub)               │ (WandB)                     │
├────────────────────────┼─────────────────────────────┤
│ GPU Utilization        │ Daily Costs                 │
│ • L4: 184 GPUs         │ • EC2: $2,400               │
│ • Efficiency: 85%      │ • Skypilot: $1,200          │
│ (EC2 + K8s)            │ (EC2 + Skypilot)            │
└────────────────────────┴─────────────────────────────┘
```

**Metrics Correlation**:

- Code velocity (GitHub) → Training activity (WandB)
- GPU allocation (K8s) → Job status (Skypilot)
- Cost trends (EC2) → Resource usage (all collectors)

---

### 9. Enhanced Health FoM Dashboard - 2 days

**Goal**: Make FoM metrics more actionable

**Enhancements**:

1. **Drill-Down Links**: Click FoM cell → Navigate to detailed metric view
2. **Trend Indicators**: Show arrows for improving/degrading metrics
3. **Historical Comparison**: Compare current week vs. previous week
4. **Threshold Tuning**: Adjust FoM formulas based on team feedback
5. **Additional FoM Metrics**:
   - Training health (WandB metrics)
   - Infrastructure health (EC2/K8s metrics)
   - Project health (Asana metrics)

**Example Layout**:

```
┌─────────────────────────────────────────────────────────┐
│ System Health Rollup                 Week of Oct 21-27  │
├─────────────┬──────┬──────┬──────┬──────┬──────┬────────┤
│ Metric      │ -6d  │ -5d  │ -4d  │ -3d  │ -2d  │ -1d    │Today│ Trend│
├─────────────┼──────┼──────┼──────┼──────┼──────┼────────┤
│ Tests       │ 1.0  │ 1.0  │ 0.0  │ 1.0  │ 1.0  │ 1.0    │ 1.0 │  ↑   │
│ Benchmarks  │ 1.0  │ 1.0  │ 1.0  │ 1.0  │ 1.0  │ 1.0    │ 1.0 │  →   │
│ Workflow    │ 0.95 │ 0.92 │ 0.88 │ 0.90 │ 0.93 │ 0.95   │ 0.97│  ↑   │
...
└─────────────┴──────┴──────┴──────┴──────┴──────┴────────┴─────┘
```

---

## 🔮 Long-term Vision (Next 6 Months)

### 10. Jsonnet Dashboard System (Optional) - 1 week

**Goal**: Composable, version-controlled dashboards

**Current Approach**: JSON files in `templates/`

**Proposed**: Jsonnet system (like Grafana's Grafonnet)

**Benefits**:

- Reusable widget components
- Mix-and-match widgets across dashboards
- Grid layouts with automatic positioning
- Type-safe dashboard configuration
- Version control for dashboard changes

**Files**:

- `devops/datadog/docs/JSONNET_DESIGN.md` - Design doc
- `devops/datadog/docs/JSONNET_PROTOTYPE.md` - Implementation guide

**Decision**: Defer until we have 10+ dashboards and see pain points with JSON approach

---

### 11. Additional Collector Ideas

**Potential Future Collectors**:

1. **SkyPilot Cluster Health** (Extended)
   - Cluster uptime/downtime
   - Node health per cluster
   - Cost per cluster
   - Job distribution across clusters

2. **PyPI Package Stats**
   - Download counts for published packages
   - Version distribution
   - User geography

3. **Documentation Metrics**
   - Doc page views (if hosted)
   - Search queries
   - Time on page

4. **Model Registry**
   - Models published
   - Model downloads
   - Benchmark results

**Evaluation Criteria**:

- Is the data source reliable and accessible?
- Will the metrics drive decisions?
- Can we maintain the collector long-term?
- Does it integrate with existing dashboards?

---

## 📊 Priority Matrix

| Task                       | Impact | Effort | Priority | Status      | Timeline     |
| -------------------------- | ------ | ------ | -------- | ----------- | ------------ |
| Merge current PR           | High   | 2h     | High     | In Progress | This week    |
| Monitor production         | High   | 1h/day | High     | Ongoing     | Continuous   |
| SLO tracking               | High   | 2d     | High     | Planned     | Next week    |
| Alerting setup             | High   | 2d     | High     | Planned     | Next week    |
| Template variables         | Medium | 1d     | Medium   | Planned     | 2 weeks      |
| Anomaly detection          | Medium | 1d     | Medium   | Planned     | 2 weeks      |
| Cross-service dashboards   | Medium | 3d     | Medium   | Planned     | 1 month      |
| Multi-environment support  | Medium | 2d     | Low      | Planned     | 1 month      |
| Enhanced FoM dashboard     | Medium | 2d     | Low      | Planned     | 6 weeks      |
| Jsonnet system             | Low    | 1w     | Low      | Deferred    | TBD          |
| ✅ Clean up old code       | Low    | 2h     | High     | ✅ Complete | Oct 24, 2025 |
| ✅ GitHub collector        | High   | 3d     | High     | ✅ Complete | Oct 23, 2025 |
| ✅ Skypilot collector      | Medium | 2d     | Medium   | ✅ Complete | Oct 23, 2025 |
| ✅ Asana collector         | Low    | 2d     | Low      | ✅ Complete | Oct 23, 2025 |
| ✅ EC2 collector           | High   | 2d     | Medium   | ✅ Complete | Oct 23, 2025 |
| ✅ WandB collector         | High   | 3d     | Medium   | ✅ Complete | Oct 23, 2025 |
| ✅ Kubernetes collector    | Medium | 2d     | Medium   | ✅ Complete | Oct 23, 2025 |
| ✅ Health FoM collector    | Medium | 2d     | High     | ✅ Complete | Oct 23, 2025 |
| ✅ Production dashboards   | High   | 4d     | High     | ✅ Complete | Oct 24, 2025 |

---

## 📈 Success Metrics

### ✅ Week 1 - ACHIEVED (Oct 24, 2025)

- ✅ 7 collectors in production
- ✅ 8 production dashboards deployed
- ✅ Clean codebase (old code removed)
- ✅ 123+ metrics flowing to Datadog
- ✅ Foundation for additional features

### Week 2-3 (Next Priorities)

- 🎯 SLO tracking configured
- 🔔 Critical alerts set up
- 📊 Template variables on key dashboards
- 🤖 Anomaly detection enabled
- 📈 Team actively using dashboards

### Month 2-3

- 🎨 Cross-service correlation dashboards
- 🌍 Multi-environment support (staging/production)
- 📊 Enhanced FoM dashboard with trends
- 🔔 Mature alerting with on-call rotation
- 📊 Regular metric reviews in team meetings

### Quarter 2

- 🤖 Advanced anomaly detection with ML
- 📊 Comprehensive dashboard suite (10+ dashboards)
- 🔌 Additional collectors as needed
- 📚 Team training on dashboard usage
- 🎯 SLOs integrated into team workflows

---

## ❓ Open Questions

1. **Dashboard tool**: Stick with JSON or invest in Jsonnet system?
   - **Current Decision**: Stick with JSON until we have 10+ dashboards and pain points emerge
   - **Rationale**: JSON is working well, Jsonnet adds complexity

2. **Alerting strategy**: How to balance visibility vs. noise?
   - **Current Decision**: Start conservative with critical alerts only
   - **Next Step**: Add alerts based on actual incidents and team feedback

3. **Multi-repo support**: Just Metta or expand to other repos?
   - **Current Decision**: Start with Metta, expand if successful
   - **Evaluation Criteria**: Team adoption, metric utility, maintenance burden

4. **FoM thresholds**: Are current 0.7/0.3 thresholds appropriate?
   - **Current Decision**: Start with these, tune based on team feedback
   - **Next Step**: Review after 2 weeks of production data

5. **Additional collectors**: What else should we monitor?
   - **Current Decision**: Focus on improving existing collectors and dashboards first
   - **Evaluation**: Revisit quarterly based on team needs

---

## 📚 Resources

### Documentation

- [Collectors Architecture](docs/COLLECTORS_ARCHITECTURE.md)
- [Adding New Collector](docs/ADDING_NEW_COLLECTOR.md)
- [Metric Conventions](docs/METRIC_CONVENTIONS.md)
- [CI/CD Metrics](docs/CI_CD_METRICS.md)
- [Wildcard Widget Guide](docs/WILDCARD_WIDGET.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

### Tools

- **CLI**: `metta datadog --help`
- **Dashboard Scripts**: `devops/datadog/scripts/`
- **Datadog API**: [Dashboard JSON Reference](https://docs.datadoghq.com/dashboards/graphing_json/)
- **Vega-Lite**: [Vega-Lite Documentation](https://vega.github.io/vega-lite/)

### Team

- **Owner**: DevOps team
- **Stakeholders**: Engineering, Research, SRE
- **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

---

**Last Updated**: 2025-10-24 **Next Review**: After current PR merge
