# Datadog Dashboard Design

## Overview

This document outlines the dashboard architecture for the Metta observability system. We design dashboards at three
levels: executive rollup, per-collector deep dives, and metric-level drill-downs.

**Status**: Planning phase **Last Updated**: 2025-10-23

## Design Principles

### 1. Information Hierarchy

**Three levels of detail:**

1. **Executive Level**: Rollup Health Table
   - High-level system health at a glance
   - Red/yellow/green status across all metrics
   - "What's broken right now?"
   - Primary audience: Leadership, team standups

2. **System Level**: Per-Collector Dashboards
   - Deep dive into specific areas (CI/CD, Training, etc.)
   - Time series trends and distributions
   - "What's the trend and why?"
   - Primary audience: Engineers, researchers

3. **Metric Level**: Individual drill-downs
   - Single metric deep dive
   - Historical analysis, anomaly detection
   - "What exactly happened?"
   - Primary audience: On-call, debugging

### 2. Dashboard Organization

**One dashboard per collector** + **One rollup dashboard**

```
Metta Observability Dashboards
├── 🎯 System Health (Rollup)          [Main entry point]
├── 🔧 GitHub CI/CD                     [Per-collector]
├── 🧠 Training & WandB                 [Per-collector]
├── ⚡ Skypilot Jobs                    [Per-collector]
├── 📊 Eval & Testing                   [Per-collector]
└── 🏥 Collector Health                 [Per-collector - meta]
```

### 3. Visual Consistency

**Standard layout for all dashboards:**

```
┌────────────────────────────────────────────────────────────┐
│ Dashboard Title                    [Time Selector] [Filters]│
├────────────────────────────────────────────────────────────┤
│ Row 1: Key Metrics (Big Numbers)                           │
│  [Metric 1]  [Metric 2]  [Metric 3]  [Metric 4]           │
├────────────────────────────────────────────────────────────┤
│ Row 2: Primary Time Series                                 │
│  [Chart 1: Most Important Trend]                           │
├────────────────────────────────────────────────────────────┤
│ Row 3: Secondary Metrics                                   │
│  [Chart 2]              [Chart 3]                          │
├────────────────────────────────────────────────────────────┤
│ Row 4: Detailed Analysis                                   │
│  [Distribution] [Table] [Breakdown]                        │
└────────────────────────────────────────────────────────────┘
```

### 4. Color Coding Standards

**Traffic light system:**

- 🟢 **Green** (0.7-1.0): Healthy, all good
- 🟡 **Yellow** (0.3-0.7): Warning, needs attention
- 🔴 **Red** (0.0-0.3): Critical, action required
- ⚪ **Gray**: No data available

**Use consistently across all dashboards**

### 5. Widget Selection Guide

| Data Type             | Best Widget                        | Use Case                    |
| --------------------- | ---------------------------------- | --------------------------- |
| Current count/status  | Query Value (big number)           | "42 open PRs"               |
| Binary status         | Query Value with conditional color | "Tests: PASSING"            |
| Trend over time       | Timeseries                         | "PR velocity over 30 days"  |
| Multiple percentiles  | Timeseries (multi-line)            | "p50, p90, p99 CI duration" |
| Categorical breakdown | Pie chart or Bar chart             | "Jobs by status"            |
| Top-N items           | Top List                           | "Top 10 slowest tests"      |
| Comparison table      | Query Table                        | "Metrics by team/repo"      |
| Raw data              | Table                              | "Recent failed workflows"   |

## Dashboard Designs

### 1. System Health Rollup Dashboard

**Purpose**: Executive summary of system health across all areas **Audience**: Leadership, team standups, at-a-glance
health checks **Update Frequency**: Hourly (FoM values recalculated)

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ 🎯 Metta System Health                  [Last 7 Days]      │
├────────────────────────────────────────────────────────────┤
│ Health Score:  Training: 🟢 0.85  |  CI/CD: 🟡 0.65  |     │
│                Eval: 🟢 0.92      |  Overall: 🟢 0.81      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  HEALTH METRICS TABLE (7-Day Rolling Window)               │
│                                                             │
│  Metric Name                -6d -5d -4d -3d -2d -1d  0d    │
│  ────────────────────────── ─── ─── ─── ─── ─── ─── ───   │
│  Training                                                   │
│    Multigpu Arena - Runs    🟢  🟢  🟡  🟢  🟢  🟢  🟢    │
│    Multigpu Arena - Hearts  🟢  🟢  🟢  🟡  🟢  🟢  🟢    │
│    Multigpu Arena - SPS     🟡  🟢  🟢  🟢  🟢  🟡  🟢    │
│    Multinode - Runs         🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Multinode - Hearts       🟢  🟡  🟢  🟢  🟢  🟢  🟢    │
│    Multinode - Shaped SPS   🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Local - Checkpoint       🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Local - Resume           🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Training Bug Count       🟢  🟢  🟢  🟢  🟡  🟢  🟢    │
│                                                             │
│  CI/CD                                                      │
│    Tests Passing on Main    🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Benchmarks Passing       🟢  🟢  🔴  🟡  🟢  🟢  🟢    │
│    Failing Workflows        🟢  🟡  🟡  🟢  🟢  🟢  🟢    │
│    Hotfix Count            🟢  🟢  🟢  🟡  🟡  🟢  🟢    │
│    Force Merge Count       🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Revert Count            🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    CI Duration (P90)       🟢  🟢  🟡  🟢  🟢  🟢  🟢    │
│    Timeout Cancellations   🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Flaky Checks            🟡  🟡  🟢  🟢  🟢  🟢  🟢    │
│    Stale PRs               🟡  🟡  🟡  🟢  🟢  🟢  🟢    │
│    PR Cycle Time           🟢  🟢  🟢  🟡  🟢  🟢  🟢    │
│                                                             │
│  Eval                                                       │
│    Local Eval - Success     🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Local Eval - Accuracy    🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Remote Eval - Success    🟢  🟢  🟢  🟢  🟢  🟢  🟢    │
│    Remote Eval - Accuracy   🟢  🟢  🟡  🟢  🟢  🟢  🟢    │
│    Remote Eval - Duration   🟢  🟢  🟢  🟢  🟢  🟡  🟢    │
│                                                             │
├────────────────────────────────────────────────────────────┤
│ Quick Links                                                 │
│  → GitHub CI/CD Dashboard    → Training Dashboard           │
│  → Skypilot Dashboard        → Eval Dashboard              │
└────────────────────────────────────────────────────────────┘
```

#### Implementation Notes

**Widget Type**: Query Table **Queries**: One per metric (25 total)

```
avg:health.training.multigpu_arena_hearts.fom{*} by {day}
avg:health.ci.tests_passing.fom{*} by {day}
...
```

**Conditional Formatting**:

- Green: `>= 0.7`
- Yellow: `>= 0.3 and < 0.7`
- Red: `< 0.3`

**Grouping**: Manual row organization with markdown headers

**Template Variables**:

- Time range selector (fixed to 7 days)
- Optional: Filter by environment (production/staging)

---

### 2. GitHub CI/CD Dashboard

**Purpose**: Deep dive into development velocity and code quality **Audience**: Engineering team, DevOps **Update
Frequency**: Every 15 minutes (collector runs)

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ 🔧 GitHub CI/CD Dashboard              [Last 30 Days] [↻] │
├────────────────────────────────────────────────────────────┤
│ Row 1: Current Status (Big Numbers)                        │
├─────────────┬──────────────┬──────────────┬───────────────┤
│ Open PRs    │ PRs Merged   │ Active       │ Tests on Main │
│             │ (7 days)     │ Developers   │               │
│     42      │     28       │     12       │  ✅ PASSING   │
│ ────────    │ ────────     │ ────────     │  ────────     │
│ query_value │ query_value  │ query_value  │ query_value   │
└─────────────┴──────────────┴──────────────┴───────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 2: PR Velocity & Health                                │
├─────────────────────────────┬──────────────────────────────┤
│ PR Cycle Time (Hours)       │ Stale PRs (>14 days)        │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - p50 (target: <24h)       │  - Count                    │
│  - p90 (target: <48h)       │  - Threshold: 20            │
│  - Trend line               │                             │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 3: Code Quality Signals                                │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Hotfixes     │ Reverts      │ Force Merges │ Workflow     │
│ (7 days)     │ (7 days)     │ (7 days)     │ Failures     │
│              │              │              │ (7 days)     │
│ [Timeseries] │ [Timeseries] │ [Timeseries] │ [Timeseries] │
│ Target: <5   │ Target: 0    │ Target: <7   │ Target: <2   │
└──────────────┴──────────────┴──────────────┴──────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 4: CI Performance                                       │
├─────────────────────────────┬──────────────────────────────┤
│ CI Duration Percentiles     │ Workflow Success Rate        │
│                             │                              │
│  [Timeseries - Multi-line]  │  [Stacked Area Chart]       │
│  - p50                      │  - Success                  │
│  - p90 (target: <5 min)     │  - Failure                  │
│  - p99                      │  - Timeout                  │
│                             │                              │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 5: Developer Activity                                   │
├─────────────────────────────┬──────────────────────────────┤
│ Commits per Developer       │ Top Contributors (30d)       │
│                             │                              │
│  [Timeseries]               │  [Top List]                 │
│  - Average                  │  1. Alice (142 commits)     │
│  - Trend                    │  2. Bob (98 commits)        │
│                             │  3. Charlie (67 commits)    │
│                             │  ...                        │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 6: Recent Activity (Tables)                             │
├────────────────────────────────────────────────────────────┤
│ Recent Failed Workflows                                     │
│  [Table Widget]                                             │
│  - Workflow Name | Branch | Duration | Failed At           │
│  - Clickable links to GitHub                               │
└────────────────────────────────────────────────────────────┘
```

#### Metrics Used

**From existing GitHub collector:**

- `github.prs.open`
- `github.prs.merged_7d`
- `github.prs.cycle_time_hours`
- `github.prs.stale_count_14d`
- `github.commits.total_7d`
- `github.commits.per_developer_7d`
- `github.commits.hotfix`
- `github.commits.reverts`
- `github.developers.active_7d`
- `github.ci.tests_passing_on_main`
- `github.ci.workflow_runs_7d`
- `github.ci.failed_workflows_7d`
- `github.ci.duration_p50_minutes`
- `github.ci.duration_p90_minutes`
- `github.ci.duration_p99_minutes`

**New metrics needed:**

- `github.commits.force_merge_7d`
- `github.ci.timeout_cancellations_7d`
- `github.ci.flaky_checks_7d`
- `github.ci.benchmarks_passing`

---

### 3. Training & WandB Dashboard

**Purpose**: Monitor training runs, model performance, resource usage **Audience**: Research team, ML engineers **Update
Frequency**: Every 15 minutes

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ 🧠 Training & WandB Dashboard          [Last 7 Days]  [↻] │
├────────────────────────────────────────────────────────────┤
│ Row 1: Training Status                                      │
├─────────────┬──────────────┬──────────────┬───────────────┤
│ Active Runs │ Completed    │ Failed       │ Queued        │
│             │ (24h)        │ (24h)        │               │
│      8      │     12       │      2       │      3        │
│ ────────    │ ────────     │ ────────     │ ────────      │
│ query_value │ query_value  │ query_value  │ query_value   │
└─────────────┴──────────────┴──────────────┴───────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 2: Model Performance (Multigpu Arena)                   │
├─────────────────────────────┬──────────────────────────────┤
│ Hearts Score (Avg)          │ Steps Per Second (SPS)      │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - Target: > 0.5            │  - Target: > 40k            │
│  - Current runs             │  - Throughput trend         │
│  - Moving average           │                             │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 3: Model Performance (Multinode)                        │
├─────────────────────────────┬──────────────────────────────┤
│ Hearts Score (Avg)          │ Shaped Reward SPS           │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - Target: > 0.5            │  - Target: > 40k            │
│  - Multi-node efficiency    │  - Scaling behavior         │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 4: Resource Utilization                                 │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ GPU          │ Training     │ Compute Cost │ Run          │
│ Utilization  │ Duration     │ (7 days)     │ Success Rate │
│              │ (Avg)        │              │              │
│ [Gauge]      │ [Timeseries] │ [Big Number] │ [Gauge]      │
│ Target: >80% │              │              │ Target: >95% │
└──────────────┴──────────────┴──────────────┴──────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 5: Training Breakdown                                   │
├─────────────────────────────┬──────────────────────────────┤
│ Jobs by Status              │ Recent Runs                  │
│                             │                              │
│  [Pie Chart]                │  [Table]                    │
│  - Running                  │  - Run ID | Started | Status│
│  - Succeeded                │  - Hearts | SPS | Duration  │
│  - Failed                   │  - Link to WandB            │
│  - Cancelled                │                             │
└─────────────────────────────┴──────────────────────────────┘
```

#### Metrics Used

**From planned WandB collector:**

- `wandb.runs.active`
- `wandb.runs.completed_24h`
- `wandb.runs.failed_24h`
- `wandb.runs.queued`
- `wandb.training.arena.avg_hearts`
- `wandb.training.arena.avg_sps`
- `wandb.training.multinode.avg_hearts`
- `wandb.training.multinode.shaped_reward_sps`
- `wandb.training.gpu_utilization_pct`
- `wandb.training.duration_hours`
- `wandb.training.cost_estimate_usd_7d`
- `wandb.training.success_rate_pct`

---

### 4. Skypilot Jobs Dashboard

**Purpose**: Monitor job orchestration, cluster health, compute costs **Audience**: DevOps, research team **Update
Frequency**: Every 10 minutes

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ ⚡ Skypilot Jobs Dashboard             [Last 7 Days]  [↻] │
├────────────────────────────────────────────────────────────┤
│ Row 1: Job Status                                           │
├─────────────┬──────────────┬──────────────┬───────────────┤
│ Running     │ Queued       │ Failed       │ Active        │
│ Jobs        │ Jobs         │ (7 days)     │ Clusters      │
│      5      │      2       │      3       │      4        │
│ ────────    │ ────────     │ ────────     │ ────────      │
│ query_value │ query_value  │ query_value  │ query_value   │
└─────────────┴──────────────┴──────────────┴───────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 2: Job Trends                                           │
├─────────────────────────────┬──────────────────────────────┤
│ Jobs Over Time              │ Job Success Rate             │
│                             │                              │
│  [Stacked Area]             │  [Timeseries]               │
│  - Running                  │  - Success %                │
│  - Queued                   │  - Failure %                │
│  - Failed                   │  - Target: >95%             │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 3: Cluster Health                                       │
├─────────────────────────────┬──────────────────────────────┤
│ Active Clusters             │ Cluster Uptime               │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - Count over time          │  - Average per cluster      │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 4: Cost Analysis                                        │
├──────────────┬──────────────┬──────────────┬──────────────┤
│ Daily Cost   │ Weekly Cost  │ Cost by      │ Cost Trend   │
│              │              │ Cluster      │              │
│ [Big Number] │ [Big Number] │ [Pie Chart]  │ [Timeseries] │
└──────────────┴──────────────┴──────────────┴──────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 5: Recent Activity                                      │
├────────────────────────────────────────────────────────────┤
│ Recent Jobs                                                 │
│  [Table]                                                    │
│  - Job ID | Status | Cluster | Started | Duration | Cost   │
└────────────────────────────────────────────────────────────┘
```

#### Metrics Used

**From existing Skypilot collector:**

- `skypilot.jobs.running`
- `skypilot.jobs.queued`
- `skypilot.jobs.failed`
- `skypilot.jobs.failed_7d`
- `skypilot.jobs.succeeded`
- `skypilot.jobs.cancelled`
- `skypilot.clusters.active`

---

### 5. Eval & Testing Dashboard

**Purpose**: Monitor evaluation runs and model testing **Audience**: Research team, QA **Update Frequency**: Every 15
minutes

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ 📊 Eval & Testing Dashboard            [Last 7 Days]  [↻] │
├────────────────────────────────────────────────────────────┤
│ Row 1: Eval Status                                          │
├─────────────┬──────────────┬──────────────┬───────────────┤
│ Local Eval  │ Remote Eval  │ Accuracy     │ Eval Duration │
│ Success     │ Success      │ Delta        │ (Avg)         │
│    ✅ PASS  │    ✅ PASS   │    < 5%      │   3.2 min     │
│ ────────    │ ────────     │ ────────     │ ────────      │
│ query_value │ query_value  │ query_value  │ query_value   │
└─────────────┴──────────────┴──────────────┴───────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 2: Evaluation Trends                                    │
├─────────────────────────────┬──────────────────────────────┤
│ Hearts Accuracy (%)         │ Eval Success Rate            │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - Local eval               │  - Local                    │
│  - Remote eval              │  - Remote                   │
│  - Target: < 10% diff       │  - Target: 100%             │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 3: Performance                                          │
├─────────────────────────────┬──────────────────────────────┤
│ Remote Eval Duration        │ Eval Runs (24h)             │
│                             │                              │
│  [Timeseries]               │  [Stacked Bar]              │
│  - p50, p90 duration        │  - Successful               │
│  - Target: < 5 min          │  - Failed                   │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 4: Recent Evaluations                                   │
├────────────────────────────────────────────────────────────┤
│ Recent Eval Runs                                            │
│  [Table]                                                    │
│  - Policy | Type | Status | Hearts Acc | Duration | Time   │
└────────────────────────────────────────────────────────────┘
```

---

### 6. Collector Health Dashboard (Meta-monitoring)

**Purpose**: Monitor the health of collectors themselves **Audience**: DevOps, on-call **Update Frequency**: Real-time

#### Layout

```
┌────────────────────────────────────────────────────────────┐
│ 🏥 Collector Health                    [Last 24h]     [↻] │
├────────────────────────────────────────────────────────────┤
│ Row 1: Collector Status                                     │
├────────────────────────────────────────────────────────────┤
│  Collector Health Matrix                                    │
│  [Table]                                                    │
│  Collector  | Last Run | Duration | Metrics | Errors | Status│
│  ──────────┼──────────┼──────────┼─────────┼────────┼───────│
│  GitHub    | 2m ago   | 1.2s     | 25      | 0      | 🟢    │
│  WandB     | 5m ago   | 2.1s     | 15      | 0      | 🟢    │
│  Skypilot  | 8m ago   | 0.8s     | 7       | 0      | 🟢    │
│  Health    | 12m ago  | 3.5s     | 25      | 0      | 🟢    │
│  Eval      | 15m ago  | 1.5s     | 5       | 1      | 🟡    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 2: Collector Performance                                │
├─────────────────────────────┬──────────────────────────────┤
│ Collection Duration         │ Error Rate                   │
│                             │                              │
│  [Timeseries]               │  [Timeseries]               │
│  - Per collector            │  - Per collector            │
│  - Alert: > 30s             │  - Alert: > 3 errors/hour   │
└─────────────────────────────┴──────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Row 3: Recent Errors                                        │
├────────────────────────────────────────────────────────────┤
│ Recent Collector Errors                                     │
│  [Table - Live Feed]                                        │
│  - Timestamp | Collector | Error Message | Metric          │
└────────────────────────────────────────────────────────────┘
```

#### Metrics Used

**Auto-emitted by all collectors:**

- `collector.{name}.duration_seconds`
- `collector.{name}.metric_count`
- `collector.{name}.error_count`
- `collector.{name}.failed` (binary)

---

## Dashboard Navigation & Links

### URL Structure

```
https://app.datadoghq.com/dashboard/{dashboard-id}

Naming convention:
- system-health-rollup
- github-cicd
- training-wandb
- skypilot-jobs
- eval-testing
- collector-health
```

### Cross-Dashboard Links

**From Rollup → Per-Collector:**

- Links in table rows
- "View Details" buttons
- Click metric name → drill down

**From Per-Collector → Rollup:**

- Breadcrumb navigation
- "Back to System Health" link

**From Any Dashboard → Collector Health:**

- Status indicator in header
- "View Collector Health" link

---

## Implementation Priority

### Phase 1: Week 1-2

**Build these first:**

1. ✅ **System Health Rollup** (main dashboard)
   - Start with 7 CI metrics we have
   - Placeholder rows for training/eval
2. ✅ **GitHub CI/CD Dashboard**
   - Use all existing 25 metrics
   - Most complete from day 1

### Phase 2: Week 3-4

**Add after collectors deployed:** 3. **Training & WandB Dashboard**

- Deploy WandB collector first
- Build dashboard

4. **Skypilot Jobs Dashboard**
   - Skypilot collector already done
   - Build dashboard

### Phase 3: Week 5-6

**Complete the suite:** 5. **Eval & Testing Dashboard**

- Deploy eval collector
- Build dashboard

6. **Collector Health Dashboard**
   - Meta-monitoring
   - Critical for operations

---

## Dashboard Management

### Creation Process

```bash
# 1. Design layout (Figma, paper, or directly in Datadog UI)

# 2. Create in Datadog UI first (easier for prototyping)
# - Use GUI to build initial version
# - Test queries and visualizations

# 3. Export JSON
metta datadog dashboard pull

# 4. Find the new dashboard JSON
ls -lt devops/datadog/templates/

# 5. Commit to version control
git add devops/datadog/templates/github_cicd.json
git commit -m "feat: add GitHub CI/CD dashboard"

# 6. Future updates via JSON
vim devops/datadog/templates/github_cicd.json
metta datadog dashboard push
```

### Version Control

- **DO** commit dashboard JSON files
- **DO** add descriptive commit messages
- **DO** review dashboard changes in PRs
- **DON'T** edit JSON by hand (use UI then export)

### Naming Convention

```
Templates directory:
- system_health_rollup.json
- github_cicd.json
- training_wandb.json
- skypilot_jobs.json
- eval_testing.json
- collector_health.json
```

---

## Open Questions

1. **Dashboard Ownership**: Who maintains each dashboard?
   - Proposal: Per-collector dashboards owned by team that owns that system
   - Rollup owned by DevOps

2. **Alert Thresholds**: When to alert vs just visualize?
   - Proposal: Start conservative, add alerts based on real incidents

3. **Refresh Rate**: Real-time vs batched updates?
   - Proposal: Collector frequency = dashboard refresh rate (5-15 min)

4. **Custom Time Windows**: Allow users to select time range?
   - Proposal: Yes for per-collector dashboards, fixed for rollup

5. **Multi-Environment**: Separate dashboards for staging/production?
   - Proposal: Use template variables to filter, not separate dashboards

---

## Next Steps

1. ✅ Review this design document with team
2. Create mockups/screenshots for key dashboards
3. Build Phase 1 dashboards (Rollup + GitHub)
4. Get user feedback
5. Iterate and expand

---

**Last Updated**: 2025-10-23 **Owner**: DevOps team **Reviewers**: Engineering, Research
