# System Health Dashboard - Implementation Specification

## Overview

A Datadog table visualization displaying normalized health metrics across Training, CI, and Eval workflows. Each row represents a metric scaled to [0,1] as a "Figure of Merit" (FoM) and visualized with a continuous red-to-green color spectrum over a 7-day rolling window.

**Status**: Planning phase
**Dependencies**: WandB collector, training metrics collector, eval metrics collector
**Existing**: GitHub collector (deployed), Skypilot collector (implemented)

## Architecture Integration

This dashboard builds on the existing Datadog collector architecture:

### Existing Infrastructure
- **Collector Framework**: `BaseCollector` pattern with `@metric` decorator
- **Deployed Collectors**: GitHub (25 metrics), Skypilot (7 metrics)
- **Dashboard Tools**: CLI commands via `metta datadog dashboard`
- **Deployment**: Kubernetes CronJobs via Helm charts
- **Documentation**: See `COLLECTORS_ARCHITECTURE.md`, `ADDING_NEW_COLLECTOR.md`

### New Components Needed
1. **FoM Processing Collector**: Reads raw metrics, calculates FoM values, emits normalized metrics
2. **Training Metrics Collectors**: WandB integration, training job status collector
3. **Eval Metrics Collector**: Local and remote evaluation metrics
4. **Health Dashboard**: Datadog query table displaying FoM values over 7 days

## Table Structure

### Visual Layout

```
┌────────────────────────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Metric Name                    │ -6d │ -5d │ -4d │ -3d │ -2d │ -1d │  0d │
├────────────────────────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ Training                       │     │     │     │     │     │     │     │
│   Multigpu Arena - Run Success│ 🟢  │ 🟢  │ 🟡  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │
│   Multigpu Arena - Hearts      │ 🟢  │ 🟢  │ 🟢  │ 🟡  │ 🟢  │ 🟢  │ 🟢  │
│   Multigpu Arena - SPS         │ 🟡  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟡  │ 🟢  │
│   ...                          │     │     │     │     │     │     │     │
│ CI/CD                          │     │     │     │     │     │     │     │
│   Main Branch - Tests Passing │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │
│   Main Branch - Benchmarks    │ 🟢  │ 🟢  │ 🔴  │ 🟡  │ 🟢  │ 🟢  │ 🟢  │
│   ...                          │     │     │     │     │     │     │     │
│ Eval                           │     │     │     │     │     │     │     │
│   Local Eval - Run Success    │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │ 🟢  │
│   ...                          │     │     │     │     │     │     │     │
└────────────────────────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Color Scale: 🔴 Red (0.0-0.3) → 🟡 Yellow (0.3-0.7) → 🟢 Green (0.7-1.0)
```

### Columns
- **Metric Name** (fixed, left column)
- **Day -6** through **Day 0** (7 daily columns, rightmost = today)
- Each cell shows FoM value in [0,1] with color coding

### Rows
Hierarchical grouping by workflow category (~35 total metrics):
1. **Training** (9 metrics)
2. **CI/CD** (11 metrics)
3. **Eval** (5 metrics)
4. **Additional** (up to 10 more)

## Metric Definitions & Sources

### Training Metrics (9 total)

#### 1. Multigpu Arena - Run Success
- **Metric Name**: `health.training.multigpu_arena_run_success.fom`
- **Raw Source**: `wandb.training.arena.runs_completed_24h`
- **Collector**: WandB collector (`devops/datadog/collectors/wandb/`)
- **Condition**: ≥ 1 successful run per day
- **FoM Formula**:
  ```python
  fom = min(successful_runs / 1.0, 1.0)
  # 1+ runs = 1.0 (green), 0 runs = 0.0 (red)
  ```

#### 2. Multigpu Arena - Hearts Score
- **Metric Name**: `health.training.multigpu_arena_hearts.fom`
- **Raw Source**: `wandb.training.arena.avg_hearts`
- **Collector**: WandB collector
- **Condition**: > 0.5 (target: 0.7)
- **FoM Formula**:
  ```python
  fom = clip((hearts_avg - 0.0) / (0.7 - 0.0), 0, 1)
  # 0.0→0.0, 0.5→0.71, 0.7+→1.0
  ```

#### 3. Multigpu Arena - SPS (Steps Per Second)
- **Metric Name**: `health.training.multigpu_arena_sps.fom`
- **Raw Source**: `wandb.training.arena.avg_sps`
- **Collector**: WandB collector
- **Condition**: > 40,000 SPS
- **FoM Formula**:
  ```python
  fom = clip((sps - 20000) / (50000 - 20000), 0, 1)
  # 20k→0.0, 40k→0.67, 50k+→1.0
  ```

#### 4. Multinode - Run Success
- **Metric Name**: `health.training.multinode_run_success.fom`
- **Raw Source**: `wandb.training.multinode.runs_completed_24h`
- **Collector**: WandB collector
- **Condition**: ≥ 1 successful run per day
- **FoM Formula**:
  ```python
  fom = min(successful_runs / 1.0, 1.0)
  ```

#### 5. Multinode - Hearts Score
- **Metric Name**: `health.training.multinode_hearts.fom`
- **Raw Source**: `wandb.training.multinode.avg_hearts`
- **Collector**: WandB collector
- **Condition**: > 0.5
- **FoM Formula**:
  ```python
  fom = clip((hearts_avg - 0.0) / (0.7 - 0.0), 0, 1)
  ```

#### 6. Multinode - Shaped Metric
- **Metric Name**: `health.training.multinode_shaped_sps.fom`
- **Raw Source**: `wandb.training.multinode.shaped_reward_sps`
- **Collector**: WandB collector
- **Condition**: > 40,000
- **FoM Formula**:
  ```python
  fom = clip((shaped_sps - 20000) / (50000 - 20000), 0, 1)
  ```

#### 7. Local Arena - Checkpoint at 10k Steps
- **Metric Name**: `health.training.local_checkpoint_success.fom`
- **Raw Source**: `training.local.checkpoint_success_24h`
- **Collector**: Training jobs collector (new)
- **Condition**: ≥ 1 successful checkpoint per day
- **FoM Formula**:
  ```python
  fom = min(checkpoint_successes / 1.0, 1.0)
  ```

#### 8. Local Arena - Resume from Checkpoint
- **Metric Name**: `health.training.local_resume_success.fom`
- **Raw Source**: `training.local.resume_success_24h`
- **Collector**: Training jobs collector (new)
- **Condition**: ≥ 1 successful resume per day
- **FoM Formula**:
  ```python
  fom = min(resume_successes / 1.0, 1.0)
  ```

#### 9. Training - Bug Count
- **Metric Name**: `health.training.bug_count.fom`
- **Raw Source**: `github.issues.training_bugs`
- **Collector**: GitHub collector (extend existing)
- **Condition**: < 1 (ideally 0)
- **FoM Formula**:
  ```python
  fom = max(1.0 - (bug_count / 3.0), 0.0)
  # 0→1.0, 1→0.67, 2→0.33, 3+→0.0
  ```

### CI/CD Metrics (11 total)

#### 10. Main Branch - Tests Passing
- **Metric Name**: `health.ci.tests_passing.fom`
- **Raw Source**: `github.ci.tests_passing_on_main` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: 1 (all tests passing)
- **FoM Formula**:
  ```python
  fom = 1.0 if tests_passing else 0.0
  # Binary: pass=1.0, fail=0.0
  ```

#### 11. Main Branch - Benchmarks Passing
- **Metric Name**: `health.ci.benchmarks_passing.fom`
- **Raw Source**: `github.ci.benchmarks_passing` (new metric)
- **Collector**: GitHub collector (extend)
- **Condition**: All benchmarks passing
- **FoM Formula**:
  ```python
  fom = benchmarks_passed / max(total_benchmarks, 1)
  ```

#### 12. Main Branch - Failing Workflows
- **Metric Name**: `health.ci.failing_workflows.fom`
- **Raw Source**: `github.ci.failed_workflows_7d` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 2 failing workflows
- **FoM Formula**:
  ```python
  fom = max(1.0 - (failing_workflows / 5.0), 0.0)
  # 0→1.0, 1→0.8, 2→0.6, 5+→0.0
  ```

#### 13. Commit History - Weekly Hotfixes
- **Metric Name**: `health.ci.hotfix_count.fom`
- **Raw Source**: `github.commits.hotfix` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 5 per week
- **FoM Formula**:
  ```python
  fom = max(1.0 - (hotfix_count / 10.0), 0.0)
  # 0→1.0, 5→0.5, 10+→0.0
  ```

#### 14. Commit History - Weekly Force Merges
- **Metric Name**: `health.ci.force_merge_count.fom`
- **Raw Source**: `github.commits.force_merge_7d` (new metric)
- **Collector**: GitHub collector (extend)
- **Condition**: < 7 per week
- **FoM Formula**:
  ```python
  fom = max(1.0 - (force_merge_count / 14.0), 0.0)
  # 0→1.0, 7→0.5, 14+→0.0
  ```

#### 15. Commit History - Weekly Reverts
- **Metric Name**: `health.ci.revert_count.fom`
- **Raw Source**: `github.commits.reverts` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 1 per week (ideally 0)
- **FoM Formula**:
  ```python
  fom = max(1.0 - (revert_count / 2.0), 0.0)
  # 0→1.0, 1→0.5, 2+→0.0
  ```

#### 16. CI Smoothness - P90 Duration
- **Metric Name**: `health.ci.duration_p90.fom`
- **Raw Source**: `github.ci.duration_p90_minutes` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 5 minutes
- **FoM Formula**:
  ```python
  fom = max(1.0 - (p90_minutes - 3.0) / (10.0 - 3.0), 0.0)
  # 3min→1.0, 5min→0.71, 10min+→0.0
  ```

#### 17. CI Smoothness - Weekly Timeout Cancellations
- **Metric Name**: `health.ci.timeout_cancellations.fom`
- **Raw Source**: `github.ci.timeout_cancellations_7d` (new metric)
- **Collector**: GitHub collector (extend)
- **Condition**: < 10 per week
- **FoM Formula**:
  ```python
  fom = max(1.0 - (timeout_count / 20.0), 0.0)
  # 0→1.0, 10→0.5, 20+→0.0
  ```

#### 18. CI Smoothness - Weekly Flaky Checks
- **Metric Name**: `health.ci.flaky_checks.fom`
- **Raw Source**: `github.ci.flaky_checks_7d` (new metric)
- **Collector**: GitHub collector (extend)
- **Condition**: < 10 per week
- **FoM Formula**:
  ```python
  fom = max(1.0 - (flaky_count / 20.0), 0.0)
  # 0→1.0, 10→0.5, 20+→0.0
  ```

#### 19. PR Velocity - Stale PRs
- **Metric Name**: `health.ci.stale_prs.fom`
- **Raw Source**: `github.prs.stale_count_14d` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 20 stale PRs
- **FoM Formula**:
  ```python
  fom = max(1.0 - (stale_prs / 50.0), 0.0)
  # 0→1.0, 20→0.6, 50+→0.0
  ```

#### 20. PR Velocity - Cycle Time
- **Metric Name**: `health.ci.pr_cycle_time.fom`
- **Raw Source**: `github.prs.cycle_time_hours` (✅ already exists!)
- **Collector**: GitHub collector (deployed)
- **Condition**: < 48 hours (p90)
- **FoM Formula**:
  ```python
  fom = max(1.0 - (cycle_time_hours - 24.0) / (72.0 - 24.0), 0.0)
  # 24h→1.0, 48h→0.5, 72h+→0.0
  ```

### Eval Metrics (5 total)

#### 21. Local Eval - Run Success
- **Metric Name**: `health.eval.local_run_success.fom`
- **Raw Source**: `eval.local.success_24h`
- **Collector**: Eval collector (new)
- **Condition**: ≥ 1 successful run per day
- **FoM Formula**:
  ```python
  fom = 1.0 if exit_code == 0 else 0.0
  ```

#### 22. Local Eval - Hearts Accuracy
- **Metric Name**: `health.eval.local_hearts_accuracy.fom`
- **Raw Source**: `eval.local.hearts_accuracy_pct`
- **Collector**: Eval collector (new)
- **Condition**: < 10% difference from expected
- **FoM Formula**:
  ```python
  fom = max(1.0 - (abs_percent_diff / 20.0), 0.0)
  # 0%→1.0, 10%→0.5, 20%+→0.0
  ```

#### 23. Remote Eval - Run Success & Fetchable
- **Metric Name**: `health.eval.remote_run_success.fom`
- **Raw Source**: `eval.remote.success_24h`
- **Collector**: Eval collector (new)
- **Condition**: ≥ 1 successful run per day
- **FoM Formula**:
  ```python
  fom = min(successful_evals / 1.0, 1.0)
  ```

#### 24. Remote Eval - Hearts Accuracy
- **Metric Name**: `health.eval.remote_hearts_accuracy.fom`
- **Raw Source**: `eval.remote.hearts_accuracy_pct`
- **Collector**: Eval collector (new)
- **Condition**: < 10% difference from expected
- **FoM Formula**:
  ```python
  fom = max(1.0 - (abs_percent_diff / 20.0), 0.0)
  ```

#### 25. Remote Eval - Duration
- **Metric Name**: `health.eval.remote_duration.fom`
- **Raw Source**: `eval.remote.duration_minutes`
- **Collector**: Eval collector (new)
- **Condition**: ≤ 5 minutes
- **FoM Formula**:
  ```python
  fom = max(1.0 - (duration_minutes - 3.0) / (10.0 - 3.0), 0.0)
  # 3min→1.0, 5min→0.71, 10min+→0.0
  ```

## Implementation Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Metrics Sources                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ GitHub       │ WandB        │ Training Jobs│ Eval System    │
│ Collector    │ Collector    │ Collector    │ Collector      │
│ (deployed)   │ (planned)    │ (new)        │ (new)          │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                           ↓
       ┌────────────────────────────────────────────┐
       │      FoM Processing Collector              │
       │  - Reads raw metrics from Datadog          │
       │  - Calculates FoM values                   │
       │  - Emits health.*.fom metrics              │
       │  - Runs every hour                         │
       └────────────────┬───────────────────────────┘
                        ↓
       ┌────────────────────────────────────────────┐
       │         Datadog (Time Series DB)           │
       │  - Stores both raw and FoM metrics         │
       │  - 7-day retention for dashboard           │
       └────────────────┬───────────────────────────┘
                        ↓
       ┌────────────────────────────────────────────┐
       │      System Health Dashboard               │
       │  - Query table widget                      │
       │  - 7-day rolling window                    │
       │  - Color-coded cells (red→yellow→green)    │
       └────────────────────────────────────────────┘
```

### FoM Processing Collector

New collector to calculate normalized health metrics:

```python
# devops/datadog/collectors/health_fom/collector.py

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.common.datadog_client import DatadogClient


class HealthFomCollector(BaseCollector):
    """
    Calculate Figure of Merit (FoM) values for system health metrics.

    Reads raw metrics from Datadog, applies normalization formulas,
    and emits health.*.fom metrics for dashboard visualization.
    """

    @property
    def name(self) -> str:
        return "health_fom"

    def collect_metrics(self) -> dict[str, float]:
        """Calculate all FoM metrics from raw sources."""
        fom_metrics = {}

        # Query Datadog for raw metrics from last 24h
        client = DatadogClient()

        # Training FoMs
        fom_metrics.update(self._training_foms(client))

        # CI FoMs
        fom_metrics.update(self._ci_foms(client))

        # Eval FoMs
        fom_metrics.update(self._eval_foms(client))

        return fom_metrics

    def _training_foms(self, client: DatadogClient) -> dict[str, float]:
        """Calculate training-related FoMs."""
        foms = {}

        # Example: Multigpu Arena Hearts
        hearts = client.query_metric("wandb.training.arena.avg_hearts")
        if hearts is not None:
            foms["health.training.multigpu_arena_hearts.fom"] = self._clip(
                (hearts - 0.0) / (0.7 - 0.0), 0, 1
            )

        # ... more training FoMs

        return foms

    def _ci_foms(self, client: DatadogClient) -> dict[str, float]:
        """Calculate CI-related FoMs."""
        foms = {}

        # Example: Tests Passing (already exists!)
        tests_passing = client.query_metric("github.ci.tests_passing_on_main")
        if tests_passing is not None:
            foms["health.ci.tests_passing.fom"] = 1.0 if tests_passing else 0.0

        # Example: Hotfix Count (already exists!)
        hotfixes = client.query_metric("github.commits.hotfix")
        if hotfixes is not None:
            foms["health.ci.hotfix_count.fom"] = max(1.0 - (hotfixes / 10.0), 0.0)

        # ... more CI FoMs

        return foms

    def _eval_foms(self, client: DatadogClient) -> dict[str, float]:
        """Calculate eval-related FoMs."""
        # ... eval FoMs
        return {}

    @staticmethod
    def _clip(value: float, min_val: float, max_val: float) -> float:
        """Clip value to [min_val, max_val] range."""
        return max(min_val, min(value, max_val))
```

### Deployment Configuration

Add to Helm values:

```yaml
# devops/charts/datadog-collectors/values.yaml

collectors:
  # ... existing collectors ...

  health_fom:
    enabled: true
    schedule: "0 * * * *"  # Every hour
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
    env:
      DATADOG_QUERY_API_KEY: "from-secrets"
```

## Dashboard Implementation

### Dashboard JSON Structure

Create dashboard using existing tools:

```bash
# 1. Create dashboard JSON
vim devops/datadog/templates/system_health.json

# 2. Build (if using Jsonnet)
metta datadog dashboard build

# 3. Push to Datadog
metta datadog dashboard push
```

### Query Table Widget Configuration

```json
{
  "title": "System Health - 7 Day Rolling",
  "type": "query_table",
  "requests": [{
    "queries": [
      {
        "name": "training_multigpu_hearts",
        "data_source": "metrics",
        "query": "avg:health.training.multigpu_arena_hearts.fom{*} by {day}"
      },
      {
        "name": "ci_tests_passing",
        "data_source": "metrics",
        "query": "avg:health.ci.tests_passing.fom{*} by {day}"
      }
      // ... more queries for each metric
    ],
    "formulas": [{
      "formula": "query1",
      "cell_display_mode": "bar",
      "conditional_formats": [
        {
          "comparator": ">=",
          "value": 0.7,
          "palette": "white_on_green"
        },
        {
          "comparator": ">=",
          "value": 0.3,
          "palette": "white_on_yellow"
        },
        {
          "comparator": "<",
          "value": 0.3,
          "palette": "white_on_red"
        }
      ]
    }]
  }]
}
```

### Color Spectrum

Standard color scale across all FoM metrics:

| FoM Range | Color | Meaning | Status |
|-----------|-------|---------|--------|
| 0.0 - 0.3 | 🔴 Red | Critical | Action required |
| 0.3 - 0.7 | 🟡 Yellow | Warning | Needs attention |
| 0.7 - 1.0 | 🟢 Green | Healthy | All good |

Datadog conditional formatting:
```json
{
  "conditional_formats": [
    {"comparator": ">=", "value": 0.7, "palette": "custom_bg", "custom_bg_color": "#28A745"},
    {"comparator": ">=", "value": 0.3, "palette": "custom_bg", "custom_bg_color": "#FFC107"},
    {"comparator": "<",  "value": 0.3, "palette": "custom_bg", "custom_bg_color": "#DC3545"}
  ]
}
```

## Implementation Plan

### Phase 1: Leverage Existing CI Metrics (Week 1)

**Goal**: Build initial dashboard with CI/CD metrics we already have

**Tasks**:
1. Create FoM processing collector skeleton
2. Implement CI FoM calculations using existing GitHub metrics:
   - `health.ci.tests_passing.fom` ← `github.ci.tests_passing_on_main`
   - `health.ci.failing_workflows.fom` ← `github.ci.failed_workflows_7d`
   - `health.ci.hotfix_count.fom` ← `github.commits.hotfix`
   - `health.ci.revert_count.fom` ← `github.commits.reverts`
   - `health.ci.duration_p90.fom` ← `github.ci.duration_p90_minutes`
   - `health.ci.stale_prs.fom` ← `github.prs.stale_count_14d`
   - `health.ci.pr_cycle_time.fom` ← `github.prs.cycle_time_hours`
3. Deploy FoM collector CronJob (hourly schedule)
4. Create initial dashboard with 7 CI metrics
5. Validate FoM calculations

**Commands**:
```bash
# Create collector
mkdir -p devops/datadog/collectors/health_fom
# ... implement collector.py

# Test locally
metta datadog collect health_fom

# Deploy
cd devops/charts
helm upgrade --install datadog-collectors ./datadog-collectors \
  --set collectors.health_fom.enabled=true

# Create dashboard
vim devops/datadog/templates/system_health.json
metta datadog dashboard push
```

### Phase 2: Extend GitHub Collector (Week 2)

**Goal**: Add missing CI metrics to GitHub collector

**New Metrics Needed**:
- `github.ci.benchmarks_passing` - Benchmark test status
- `github.commits.force_merge_7d` - Force merge count
- `github.ci.timeout_cancellations_7d` - Timeout cancellations
- `github.ci.flaky_checks_7d` - Flaky check retries

**Implementation**:
```bash
# See: devops/datadog/docs/ADDING_NEW_COLLECTOR.md
# (but extending existing collector)

# 1. Add metrics to collectors/github/metrics.py
vim devops/datadog/collectors/github/metrics.py

# 2. Test locally
metta datadog collect github

# 3. Deploy updated collector
# (automatic on next CronJob run)

# 4. Update FoM collector to use new metrics
vim devops/datadog/collectors/health_fom/collector.py

# 5. Update dashboard
metta datadog dashboard build && metta datadog dashboard push
```

### Phase 3: Add WandB Collector (Week 3-4)

**Goal**: Implement WandB collector for training metrics

**See**: `devops/datadog/docs/ADDING_NEW_COLLECTOR.md`

**New Collector**:
- Location: `devops/datadog/collectors/wandb/`
- Metrics: 15+ training metrics
- Schedule: Every 15 minutes

**Key Metrics**:
- `wandb.training.arena.runs_completed_24h`
- `wandb.training.arena.avg_hearts`
- `wandb.training.arena.avg_sps`
- `wandb.training.multinode.runs_completed_24h`
- `wandb.training.multinode.avg_hearts`
- `wandb.training.multinode.shaped_reward_sps`

**Implementation**:
```bash
# Follow standard collector pattern
# See: WORKPLAN.md "WandB Collector (Highest Priority)"

# 1. Create collector structure
mkdir -p devops/datadog/collectors/wandb
touch devops/datadog/collectors/wandb/{__init__.py,collector.py,metrics.py,README.md}

# 2. Implement collector (see ADDING_NEW_COLLECTOR.md)

# 3. Add secrets to AWS Secrets Manager
aws secretsmanager create-secret \
  --name wandb/api-key \
  --secret-string "your-wandb-api-key"

# 4. Deploy
cd devops/charts
helm upgrade --install datadog-collectors ./datadog-collectors \
  --set collectors.wandb.enabled=true

# 5. Update FoM collector
# 6. Update dashboard
```

### Phase 4: Add Training Jobs Collector (Week 5)

**Goal**: Collect metrics from local training jobs

**New Collector**:
- Location: `devops/datadog/collectors/training_jobs/`
- Metrics: Checkpoint success, resume success
- Schedule: Every 30 minutes

**Metrics**:
- `training.local.checkpoint_success_24h`
- `training.local.resume_success_24h`

**Implementation Options**:
1. Parse training logs from Skypilot jobs
2. Emit metrics directly from training code
3. Query training status API

### Phase 5: Add Eval Collector (Week 6)

**Goal**: Collect evaluation metrics

**New Collector**:
- Location: `devops/datadog/collectors/eval/`
- Metrics: Local eval, remote eval status
- Schedule: Every 15 minutes

**Metrics**:
- `eval.local.success_24h`
- `eval.local.hearts_accuracy_pct`
- `eval.remote.success_24h`
- `eval.remote.hearts_accuracy_pct`
- `eval.remote.duration_minutes`

### Phase 6: Polish & Alerting (Week 7)

**Goal**: Production-ready dashboard with alerts

**Tasks**:
1. Add all remaining metrics to dashboard
2. Configure Datadog monitors for critical FoMs
3. Set up alert channels (Slack, PagerDuty)
4. Create runbooks for common issues
5. Team training on dashboard usage

**Monitors**:
```yaml
# Critical: Tests failing on main
Alert when: health.ci.tests_passing.fom < 0.5
Severity: Critical
Channel: #alerts-critical

# Warning: Training performance degraded
Alert when: health.training.multigpu_arena_hearts.fom < 0.5
Severity: Warning
Channel: #alerts-training

# Warning: High hotfix rate
Alert when: health.ci.hotfix_count.fom < 0.5
Severity: Warning
Channel: #alerts-ci
```

## Testing Strategy

### Unit Tests

Test FoM calculations:

```python
# devops/datadog/tests/collectors/test_health_fom.py

def test_hearts_fom_calculation():
    """Test hearts FoM formula."""
    collector = HealthFomCollector()

    # Test boundary conditions
    assert collector._calculate_hearts_fom(0.0) == 0.0
    assert collector._calculate_hearts_fom(0.7) == 1.0
    assert 0.7 < collector._calculate_hearts_fom(0.5) < 0.8

def test_binary_fom():
    """Test binary FoM (tests passing)."""
    collector = HealthFomCollector()

    assert collector._calculate_binary_fom(True) == 1.0
    assert collector._calculate_binary_fom(False) == 0.0
```

### Integration Tests

Test end-to-end flow:

```bash
# 1. Run all collectors
metta datadog collect github --push
metta datadog collect wandb --push
metta datadog collect health_fom --push

# 2. Verify FoM metrics in Datadog
# (check via Datadog UI or API)

# 3. Verify dashboard displays correctly
# (manual verification)
```

## Maintenance

### Regular Reviews

**Weekly**:
- Review FoM values for anomalies
- Check for missing data (gray cells)
- Verify collector health metrics

**Monthly**:
- Adjust FoM formulas based on team feedback
- Add/remove metrics as needed
- Update alert thresholds

### Collector Health

Monitor FoM collector itself:

```yaml
# Alert if FoM collector fails
Query: avg(last_30m):sum:collector.health_fom.error_count{} > 3
Alert: FoM collector experiencing errors
```

## Success Criteria

### Phase 1 (CI Metrics - Week 1)
- ✅ FoM collector deployed and running hourly
- ✅ 7 CI FoM metrics calculated correctly
- ✅ Dashboard displaying 7-day view with color coding
- ✅ No missing data for CI metrics

### Phase 3 (Training Metrics - Week 4)
- ✅ WandB collector deployed
- ✅ 9 training FoM metrics calculated
- ✅ Dashboard showing full training + CI view
- ✅ Team using dashboard daily

### Phase 6 (Complete - Week 7)
- ✅ All 25+ metrics in dashboard
- ✅ Alerts configured and tested
- ✅ < 5% missing data (gray cells)
- ✅ Team trained on dashboard
- ✅ Runbooks documented

## Open Questions

1. **WandB API Access**: Do we have API keys? Rate limits?
   - Action: Check with team about WandB access

2. **Training Job Metrics**: Best way to collect checkpoint/resume status?
   - Option A: Parse Skypilot logs
   - Option B: Emit from training code
   - Option C: New training status API

3. **Dashboard Refresh Rate**: Hourly FoM updates sufficient?
   - Consider: Real-time critical metrics vs. historical trends

4. **Alert Fatigue**: How many monitors to start with?
   - Start conservative: Only critical alerts
   - Expand based on incidents

## References

### Existing Documentation
- [Collectors Architecture](COLLECTORS_ARCHITECTURE.md) - System design
- [Adding New Collector](ADDING_NEW_COLLECTOR.md) - Implementation guide
- [Metric Conventions](METRIC_CONVENTIONS.md) - Naming standards
- [CI/CD Metrics](CI_CD_METRICS.md) - GitHub metric catalog
- [Work Plan](../WORKPLAN.md) - Project roadmap

### Datadog Resources
- [Query Table Widget Docs](https://docs.datadoghq.com/dashboards/widgets/table/)
- [Conditional Formatting](https://docs.datadoghq.com/dashboards/guide/custom_time_frames/)
- [Dashboard JSON API](https://docs.datadoghq.com/api/latest/dashboards/)

### Team Resources
- CLI: `metta datadog --help`
- Slack: #datadog-monitoring
- Owner: DevOps team

---

**Last Updated**: 2025-10-23
**Status**: Planning phase
**Next Steps**: Phase 1 - Implement FoM collector with existing CI metrics
