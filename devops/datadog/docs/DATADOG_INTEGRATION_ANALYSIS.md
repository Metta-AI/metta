# Datadog Integration Analysis

## Overview

This document analyzes where and how we push data to Datadog across the Metta codebase.

## Current Architecture

### 1. Metric Collection and Submission

**Location**: `softmax/src/softmax/dashboard/`

The metric submission system is centralized in the `softmax` package with three main components:

#### A. Registry Pattern (`registry.py`)
- **Purpose**: Central registry for metric collectors using decorator pattern
- **Key Functions**:
  - `@system_health_metric(metric_key="...")` - Decorator to register metric collectors
  - `get_system_health_metrics()` - Returns all registered metrics
  - `collect_metrics()` - Collects all metrics by calling registered functions

```python
# Example registration
@system_health_metric(metric_key="ci.tests_passing_on_main")
def get_latest_unit_tests_failed() -> int | None:
    # ... implementation
```

#### B. Metric Implementations (`metrics.py`)
Currently implemented metrics:
- `ci.tests_passing_on_main` - Main branch CI test status (from GitHub Actions API)
- `commits.hotfix` - Count of hotfix commits in last 7 days
- `commits.reverts` - Count of revert commits in last 7 days

**Data Sources**:
- GitHub API (via `gitta` library):
  - Workflow runs and job statuses
  - Commit history analysis
- AWS Secrets Manager for credentials:
  - `github/dashboard-token` - GitHub API access
  - `datadog/api-key` - Datadog API key
  - `datadog/app-key` - Datadog application key

#### C. Submission Tool (`report.py`)
- **Purpose**: CLI tool to collect and submit metrics
- **Implementation**: Uses Datadog API client v2
- **Metric Type**: GAUGE (all metrics are submitted as gauge values)
- **Tags Applied**:
  - `source:softmax-system-health`
  - `service:{DD_SERVICE}` (from config)
  - `env:{DD_ENV}` (from config)
  - `version:{DD_VERSION}` (from config, if set)

**Usage**:
```bash
# Collect metrics (dry run)
uv run python -m softmax.dashboard.report

# Collect and push to Datadog
uv run python -m softmax.dashboard.report --push

# Via metta CLI
metta softmax-system-health report --push
```

### 2. Scheduled Execution

**Location**: `devops/charts/dashboard-cronjob/values.yaml`

- **Schedule**: Every 15 minutes (`*/15 * * * *`)
- **Kubernetes CronJob**: Deployed to production cluster
- **Image**: `751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:latest`
- **Command**: `["uv", "run", "python", "-m", "softmax.dashboard.report", "--push"]`
- **Resources**:
  - Requests: 100m CPU, 2Gi memory
  - Limits: 500m CPU, 2Gi memory
- **Concurrency**: `Forbid` (prevents overlapping executions)
- **IAM Role**: `arn:aws:iam::751442549699:role/dashboard-cronjob`

### 3. APM Tracing Integration

**Location**: `common/src/metta/common/datadog/`

#### Configuration (`config.py`)
- **Purpose**: Centralized Datadog configuration
- **Settings**:
  - `DD_TRACE_ENABLED` - Enable/disable tracing (default: False)
  - `DD_SERVICE` - Service name (default: "metta")
  - `DD_ENV` - Environment (default: "development")
  - `DD_VERSION` - Service version (optional)
  - `DD_AGENT_HOST` - Datadog agent hostname (optional)
  - `DD_TRACE_AGENT_PORT` - Trace agent port (default: 8126)
  - `DD_TRACE_AGENT_URL` - Full trace agent URL (overrides host/port)
  - `DD_SITE` - Datadog site (default: "datadoghq.com")

#### Tracing (`tracing.py`)
- **Purpose**: APM tracing initialization and decorator
- **Implementation**: Uses `ddtrace` library
- **Key Functions**:
  - `init_tracing()` - Initialize tracer based on config
  - `@trace(name="...")` - Decorator for tracing functions (supports both sync and async)

**Usage Locations**:
- `app_backend/src/metta/app_backend/container_managers/docker.py` - Container management
- `app_backend/src/metta/app_backend/container_managers/k8s.py` - Kubernetes management
- `app_backend/src/metta/app_backend/eval_task_worker.py` - Evaluation task workers
- `app_backend/src/metta/app_backend/eval_task_orchestrator.py` - Task orchestration

### 4. Dashboard Configuration

**Location**: `devops/datadog/`

The dashboard configuration system is separate from metric submission:
- **Dashboards**: Defined in Jsonnet (`dashboards/*.jsonnet`)
- **Components**: Reusable widget collections (`components/*.libsonnet`)
- **Widgets**: Primitive widget builders (`lib/widgets.libsonnet`)
- **Push Tool**: `scripts/push_dashboard.py` - Uploads dashboard JSON to Datadog

**Note**: Dashboard configuration is about visualization, not metric submission.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Metric Collection                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  softmax/dashboard/metrics.py                              │
│  ├── @system_health_metric decorators                      │
│  ├── GitHub API calls (workflow status, commits)           │
│  └── Returns: int | None                                   │
│                                                             │
│  softmax/dashboard/registry.py                             │
│  ├── collect_metrics() → dict[str, float]                 │
│  └── Decorator registry pattern                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Metric Submission                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  softmax/dashboard/report.py                               │
│  ├── Uses Datadog API client v2                            │
│  ├── MetricsApi.submit_metrics()                           │
│  ├── Type: GAUGE                                           │
│  └── Tags: source, service, env, version                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Option 1: Manual CLI                                      │
│  └── metta softmax-system-health report --push            │
│                                                             │
│  Option 2: Kubernetes CronJob (Production)                │
│  ├── Schedule: Every 15 minutes                            │
│  ├── Image: softmax-dashboard:latest                       │
│  └── Command: report --push                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Datadog Services                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Metrics API (datadoghq.com)                               │
│  └── Receives: ci.*, commits.* metrics                     │
│                                                             │
│  APM Traces (separate integration)                         │
│  ├── app_backend service traces                            │
│  └── eval_task_* operation traces                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Centralization Assessment

### Current State: Already Well-Centralized ✅

The metric submission is already centralized with good separation of concerns:

1. **Single Registry**: All metrics registered in one place (`softmax/dashboard/registry.py`)
2. **Decorator Pattern**: Clean registration via `@system_health_metric`
3. **Single Submission Point**: All metrics submitted through `report.py`
4. **Shared Configuration**: Common Datadog config in `metta.common.datadog.config`
5. **Standard Tags**: Consistent tagging across all metrics

### Strengths

- **Extensibility**: Adding new metrics requires only:
  1. Write metric function
  2. Add `@system_health_metric` decorator
  3. Function automatically registered and submitted
- **Consistency**: All metrics use same API client, tags, and submission format
- **Separation**: Metric collection (business logic) separated from submission (infrastructure)
- **Testability**: Each metric function can be tested independently

### Potential Improvements

1. **Add More Metric Types**:
   - Currently only GAUGE metrics
   - Could support COUNT, RATE, HISTOGRAM if needed

2. **Metric Validation**:
   - Add schema validation for metric values
   - Type hints are good, but runtime validation would help

3. **Error Handling**:
   - Currently errors are logged but metrics are skipped
   - Could add retry logic for transient failures
   - Could track metric collection success/failure rates

4. **Local Development**:
   - Add local metric sink for development
   - Mock Datadog API for testing
   - Dry-run mode already exists, could be enhanced

5. **APM Integration**:
   - Tracing is separate from metrics
   - Could consolidate configuration
   - Currently uses different libraries (ddtrace vs datadog_api_client)

## Recommendations

### Short Term (No Action Needed)

The current architecture is solid and doesn't require immediate changes. The system is:
- Centralized
- Well-structured
- Easy to extend
- Production-ready

### Medium Term (Optional Enhancements)

1. **Add Metric Documentation**:
   - Create registry of all metrics with descriptions
   - Document expected values and units
   - Add to `softmax/dashboard/README.md`

2. **Monitoring Metrics**:
   - Add meta-metrics about metric collection
   - Track: collection duration, success rate, API latency
   - Submit these to Datadog as well

3. **Configuration Unification**:
   - Consider consolidating `common/datadog/config.py` and `softmax/dashboard/report.py` configuration
   - Single source of truth for Datadog credentials and settings

### Long Term (If Needed)

1. **Support Multiple Backends**:
   - Abstract metric submission behind interface
   - Support Prometheus, CloudWatch, etc.
   - Keep Datadog as default

2. **Metric Aggregation**:
   - Add local aggregation before submission
   - Support high-frequency metrics with client-side rollup
   - Reduce API calls and costs

## Files Involved

### Metric Collection
- `softmax/src/softmax/dashboard/metrics.py` - Metric implementations
- `softmax/src/softmax/dashboard/registry.py` - Registration system
- `softmax/src/softmax/dashboard/report.py` - Submission tool
- `softmax/src/softmax/dashboard/README.md` - Documentation

### Configuration
- `common/src/metta/common/datadog/config.py` - Datadog configuration
- `common/src/metta/common/datadog/tracing.py` - APM tracing

### Deployment
- `devops/charts/dashboard-cronjob/values.yaml` - Kubernetes CronJob config
- `softmax/Dockerfile` - Container image build

### CLI Integration
- `metta/setup/metta_cli.py` - Metta CLI integration (line 22, 889)

### Dashboard Visualization (Separate System)
- `devops/datadog/dashboards/` - Dashboard definitions
- `devops/datadog/components/` - Widget collections
- `devops/datadog/lib/` - Widget primitives
- `devops/datadog/scripts/` - Dashboard management tools

## Conclusion

**The Datadog metric submission is already well-centralized.** The architecture follows best practices:
- Single responsibility principle
- Decorator pattern for extensibility
- Clean separation of concerns
- Consistent error handling
- Production-ready deployment

**No immediate restructuring is needed.** The system can be extended by adding new `@system_health_metric` decorated functions to `metrics.py`.

The APM tracing integration is separate but also centralized through `common/datadog/config.py` and `tracing.py`. This separation makes sense as they serve different purposes (metrics vs traces) and use different Datadog APIs.
