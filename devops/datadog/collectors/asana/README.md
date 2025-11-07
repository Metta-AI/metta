# Asana Collector

Collects project management and bug tracking metrics.

## Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `asana.projects` | gauge | `status:active\|on_track\|at_risk\|off_track` | Project counts by status |
| `asana.bugs` | gauge | `status:open`, `section:triage\|active\|backlog\|other`, `event:created\|completed`, `timeframe:7d\|30d` | Bug counts |
| `asana.bugs.age_days` | gauge | `metric:avg\|max` | Bug age statistics |

**Total**: 3 metric names with dimensional tags

## Configuration

```bash
# Secrets (AWS Secrets Manager)
asana/access-token          # Asana API token
asana/workspace-gid         # Workspace GID
asana/bugs-project-gid      # Bugs project GID (optional)

# Environment Variables
# (All configuration from secrets)
```

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py asana --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py asana --push
```

## Dashboard Queries

```python
# Bug backlog
sum:asana.bugs{section:backlog}

# Bugs created this week
sum:asana.bugs{event:created,timeframe:7d}

# Project health
sum:asana.projects{status:at_risk}
```
