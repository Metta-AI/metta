# GitHub Collector

Collects development metrics from GitHub API.

## Metrics

| Metric | Type | Tags | Description |
|--------|------|------|-------------|
| `github.prs` | gauge | `status:open\|merged`, `timeframe:7d` | Pull request counts |
| `github.prs.stale` | gauge | `threshold:14d` | PRs open >14 days |
| `github.pr.time_to_merge_hours` | gauge | `pr_id:{number}` | Time from creation to merge (per-PR) |
| `github.pr.comments` | gauge | `pr_id:{number}` | Comment count (per-PR) |
| `github.branches` | gauge | `type:active` | Active branches (excluding main) |
| `github.commits` | gauge | `timeframe:7d` | Total commits |
| `github.commits.special` | gauge | `type:hotfix\|revert\|force_merge`, `timeframe:7d` | Special commit types |
| `github.code.lines` | gauge | `type:added\|deleted`, `timeframe:7d` | Lines of code changed |
| `github.code.files_changed` | gauge | `timeframe:7d` | Unique files modified |
| `github.ci.tests` | gauge | `branch:main`, `status:latest` | Latest test status (1=pass, 0=fail) |
| `github.ci.benchmarks` | gauge | `branch:main`, `status:latest` | Latest benchmark status (1=pass, 0=fail) |
| `github.ci.runs` | gauge | `status:failed\|cancelled\|flaky`, `timeframe:7d` | Workflow run counts |
| `github.ci.duration_minutes` | gauge | `metric:mean\|p50\|p90\|p99` | Workflow duration percentiles |
| `github.developers` | gauge | `status:active`, `timeframe:7d` | Active developer count |
| `github.commits_per_developer` | gauge | `timeframe:7d` | Average commits per developer |

**Total**: 15 metric names with dimensional tags

## Configuration

```bash
# Secrets (AWS Secrets Manager)
github/dashboard-token  # GitHub token with repo access

# Environment Variables
GITHUB_ORG=PufferAI     # Organization name
GITHUB_REPO=metta       # Repository name
```

## Usage

```bash
# Test locally
uv run python devops/datadog/scripts/run_collector.py github --verbose

# Push to Datadog
uv run python devops/datadog/scripts/run_collector.py github --push
```

## Dashboard Queries

```python
# PR velocity
avg:github.pr.time_to_merge_hours{*}

# Commit quality
sum:github.commits.special{type:revert,timeframe:7d}

# CI health
sum:github.ci.runs{status:failed,timeframe:7d}

# Developer activity
avg:github.developers{status:active,timeframe:7d}
```
