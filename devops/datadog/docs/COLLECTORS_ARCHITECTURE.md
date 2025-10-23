# Data Collectors Architecture

## Vision

A scalable, modular system for collecting metrics from multiple services (GitHub, Skypilot, WandB, EC2, Asana, etc.) and submitting them to Datadog. Each collector is a self-contained module that knows how to:
1. Authenticate with its service
2. Fetch relevant data via API
3. Transform data into Datadog metrics
4. Handle errors gracefully

All collectors run as scheduled Kubernetes CronJobs, deployed via Helm charts.

## Directory Structure

```
devops/datadog/
├── collectors/                    # Data collector modules
│   ├── __init__.py
│   ├── base.py                    # Base collector class and utilities
│   ├── github/                    # GitHub metrics collector
│   │   ├── __init__.py
│   │   ├── collector.py           # Main collector implementation
│   │   ├── metrics.py             # Metric definitions
│   │   └── README.md              # Collector-specific documentation
│   ├── skypilot/                  # Skypilot metrics collector
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   ├── metrics.py
│   │   └── README.md
│   ├── wandb/                     # Weights & Biases collector
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   ├── metrics.py
│   │   └── README.md
│   ├── ec2/                       # AWS EC2 metrics collector
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   ├── metrics.py
│   │   └── README.md
│   └── asana/                     # Asana project tracking collector
│       ├── __init__.py
│       ├── collector.py
│       ├── metrics.py
│       └── README.md
│
├── common/                        # Shared utilities
│   ├── __init__.py
│   ├── datadog_client.py          # Datadog submission client
│   ├── secrets.py                 # Secrets management
│   ├── registry.py                # Collector registry
│   └── decorators.py              # Metric decorators
│
├── charts/                        # Helm charts
│   ├── collector-cronjobs/        # Unified CronJob chart
│   │   ├── Chart.yaml
│   │   ├── values.yaml            # Default values
│   │   ├── values-production.yaml # Production overrides
│   │   ├── values-staging.yaml    # Staging overrides
│   │   └── templates/
│   │       ├── cronjob.yaml       # CronJob template (one per collector)
│   │       ├── configmap.yaml     # Configuration
│   │       └── serviceaccount.yaml
│   └── collector-image/           # Docker image for all collectors
│       └── Dockerfile
│
├── scripts/                       # Management scripts
│   ├── run_collector.py           # CLI to run any collector
│   ├── list_collectors.py         # List available collectors
│   └── test_collector.py          # Test collector locally
│
├── docs/                          # Documentation
│   ├── COLLECTORS_ARCHITECTURE.md (this file)
│   ├── ADDING_NEW_COLLECTOR.md    # Step-by-step guide
│   ├── SECRETS_MANAGEMENT.md      # Secrets strategy
│   └── DEPLOYMENT.md              # Helm deployment guide
│
└── tests/                         # Tests
    └── collectors/
        ├── test_base.py
        ├── test_github.py
        ├── test_skypilot.py
        └── ...
```

## Architecture Principles

### 1. Single Responsibility
Each collector focuses on one service and knows nothing about others.

### 2. Common Interface
All collectors implement the same base interface for consistency.

### 3. Declarative Metrics
Metrics are declared using decorators, automatically registered and collected.

### 4. Fail-Safe
Individual metric failures don't crash the collector; errors are logged and reported.

### 5. Configuration-Driven
Collectors are configured via environment variables and config files, not hardcoded values.

### 6. Observable
All collectors emit metadata about their own health (collection time, error counts, metric counts).

## Base Collector Pattern

### Abstract Base Class

```python
# devops/datadog/collectors/base.py

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
import logging

from devops.datadog.common.datadog_client import DatadogClient
from devops.datadog.common.registry import MetricRegistry

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Base class for all data collectors.

    Subclasses must implement:
    - name: Unique collector identifier
    - collect_metrics(): Fetch and return metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.registry = MetricRegistry()
        self._errors: list[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this collector (e.g., 'github', 'skypilot')."""
        pass

    @property
    def source_tag(self) -> str:
        """Source tag for Datadog metrics."""
        return f"source:{self.name}-collector"

    @abstractmethod
    def collect_metrics(self) -> dict[str, float]:
        """
        Collect metrics from the service.

        Returns:
            Dictionary of metric_name -> value

        Raises:
            CollectorError: If collection fails critically
        """
        pass

    def run(self, push: bool = True) -> dict[str, float]:
        """
        Execute the collector: fetch metrics and optionally push to Datadog.

        Args:
            push: Whether to submit metrics to Datadog

        Returns:
            Collected metrics dictionary
        """
        logger.info(f"Running {self.name} collector")
        start_time = datetime.now(timezone.utc)

        try:
            # Collect metrics
            metrics = self.collect_metrics()

            # Add collector health metrics
            collection_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            metrics[f"collector.{self.name}.duration_seconds"] = collection_duration
            metrics[f"collector.{self.name}.metric_count"] = len(metrics) - 2  # Exclude health metrics
            metrics[f"collector.{self.name}.error_count"] = len(self._errors)

            logger.info(f"{self.name} collector: Collected {len(metrics)} metrics in {collection_duration:.2f}s")

            if push:
                self._push_to_datadog(metrics)

            return metrics

        except Exception as e:
            logger.error(f"{self.name} collector failed: {e}", exc_info=True)
            # Push error metric
            error_metrics = {
                f"collector.{self.name}.failed": 1,
                f"collector.{self.name}.duration_seconds":
                    (datetime.now(timezone.utc) - start_time).total_seconds(),
            }
            if push:
                self._push_to_datadog(error_metrics)
            raise

    def _push_to_datadog(self, metrics: dict[str, float]) -> None:
        """Submit metrics to Datadog."""
        client = DatadogClient(
            source=self.source_tag,
            env=self.config.get("env", "production"),
            service=f"{self.name}-collector",
        )
        client.submit_metrics(metrics)
        logger.info(f"Pushed {len(metrics)} metrics to Datadog")

    def _record_error(self, metric_name: str, error: Exception) -> None:
        """Record a metric collection error."""
        error_msg = f"{metric_name}: {error}"
        self._errors.append(error_msg)
        logger.error(f"Metric collection error - {error_msg}")


class CollectorError(Exception):
    """Raised when collector encounters a critical error."""
    pass
```

### Metric Decorator

```python
# devops/datadog/common/decorators.py

from functools import wraps
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


def metric(
    metric_key: str,
    unit: str | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a metric collector function.

    Args:
        metric_key: Datadog metric name (e.g., "github.prs.open")
        unit: Metric unit (optional, for documentation)
        tags: Additional tags to apply (optional)

    Example:
        @metric("github.prs.open", unit="count")
        def get_open_prs() -> int:
            return len(fetch_prs(state="open"))
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Store metadata on function
        func._metric_key = metric_key  # type: ignore
        func._metric_unit = unit  # type: ignore
        func._metric_tags = tags or []  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                value = func(*args, **kwargs)
                if value is None:
                    logger.warning(f"Metric {metric_key} returned None, skipping")
                    return None
                return float(value)
            except Exception as e:
                logger.error(f"Error collecting metric {metric_key}: {e}")
                return None

        return wrapper
    return decorator
```

## Example Collector Implementation

### GitHub Collector

```python
# devops/datadog/collectors/github/collector.py

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.github import metrics
from devops.datadog.common.registry import auto_discover_metrics


class GitHubCollector(BaseCollector):
    """Collects metrics from GitHub API (PRs, commits, CI/CD, branches)."""

    @property
    def name(self) -> str:
        return "github"

    def collect_metrics(self) -> dict[str, float]:
        """Collect all GitHub metrics."""
        # Auto-discover all @metric decorated functions in metrics module
        metric_functions = auto_discover_metrics(metrics)

        collected = {}
        for metric_key, func in metric_functions.items():
            try:
                value = func()
                if value is not None:
                    collected[metric_key] = value
            except Exception as e:
                self._record_error(metric_key, e)

        return collected
```

```python
# devops/datadog/collectors/github/metrics.py

from datetime import datetime, timedelta, timezone

from gitta import get_pull_requests, get_branches, get_commits
from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret

REPO = "softmax-research/metta"


def _get_auth_header() -> str:
    """Get GitHub authentication header."""
    token = get_secret("github/dashboard-token")
    return f"Basic {token}"


@metric("github.prs.open", unit="count")
def get_open_prs() -> int:
    """Currently open pull requests."""
    prs = get_pull_requests(
        repo=REPO,
        state="open",
        Authorization=_get_auth_header(),
    )
    return len(prs)


@metric("github.prs.merged_7d", unit="count")
def get_merged_prs_7d() -> int:
    """Pull requests merged in last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    prs = get_pull_requests(
        repo=REPO,
        state="closed",
        since=since,
        Authorization=_get_auth_header(),
    )
    merged = [pr for pr in prs if pr.get("merged_at")]
    return len(merged)


# ... more metrics ...
```

### Skypilot Collector

```python
# devops/datadog/collectors/skypilot/collector.py

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.skypilot import metrics
from devops.datadog.common.registry import auto_discover_metrics


class SkypilotCollector(BaseCollector):
    """Collects metrics from Skypilot API (jobs, clusters, costs)."""

    @property
    def name(self) -> str:
        return "skypilot"

    def collect_metrics(self) -> dict[str, float]:
        """Collect all Skypilot metrics."""
        metric_functions = auto_discover_metrics(metrics)

        collected = {}
        for metric_key, func in metric_functions.items():
            try:
                value = func()
                if value is not None:
                    collected[metric_key] = value
            except Exception as e:
                self._record_error(metric_key, e)

        return collected
```

```python
# devops/datadog/collectors/skypilot/metrics.py

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret
import httpx


@metric("skypilot.jobs.running", unit="count")
def get_running_jobs() -> int:
    """Number of currently running Skypilot jobs."""
    api_url = get_secret("skypilot/api-url")
    api_key = get_secret("skypilot/api-key")

    response = httpx.get(
        f"{api_url}/jobs",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"status": "running"},
    )
    response.raise_for_status()
    return len(response.json())


@metric("skypilot.compute.total_cost_7d", unit="usd")
def get_total_cost_7d() -> float:
    """Total compute cost in last 7 days (USD)."""
    # Implementation...
    pass


# ... more metrics ...
```

## Secrets Management

### Strategy

All service credentials stored in AWS Secrets Manager with standardized naming:

```
{service}/{credential-type}

Examples:
- github/dashboard-token
- github/api-key
- skypilot/api-url
- skypilot/api-key
- wandb/api-key
- ec2/access-key-id
- ec2/secret-access-key
- asana/personal-access-token
- datadog/api-key
- datadog/app-key
```

### Access Pattern

```python
# devops/datadog/common/secrets.py

import boto3
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_secret(secret_name: str, region: str = "us-east-1") -> str:
    """
    Retrieve secret from AWS Secrets Manager.

    Args:
        secret_name: Secret name (e.g., "github/dashboard-token")
        region: AWS region

    Returns:
        Secret value as string

    Raises:
        SecretNotFoundError: If secret doesn't exist
    """
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return response["SecretString"].strip()
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {e}")
        raise SecretNotFoundError(f"Secret {secret_name} not found") from e


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass
```

### IAM Permissions

Each collector's ServiceAccount needs:

```yaml
# Kubernetes ServiceAccount with IAM role annotation
apiVersion: v1
kind: ServiceAccount
metadata:
  name: collector-sa
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/collector-role

# IAM Policy for the role
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:github/*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:skypilot/*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:wandb/*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:ec2/*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:asana/*",
        "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:datadog/*"
      ]
    }
  ]
}
```

## Helm Chart Architecture

### Single Unified Chart

One Helm chart deploys all collectors as separate CronJobs.

```yaml
# charts/collector-cronjobs/values.yaml

# Global settings
global:
  image:
    registry: 751442549699.dkr.ecr.us-east-1.amazonaws.com
    repository: datadog-collectors
    tag: "latest"
    pullPolicy: Always

  datadog:
    env: "production"
    site: "datadoghq.com"

  serviceAccount:
    create: true
    annotations:
      eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/datadog-collectors

# Collector-specific settings
collectors:
  github:
    enabled: true
    schedule: "*/15 * * * *"  # Every 15 minutes
    resources:
      requests:
        cpu: 100m
        memory: 512Mi
      limits:
        cpu: 500m
        memory: 1Gi
    env:
      GITHUB_ORG: "softmax-research"
      GITHUB_REPO: "metta"

  skypilot:
    enabled: true
    schedule: "*/10 * * * *"  # Every 10 minutes
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 300m
        memory: 512Mi

  wandb:
    enabled: true
    schedule: "*/30 * * * *"  # Every 30 minutes
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 300m
        memory: 512Mi

  ec2:
    enabled: true
    schedule: "*/5 * * * *"  # Every 5 minutes
    resources:
      requests:
        cpu: 50m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi

  asana:
    enabled: false  # Disabled by default
    schedule: "0 */6 * * *"  # Every 6 hours
    resources:
      requests:
        cpu: 50m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi

# Common CronJob settings
cronJob:
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid  # Prevent overlapping runs
  restartPolicy: OnFailure
  backoffLimit: 2
```

### CronJob Template

```yaml
# charts/collector-cronjobs/templates/cronjob.yaml

{{- range $name, $config := .Values.collectors }}
{{- if $config.enabled }}
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ $name }}-collector
  labels:
    app: datadog-collector
    collector: {{ $name }}
spec:
  schedule: {{ $config.schedule | quote }}
  successfulJobsHistoryLimit: {{ $.Values.cronJob.successfulJobsHistoryLimit }}
  failedJobsHistoryLimit: {{ $.Values.cronJob.failedJobsHistoryLimit }}
  concurrencyPolicy: {{ $.Values.cronJob.concurrencyPolicy }}
  jobTemplate:
    spec:
      backoffLimit: {{ $.Values.cronJob.backoffLimit }}
      template:
        metadata:
          labels:
            app: datadog-collector
            collector: {{ $name }}
        spec:
          serviceAccountName: {{ $.Values.global.serviceAccount.name }}
          restartPolicy: {{ $.Values.cronJob.restartPolicy }}
          containers:
          - name: collector
            image: "{{ $.Values.global.image.registry }}/{{ $.Values.global.image.repository }}:{{ $.Values.global.image.tag }}"
            imagePullPolicy: {{ $.Values.global.image.pullPolicy }}
            command: ["python", "-m", "devops.datadog.scripts.run_collector"]
            args: ["{{ $name }}", "--push"]
            env:
            - name: DD_ENV
              value: {{ $.Values.global.datadog.env | quote }}
            - name: DD_SITE
              value: {{ $.Values.global.datadog.site | quote }}
            - name: DD_SERVICE
              value: "{{ $name }}-collector"
            {{- range $key, $value := $config.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
            resources:
              {{- toYaml $config.resources | nindent 14 }}
{{- end }}
{{- end }}
```

## CLI Tool

### Run Collector Script

```python
# devops/datadog/scripts/run_collector.py

#!/usr/bin/env python3
"""
Run a specific data collector.

Usage:
    python -m devops.datadog.scripts.run_collector <collector_name> [--push]

Examples:
    python -m devops.datadog.scripts.run_collector github
    python -m devops.datadog.scripts.run_collector skypilot --push
"""

import argparse
import importlib
import sys
import logging

from devops.datadog.common.log_config import init_logging

logger = logging.getLogger(__name__)


def load_collector(name: str):
    """Dynamically load collector class by name."""
    try:
        module = importlib.import_module(f"devops.datadog.collectors.{name}.collector")
        # Find the collector class (should be NameCollector)
        collector_class_name = f"{name.capitalize()}Collector"
        collector_class = getattr(module, collector_class_name)
        return collector_class()
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load collector '{name}': {e}")
        raise ValueError(f"Collector '{name}' not found") from e


def main():
    parser = argparse.ArgumentParser(description="Run a Datadog metrics collector")
    parser.add_argument("collector", help="Collector name (e.g., github, skypilot)")
    parser.add_argument("--push", action="store_true", help="Push metrics to Datadog")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    init_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        # Load and run collector
        collector = load_collector(args.collector)
        metrics = collector.run(push=args.push)

        # Print results
        print(f"\nCollected {len(metrics)} metrics:")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value}")

        if not args.push:
            print("\n(Dry run - metrics not pushed to Datadog)")
        else:
            print("\nMetrics pushed to Datadog successfully")

        return 0

    except Exception as e:
        logger.error(f"Collector failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

## Testing Strategy

### Unit Tests

```python
# devops/datadog/tests/collectors/test_github.py

import pytest
from unittest.mock import Mock, patch

from devops.datadog.collectors.github.collector import GitHubCollector
from devops.datadog.collectors.github import metrics


class TestGitHubCollector:
    def test_collector_name(self):
        collector = GitHubCollector()
        assert collector.name == "github"

    @patch("devops.datadog.collectors.github.metrics.get_pull_requests")
    def test_collect_metrics(self, mock_get_prs):
        mock_get_prs.return_value = [{"number": 1}, {"number": 2}]

        collector = GitHubCollector()
        metrics = collector.collect_metrics()

        assert "github.prs.open" in metrics
        assert metrics["github.prs.open"] == 2.0

    def test_error_handling(self):
        collector = GitHubCollector()

        # Mock a metric function that raises
        with patch.object(metrics, "get_open_prs", side_effect=Exception("API error")):
            result = collector.collect_metrics()

            # Collector should not crash
            assert len(collector._errors) == 1
            assert "API error" in collector._errors[0]
```

### Integration Tests

```python
# devops/datadog/tests/test_integration.py

import pytest
from devops.datadog.scripts.run_collector import load_collector


@pytest.mark.integration
def test_github_collector_live():
    """Test GitHub collector against real API (requires credentials)."""
    collector = load_collector("github")
    metrics = collector.run(push=False)

    # Verify we got some metrics
    assert len(metrics) > 0

    # Verify expected metrics exist
    assert "github.prs.open" in metrics
    assert "github.commits.total_7d" in metrics

    # Verify collector health metrics
    assert "collector.github.duration_seconds" in metrics
    assert "collector.github.metric_count" in metrics
```

## Deployment Workflow

### 1. Local Development

```bash
# Test collector locally
python -m devops.datadog.scripts.run_collector github --verbose

# Run tests
pytest devops/datadog/tests/collectors/test_github.py -v
```

### 2. Build Docker Image

```bash
# Build and push image
cd devops/datadog
docker build -t 751442549699.dkr.ecr.us-east-1.amazonaws.com/datadog-collectors:latest -f charts/collector-image/Dockerfile .
docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/datadog-collectors:latest
```

### 3. Deploy to Kubernetes

```bash
# Deploy all collectors
helm upgrade --install datadog-collectors \
  ./charts/collector-cronjobs \
  --namespace monitoring \
  --create-namespace \
  --values ./charts/collector-cronjobs/values-production.yaml

# Enable specific collector
helm upgrade --install datadog-collectors \
  ./charts/collector-cronjobs \
  --namespace monitoring \
  --set collectors.wandb.enabled=true

# Check CronJob status
kubectl get cronjobs -n monitoring
kubectl get jobs -n monitoring
kubectl logs -n monitoring -l collector=github --tail=100
```

## Monitoring Collector Health

### Self-Monitoring Metrics

Every collector automatically emits:

- `collector.{name}.duration_seconds` - Collection time
- `collector.{name}.metric_count` - Number of metrics collected
- `collector.{name}.error_count` - Number of errors encountered
- `collector.{name}.failed` - Boolean indicating total failure (1 = failed, absent = success)

### Datadog Monitors

Create monitors for collector health:

```yaml
# Monitor: GitHub collector not running
Query: avg(last_15m):sum:collector.github.metric_count{env:production} < 1
Alert: GitHub collector hasn't reported metrics in 15 minutes

# Monitor: High error rate
Query: avg(last_1h):sum:collector.github.error_count{env:production} > 5
Alert: GitHub collector experiencing high error rate

# Monitor: Slow collection
Query: avg(last_30m):avg:collector.github.duration_seconds{env:production} > 60
Alert: GitHub collector taking >60s to run
```

## Adding a New Collector

See `docs/ADDING_NEW_COLLECTOR.md` for step-by-step guide.

Quick checklist:
1. Create `devops/datadog/collectors/{name}/` directory
2. Implement `collector.py` (subclass `BaseCollector`)
3. Define metrics in `metrics.py` using `@metric` decorator
4. Add secrets to AWS Secrets Manager (`{name}/*`)
5. Add collector config to `charts/collector-cronjobs/values.yaml`
6. Write tests in `tests/collectors/test_{name}.py`
7. Document in `collectors/{name}/README.md`
8. Deploy with Helm

## Migration Plan

### Phase 1: Refactor Existing (Current)

Move current GitHub metrics to new structure:
- Create `devops/datadog/collectors/github/`
- Migrate `softmax/dashboard/metrics.py` → `collectors/github/metrics.py`
- Create `collectors/github/collector.py`
- Update CronJob to use new structure
- Verify metrics still flowing

### Phase 2: Add New Collectors

Add collectors one at a time:
1. Skypilot (job tracking, compute costs)
2. WandB (training runs, experiment metrics)
3. EC2 (instance counts, costs, utilization)
4. Asana (project velocity, task completion)

### Phase 3: Enhanced Features

- Dashboard auto-generation from metric metadata
- Collector dependency graph (some collectors depend on others)
- Metric validation and schema enforcement
- Historical data backfill tools

## Benefits of This Architecture

1. **Scalability**: Easy to add new collectors without modifying existing ones
2. **Maintainability**: Each collector is self-contained with clear boundaries
3. **Testability**: Mock service APIs for testing without credentials
4. **Observability**: Every collector monitors itself
5. **Flexibility**: Different schedules, resources, and configs per collector
6. **Safety**: Errors in one collector don't affect others
7. **Consistency**: All collectors follow same patterns and conventions

## Open Questions

1. **Rate Limiting**: Should we implement global rate limiting across collectors?
2. **Data Retention**: How long should we keep collector logs and job history?
3. **Alerting**: Should collectors alert on their own failures or rely on Datadog monitors?
4. **Dependencies**: How to handle collectors that depend on data from other collectors?
5. **Backfill**: Do we need tools to backfill historical data when adding new metrics?

## Next Steps

1. Review and approve architecture
2. Create `ADDING_NEW_COLLECTOR.md` step-by-step guide
3. Migrate existing GitHub metrics to new structure
4. Implement first new collector (Skypilot or WandB)
5. Document deployment process
6. Set up monitoring and alerts
