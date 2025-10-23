# Adding a New Collector

This guide walks through adding a new data collector to the Datadog metrics system.

## Overview

Adding a new collector involves:
1. Creating the collector module
2. Defining metrics
3. Adding secrets
4. Configuring deployment
5. Testing
6. Documentation

**Time estimate**: 1-2 hours for a simple collector

## Step-by-Step Guide

### Step 1: Create Collector Directory

Create a new directory for your collector:

```bash
mkdir -p devops/datadog/collectors/{collector_name}
cd devops/datadog/collectors/{collector_name}
touch __init__.py collector.py metrics.py README.md
```

Replace `{collector_name}` with your service name (e.g., `skypilot`, `wandb`, `asana`).

### Step 2: Implement Collector Class

Create `collector.py`:

```python
# devops/datadog/collectors/{collector_name}/collector.py

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.{collector_name} import metrics
from devops.datadog.common.registry import auto_discover_metrics


class {CollectorName}Collector(BaseCollector):
    """
    Collects metrics from {Service Name}.

    Metrics collected:
    - {metric_category_1}: {description}
    - {metric_category_2}: {description}

    Requirements:
    - AWS Secret: {service}/{credential-name}
    - API Access: {permission requirements}
    """

    @property
    def name(self) -> str:
        return "{collector_name}"

    def collect_metrics(self) -> dict[str, float]:
        """Collect all {Service Name} metrics."""
        # Auto-discover all @metric decorated functions
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

**Example for Skypilot**:

```python
# devops/datadog/collectors/skypilot/collector.py

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.collectors.skypilot import metrics
from devops.datadog.common.registry import auto_discover_metrics


class SkypilotCollector(BaseCollector):
    """
    Collects metrics from Skypilot API.

    Metrics collected:
    - Job metrics: running, queued, completed, failed
    - Cluster metrics: active clusters, total nodes
    - Cost metrics: compute spend by time period

    Requirements:
    - AWS Secret: skypilot/api-key
    - AWS Secret: skypilot/api-url
    """

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

### Step 3: Define Metrics

Create `metrics.py` with your metric functions:

```python
# devops/datadog/collectors/{collector_name}/metrics.py

from datetime import datetime, timedelta, timezone
import httpx

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret


def _get_api_client() -> httpx.Client:
    """Create authenticated API client for {Service Name}."""
    api_key = get_secret("{service}/api-key")
    api_url = get_secret("{service}/api-url")

    return httpx.Client(
        base_url=api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        timeout=30.0,
    )


@metric("{service}.{category}.{metric_name}", unit="count")
def get_metric_1() -> int:
    """Description of what this metric measures."""
    with _get_api_client() as client:
        response = client.get("/endpoint")
        response.raise_for_status()
        data = response.json()
        return len(data["items"])


@metric("{service}.{category}.{metric_name}_7d", unit="count")
def get_metric_2() -> int:
    """Description of metric over 7-day window."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with _get_api_client() as client:
        response = client.get("/endpoint", params={"since": since})
        response.raise_for_status()
        return response.json()["count"]


@metric("{service}.{category}.avg_{metric_name}", unit="{unit}")
def get_metric_3() -> float | None:
    """Average calculation that might return None if no data."""
    with _get_api_client() as client:
        response = client.get("/stats")
        response.raise_for_status()
        data = response.json()

        if not data["values"]:
            return None

        return sum(data["values"]) / len(data["values"])
```

**Example for Skypilot**:

```python
# devops/datadog/collectors/skypilot/metrics.py

from datetime import datetime, timedelta, timezone
import httpx

from devops.datadog.common.decorators import metric
from devops.datadog.common.secrets import get_secret


def _get_api_client() -> httpx.Client:
    """Create authenticated Skypilot API client."""
    api_key = get_secret("skypilot/api-key")
    api_url = get_secret("skypilot/api-url")

    return httpx.Client(
        base_url=api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        timeout=30.0,
    )


@metric("skypilot.jobs.running", unit="count")
def get_running_jobs() -> int:
    """Number of currently running Skypilot jobs."""
    with _get_api_client() as client:
        response = client.get("/jobs", params={"status": "running"})
        response.raise_for_status()
        return len(response.json()["jobs"])


@metric("skypilot.jobs.queued", unit="count")
def get_queued_jobs() -> int:
    """Number of queued Skypilot jobs waiting to start."""
    with _get_api_client() as client:
        response = client.get("/jobs", params={"status": "queued"})
        response.raise_for_status()
        return len(response.json()["jobs"])


@metric("skypilot.compute.cost_7d", unit="usd")
def get_compute_cost_7d() -> float:
    """Total compute cost in last 7 days (USD)."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with _get_api_client() as client:
        response = client.get("/billing/costs", params={"since": since})
        response.raise_for_status()
        return response.json()["total_cost"]


@metric("skypilot.clusters.active", unit="count")
def get_active_clusters() -> int:
    """Number of active Skypilot clusters."""
    with _get_api_client() as client:
        response = client.get("/clusters", params={"status": "active"})
        response.raise_for_status()
        return len(response.json()["clusters"])


@metric("skypilot.jobs.avg_duration_minutes", unit="minutes")
def get_avg_job_duration() -> float | None:
    """Average job duration for jobs completed in last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with _get_api_client() as client:
        response = client.get("/jobs", params={"status": "completed", "since": since})
        response.raise_for_status()
        jobs = response.json()["jobs"]

        if not jobs:
            return None

        durations = []
        for job in jobs:
            start = datetime.fromisoformat(job["started_at"])
            end = datetime.fromisoformat(job["completed_at"])
            duration_minutes = (end - start).total_seconds() / 60
            durations.append(duration_minutes)

        return sum(durations) / len(durations)
```

### Metric Naming Convention

Follow this pattern: `{service}.{category}.{metric_name}`

**Examples**:
- `github.prs.open` - Current open PRs
- `github.prs.merged_7d` - PRs merged in 7 days
- `skypilot.jobs.running` - Running jobs
- `wandb.runs.active` - Active training runs
- `ec2.instances.running` - Running instances
- `asana.tasks.completed_7d` - Tasks completed

**Units** (for documentation):
- `count` - Discrete count
- `percent` - Percentage (0-100)
- `seconds`, `minutes`, `hours` - Time duration
- `bytes`, `kb`, `mb`, `gb` - Data size
- `usd` - US dollars
- `rate` - Per-second rate

### Step 4: Add Secrets to AWS

Store service credentials in AWS Secrets Manager:

```bash
# Add API key
aws secretsmanager create-secret \
  --name {service}/api-key \
  --secret-string "your-api-key-here" \
  --region us-east-1

# Add API URL
aws secretsmanager create-secret \
  --name {service}/api-url \
  --secret-string "https://api.service.com" \
  --region us-east-1
```

**Common secret patterns**:

| Service | Secrets Needed |
|---------|----------------|
| REST API | `{service}/api-key`, `{service}/api-url` |
| OAuth | `{service}/client-id`, `{service}/client-secret`, `{service}/access-token` |
| AWS Service | `{service}/access-key-id`, `{service}/secret-access-key` |
| Token Auth | `{service}/token` or `{service}/personal-access-token` |

### Step 5: Configure Helm Deployment

Add your collector to `devops/charts/datadog-collectors/values.yaml`:

```yaml
collectors:
  # ... existing collectors ...

  {collector_name}:
    enabled: true
    schedule: "*/15 * * * *"  # Adjust frequency as needed
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
    env:
      # Service-specific environment variables
      SERVICE_REGION: "us-east-1"
      # Add any non-secret config here
```

**Schedule recommendations**:
- High-frequency metrics (instance counts, job status): `*/5 * * * *` (every 5 min)
- Medium-frequency (costs, usage stats): `*/15 * * * *` (every 15 min)
- Low-frequency (historical aggregates): `0 * * * *` (hourly)
- Very low-frequency (billing, reports): `0 */6 * * *` (every 6 hours)

### Step 6: Write Tests

Create `devops/datadog/tests/collectors/test_{collector_name}.py`:

```python
# devops/datadog/tests/collectors/test_{collector_name}.py

import pytest
from unittest.mock import Mock, patch, MagicMock

from devops.datadog.collectors.{collector_name}.collector import {CollectorName}Collector
from devops.datadog.collectors.{collector_name} import metrics


class Test{CollectorName}Collector:
    """Test suite for {CollectorName} collector."""

    def test_collector_name(self):
        """Verify collector name is correct."""
        collector = {CollectorName}Collector()
        assert collector.name == "{collector_name}"

    def test_source_tag(self):
        """Verify Datadog source tag."""
        collector = {CollectorName}Collector()
        assert collector.source_tag == "source:{collector_name}-collector"

    @patch("devops.datadog.collectors.{collector_name}.metrics._get_api_client")
    def test_collect_metrics_success(self, mock_client):
        """Test successful metric collection."""
        # Mock API responses
        mock_response = Mock()
        mock_response.json.return_value = {"items": [1, 2, 3]}
        mock_response.raise_for_status = Mock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        collector = {CollectorName}Collector()
        result = collector.collect_metrics()

        # Verify metrics were collected
        assert "{service}.{category}.{metric}" in result
        assert result["{service}.{category}.{metric}"] > 0

    @patch("devops.datadog.collectors.{collector_name}.metrics._get_api_client")
    def test_collect_metrics_with_errors(self, mock_client):
        """Test metric collection handles API errors gracefully."""
        # Mock API failure
        mock_client.side_effect = Exception("API Error")

        collector = {CollectorName}Collector()
        result = collector.collect_metrics()

        # Collector should not crash, just record errors
        assert len(collector._errors) > 0

    @patch("devops.datadog.common.secrets.get_secret")
    def test_secrets_loading(self, mock_get_secret):
        """Test that secrets are loaded correctly."""
        mock_get_secret.side_effect = lambda name: {
            "{service}/api-key": "test-key",
            "{service}/api-url": "https://test.api",
        }[name]

        # Verify secrets are used
        with patch("httpx.Client") as mock_client:
            metrics._get_api_client()
            mock_client.assert_called_once()
```

**Example for Skypilot**:

```python
# devops/datadog/tests/collectors/test_skypilot.py

import pytest
from unittest.mock import Mock, patch, MagicMock

from devops.datadog.collectors.skypilot.collector import SkypilotCollector
from devops.datadog.collectors.skypilot import metrics


class TestSkypilotCollector:
    def test_collector_name(self):
        collector = SkypilotCollector()
        assert collector.name == "skypilot"

    @patch("devops.datadog.collectors.skypilot.metrics._get_api_client")
    def test_get_running_jobs(self, mock_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "jobs": [
                {"id": 1, "status": "running"},
                {"id": 2, "status": "running"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        result = metrics.get_running_jobs()
        assert result == 2

    @patch("devops.datadog.collectors.skypilot.metrics._get_api_client")
    def test_collect_all_metrics(self, mock_client):
        # Setup mock responses for all endpoints
        mock_response = Mock()
        mock_response.json.return_value = {
            "jobs": [],
            "clusters": [],
            "total_cost": 0.0,
        }
        mock_response.raise_for_status = Mock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        collector = SkypilotCollector()
        metrics_dict = collector.collect_metrics()

        # Verify expected metrics are present
        assert "skypilot.jobs.running" in metrics_dict
        assert "skypilot.clusters.active" in metrics_dict
```

### Step 7: Test Locally

Run the collector locally before deploying:

```bash
# Test without pushing to Datadog
python -m devops.datadog.scripts.run_collector {collector_name} --verbose

# Expected output:
# Running {collector_name} collector
# {collector_name} collector: Collected 5 metrics in 1.23s
#
# Collected 5 metrics:
#   {service}.{metric1}: 42.0
#   {service}.{metric2}: 100.0
#   collector.{collector_name}.duration_seconds: 1.23
#   collector.{collector_name}.metric_count: 3.0
#   collector.{collector_name}.error_count: 0.0
#
# (Dry run - metrics not pushed to Datadog)

# Run unit tests
pytest devops/datadog/tests/collectors/test_{collector_name}.py -v
```

### Step 8: Document

Create `README.md` in your collector directory:

```markdown
# {Service Name} Collector

Collects metrics from {Service Name} API for monitoring in Datadog.

## Metrics Collected

### {Category 1}

| Metric | Description | Unit | Type |
|--------|-------------|------|------|
| `{service}.{category}.{metric1}` | Description of metric | count | GAUGE |
| `{service}.{category}.{metric2}` | Description of metric | seconds | GAUGE |

### {Category 2}

| Metric | Description | Unit | Type |
|--------|-------------|------|------|
| `{service}.{category}.{metric3}` | Description of metric | count | GAUGE |

## Configuration

### Required Secrets

Store in AWS Secrets Manager (us-east-1):

- `{service}/api-key` - API authentication key
- `{service}/api-url` - Base URL for API

### Environment Variables

- `SERVICE_REGION` - Service region (default: us-east-1)
- `DD_ENV` - Datadog environment tag (production/staging)

## Deployment

Deployed as Kubernetes CronJob via Helm:

```bash
cd devops/charts
helm upgrade --install datadog-collectors \
  ./datadog-collectors \
  --set collectors.{collector_name}.enabled=true
```

Default schedule: Every 15 minutes

## Local Testing

```bash
# Dry run (no Datadog push)
python -m devops.datadog.scripts.run_collector {collector_name}

# Push to Datadog
python -m devops.datadog.scripts.run_collector {collector_name} --push
```

## API Documentation

- [{Service Name} API Docs](https://api.service.com/docs)
- [Authentication Guide](https://docs.service.com/auth)

## Troubleshooting

### Metric returns None

Check that:
1. API credentials are valid
2. API endpoint is accessible
3. Response format matches expected schema

### API rate limiting

If hitting rate limits:
1. Reduce collection frequency in `values.yaml`
2. Implement caching for expensive calls
3. Request higher rate limits from service

## Maintenance

- **Owner**: Team/Person responsible
- **API Version**: v1.2.3
- **Dependencies**: httpx, boto3
- **Last Updated**: 2025-01-15
```

### Step 9: Deploy

Build and deploy your collector:

```bash
# 1. Build Docker image (includes all collectors)
cd devops/datadog
docker build -t 751442549699.dkr.ecr.us-east-1.amazonaws.com/datadog-collectors:latest \
  -f docker/Dockerfile ../..

# 2. Push to ECR
docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/datadog-collectors:latest

# 3. Deploy with Helm
cd ../charts
helm upgrade --install datadog-collectors \
  ./datadog-collectors \
  --namespace monitoring \
  --set collectors.{collector_name}.enabled=true

# 4. Verify deployment
kubectl get cronjobs -n monitoring
kubectl describe cronjob {collector_name}-collector -n monitoring

# 5. Check first run
kubectl get jobs -n monitoring | grep {collector_name}
kubectl logs -n monitoring -l collector={collector_name} --tail=100
```

### Step 10: Monitor

Create Datadog monitors for your collector:

```yaml
# Monitor: Collector not running
Query: avg(last_30m):sum:{service}.{any_metric}{env:production} < 1
Alert: {Service} collector hasn't reported metrics in 30 minutes

# Monitor: High error rate
Query: avg(last_1h):sum:collector.{collector_name}.error_count{env:production} > 3
Alert: {Service} collector experiencing errors

# Monitor: Slow collection
Query: avg(last_30m):avg:collector.{collector_name}.duration_seconds{env:production} > 30
Alert: {Service} collector taking >30s to run
```

## Checklist

Before considering your collector complete:

- [ ] Collector class created (`collector.py`)
- [ ] Metrics defined (`metrics.py`)
- [ ] Secrets added to AWS Secrets Manager
- [ ] Helm values updated (`values.yaml`)
- [ ] Unit tests written and passing
- [ ] Tested locally (dry run)
- [ ] README.md created
- [ ] Deployed to staging (if available)
- [ ] Verified metrics in Datadog
- [ ] Datadog monitors created
- [ ] Documentation updated

## Common Patterns

### Pattern 1: Paginated API

```python
@metric("{service}.items.total", unit="count")
def get_total_items() -> int:
    """Get all items across multiple pages."""
    total = 0
    page = 1

    with _get_api_client() as client:
        while True:
            response = client.get("/items", params={"page": page, "per_page": 100})
            response.raise_for_status()
            data = response.json()

            items = data["items"]
            total += len(items)

            if len(items) < 100:  # Last page
                break

            page += 1

    return total
```

### Pattern 2: Time-Based Filtering

```python
@metric("{service}.events.last_7d", unit="count")
def get_recent_events() -> int:
    """Get events from last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with _get_api_client() as client:
        response = client.get("/events", params={"since": since})
        response.raise_for_status()
        return len(response.json()["events"])
```

### Pattern 3: Aggregation/Average

```python
@metric("{service}.metric.average", unit="{unit}")
def get_average_metric() -> float | None:
    """Calculate average, return None if no data."""
    with _get_api_client() as client:
        response = client.get("/stats")
        response.raise_for_status()
        values = response.json()["values"]

        if not values:
            return None

        return sum(values) / len(values)
```

### Pattern 4: Multiple Endpoints

```python
@metric("{service}.combined.metric", unit="count")
def get_combined_metric() -> int:
    """Combine data from multiple endpoints."""
    with _get_api_client() as client:
        # Endpoint 1
        response1 = client.get("/endpoint1")
        response1.raise_for_status()
        count1 = len(response1.json()["items"])

        # Endpoint 2
        response2 = client.get("/endpoint2")
        response2.raise_for_status()
        count2 = len(response2.json()["items"])

        return count1 + count2
```

## Tips

1. **Start Small**: Begin with 3-5 core metrics, expand later
2. **Mock Everything**: Use mocks in tests to avoid hitting real APIs
3. **Handle None**: Return `None` for metrics that can't be calculated
4. **Log Errors**: Use `logger.error()` for debugging
5. **Test Failures**: Write tests for API error cases
6. **Cache Wisely**: Cache API clients, not data (data should be fresh)
7. **Document Units**: Always specify units in `@metric` decorator

## Need Help?

- Architecture questions: See `COLLECTORS_ARCHITECTURE.md`
- Deployment issues: See `DEPLOYMENT.md`
- Secrets problems: See `SECRETS_MANAGEMENT.md`
- Slack channel: #datadog-collectors
