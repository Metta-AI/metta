"""Datadog API client wrapper for metric submission."""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries

logger = logging.getLogger(__name__)


class DatadogClient:
    """Wrapper for Datadog API metric submission with batching and retry logic."""

    def __init__(
        self,
        api_key: str,
        app_key: str,
        site: str = "datadoghq.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Datadog client.

        Args:
            api_key: Datadog API key
            app_key: Datadog application key
            site: Datadog site (e.g., "datadoghq.com", "datadoghq.eu")
            max_retries: Maximum number of retry attempts for failed submissions
            retry_delay: Initial delay between retries in seconds (exponential backoff)
        """
        self.configuration = Configuration()
        self.configuration.api_key["apiKeyAuth"] = api_key
        self.configuration.api_key["appKeyAuth"] = app_key
        self.configuration.server_variables["site"] = site
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def submit_metric(
        self,
        metric_name: str,
        value: float | int,
        metric_type: str = "gauge",
        tags: list[str] | None = None,
        timestamp: int | None = None,
    ) -> bool:
        """Submit a single metric to Datadog.

        Args:
            metric_name: Metric name (e.g., "github.prs.open")
            value: Metric value
            metric_type: Metric type ("gauge", "count", "rate")
            tags: List of tags to attach
            timestamp: Unix timestamp (defaults to now)

        Returns:
            True if submission succeeded, False otherwise
        """
        return self.submit_metrics_batch(
            [
                {
                    "metric": metric_name,
                    "value": value,
                    "type": metric_type,
                    "tags": tags,
                    "timestamp": timestamp,
                }
            ]
        )

    def submit_metrics_batch(
        self,
        metrics: list[dict[str, Any]],
    ) -> bool:
        """Submit multiple metrics to Datadog in a single API call.

        Args:
            metrics: List of metric dictionaries, each containing:
                - metric: Metric name
                - value: Metric value
                - type: Metric type (default: "gauge")
                - tags: List of tags (optional)
                - timestamp: Unix timestamp (optional, defaults to now)

        Returns:
            True if submission succeeded, False otherwise
        """
        if not metrics:
            logger.warning("No metrics to submit")
            return True

        current_time = int(datetime.now(timezone.utc).timestamp())

        # Convert metric type string to MetricIntakeType enum
        type_map = {
            "gauge": MetricIntakeType.GAUGE,
            "count": MetricIntakeType.COUNT,
            "rate": MetricIntakeType.RATE,
        }

        # Build metric series
        series = []
        for metric in metrics:
            timestamp = metric.get("timestamp", current_time)
            metric_type_str = metric.get("type", "gauge")
            metric_type = type_map.get(metric_type_str, MetricIntakeType.GAUGE)

            series.append(
                MetricSeries(
                    metric=metric["metric"],
                    type=metric_type,
                    points=[MetricPoint(timestamp=timestamp, value=metric["value"])],
                    tags=metric.get("tags", []),
                )
            )

        payload = MetricPayload(series=series)
        return self._submit_with_retry(payload)

    def _submit_with_retry(self, payload: MetricPayload) -> bool:
        """Submit metrics with exponential backoff retry logic.

        Args:
            payload: Metric payload for Datadog API

        Returns:
            True if submission succeeded, False otherwise
        """
        last_error = None
        metric_count = len(payload.series)

        for attempt in range(self.max_retries):
            try:
                with ApiClient(self.configuration) as api_client:
                    api_instance = MetricsApi(api_client)
                    api_instance.submit_metrics(body=payload)

                logger.info(f"Successfully submitted {metric_count} metrics to Datadog")
                return True

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed to submit metrics: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

        # All retries exhausted
        logger.error(f"Failed to submit {metric_count} metrics after {self.max_retries} attempts: {last_error}")
        return False

    def submit_from_registry(
        self,
        registry: dict[str, dict[str, Any]],
        global_tags: list[str] | None = None,
    ) -> tuple[int, int]:
        """Collect and submit all metrics from a registry.

        Args:
            registry: Metric registry from decorators.get_registered_metrics()
            global_tags: Tags to attach to all metrics

        Returns:
            Tuple of (successful_count, failed_count)
        """
        metrics_to_submit = []
        failed_count = 0

        for name, metadata in registry.items():
            func = metadata["function"]
            metric_type = metadata["type"]
            metric_tags = metadata.get("tags", [])

            # Combine global and metric-specific tags
            all_tags = (global_tags or []) + metric_tags

            try:
                value = func()

                # Skip None values
                if value is None:
                    logger.debug(f"Skipping metric {name}: value is None")
                    continue

                metrics_to_submit.append(
                    {
                        "metric": name,
                        "value": value,
                        "type": metric_type,
                        "tags": all_tags,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to collect metric {name}: {e}")
                failed_count += 1

        # Submit all metrics in a single batch
        if metrics_to_submit:
            success = self.submit_metrics_batch(metrics_to_submit)
            if success:
                return len(metrics_to_submit), failed_count
            else:
                return 0, failed_count + len(metrics_to_submit)

        return 0, failed_count
