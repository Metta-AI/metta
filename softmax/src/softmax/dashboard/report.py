from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import typer
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries

from devops.datadog.collectors import available_collectors, get_collector
from devops.datadog.datadog_client import DatadogMetricsClient
from metta.common.datadog.config import datadog_config
from metta.common.util.log_config import init_logging
from softmax.aws.secrets_manager import get_secretsmanager_secret
from softmax.dashboard.registry import collect_metrics

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Collect Softmax dashboard metrics and interact with Datadog.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _base_tags() -> list[str]:
    tags: list[str] = [
        "source:softmax-system-health",
    ]
    if datadog_config.DD_SERVICE:
        tags.append(f"service:{datadog_config.DD_SERVICE}")
    if datadog_config.DD_ENV:
        tags.append(f"env:{datadog_config.DD_ENV}")
    if datadog_config.DD_VERSION:
        tags.append(f"version:{datadog_config.DD_VERSION}")
    return tags


def _build_series_payload(
    metrics: dict[str, float],
) -> MetricPayload:
    point_time = int((datetime.now(timezone.utc)).timestamp())

    series = [
        MetricSeries(
            metric=metric_name,
            type=MetricIntakeType.GAUGE,
            points=[MetricPoint(timestamp=point_time, value=value)],
            tags=_base_tags(),
        )
        for metric_name, value in metrics.items()
    ]

    return MetricPayload(series=series)


def _datadog_configuration() -> Configuration:
    configuration = Configuration()
    configuration.server_variables["site"] = os.environ.get("DD_SITE", datadog_config.DD_SITE)
    configuration.api_key["apiKeyAuth"] = get_secretsmanager_secret("datadog/api-key")
    configuration.api_key["appKeyAuth"] = get_secretsmanager_secret("datadog/app-key")
    return configuration


@app.command()
def report(
    push: bool = typer.Option(False, "--push", "-p", help="Push metrics to Datadog."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print metrics instead of pushing to Datadog."),
) -> None:
    """Collect registered metrics and optionally send them to Datadog.

    This command collects metrics from:
    1. Softmax dashboard registry (existing metrics)
    2. Datadog collectors (ci, training, eval) for infra health metrics
    """
    all_samples = []

    # Collect from existing softmax dashboard registry (backward compatibility)
    try:
        metrics = collect_metrics()
        typer.echo("Softmax dashboard metrics:")
        typer.echo(json.dumps(metrics, indent=2, sort_keys=True))

        # Convert to MetricSample format for consistency (if needed)
        # For now, we'll keep the old metrics separate and only send collector metrics
        # to avoid breaking existing dashboards
    except Exception as e:
        logger.warning("Failed to collect softmax dashboard metrics: %s", e, exc_info=True)

    # Collect from Datadog collectors (ci, training, eval)
    collector_slugs = available_collectors()

    for slug in collector_slugs:
        try:
            typer.echo(f"▶ running collector: {slug}")
            collector = get_collector(slug)
            samples = collector.collect()

            if not isinstance(samples, list):
                logger.error("Collector %s returned non-list: %s", slug, type(samples))
                continue

            all_samples.extend(samples)
            typer.echo(f"… emitted {len(samples)} metrics")

        except Exception as e:
            logger.error("Collector %s failed: %s", slug, e, exc_info=True)
            typer.echo(f"✗ collector {slug} failed: {e}")
            # Continue with other collectors

    if not all_samples:
        typer.echo("⚠ No metrics collected from collectors")
        return

    # Print collected metrics in dry-run mode
    if dry_run:
        typer.echo(f"\nCollected {len(all_samples)} total metrics (dry-run mode):")
        for sample in all_samples:
            typer.echo(f"  {sample.name} = {sample.value} (tags: {len(sample.tags)} tags)")
        return

    # Send to Datadog if push is enabled
    if push:
        try:
            client = DatadogMetricsClient()
            client.submit(all_samples)
            typer.echo("\n✓ all metrics sent to datadog")
        except Exception as e:
            logger.error("Failed to push metrics to Datadog: %s", e, exc_info=True)
            typer.echo(f"\n✗ Failed to push metrics: {e}")
            raise
    else:
        typer.echo(f"\nCollected {len(all_samples)} metrics (use --push to send to Datadog)")

    # Also send existing softmax dashboard metrics if push is enabled
    if push and metrics:
        try:
            typer.echo("Pushing softmax dashboard metrics to Datadog...")
            configuration = _datadog_configuration()
            payload = _build_series_payload(metrics)
            with ApiClient(configuration) as api_client:
                MetricsApi(api_client).submit_metrics(body=payload)
            typer.echo("Softmax dashboard metrics pushed to Datadog")
        except Exception as e:
            logger.error("Failed to push softmax dashboard metrics: %s", e, exc_info=True)
            typer.echo(f"Warning: Failed to push softmax dashboard metrics: {e}")


def main() -> None:
    init_logging()
    app()


if __name__ == "__main__":
    main()
