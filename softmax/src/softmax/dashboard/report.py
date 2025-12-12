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

    This command collects metrics from the Softmax dashboard registry.
    """
    if push and dry_run:
        typer.echo("Error: --push and --dry-run are mutually exclusive")
        raise typer.Exit(1)

    metrics = None  # Initialize before try block

    # Collect from existing softmax dashboard registry
    try:
        metrics = collect_metrics()
        typer.echo("Softmax dashboard metrics:")
        typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
    except Exception as e:
        logger.warning("Failed to collect softmax dashboard metrics: %s", e, exc_info=True)

    # Send to Datadog if push is enabled
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
    elif dry_run:
        typer.echo("Dry-run mode: skipping Datadog push")


def main() -> None:
    init_logging()
    app()


if __name__ == "__main__":
    main()
