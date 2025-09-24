from __future__ import annotations

import json
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
) -> None:
    """Collect registered metrics and optionally send them to Datadog."""
    metrics = collect_metrics()

    typer.echo("Metrics:")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
    if not push:
        typer.echo("Skipping Datadog push")
        return

    typer.echo("Pushing metrics to Datadog...")

    configuration = _datadog_configuration()
    payload = _build_series_payload(metrics)
    with ApiClient(configuration) as api_client:
        MetricsApi(api_client).submit_metrics(body=payload)
    typer.echo("Metrics pushed to Datadog")


def main() -> None:
    init_logging()
    app()


if __name__ == "__main__":
    main()
