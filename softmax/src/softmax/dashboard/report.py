import json
from datetime import datetime, timezone
from typing import Iterable

import typer
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.exceptions import ApiException
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries

from metta.common.datadog.config import datadog_config
from softmax.aws.secrets_manager import get_secret
from softmax.dashboard.registry import collect_metrics


def _base_tags() -> list[str]:
    tags: list[str] = ["source:softmax-dashboard"]
    if datadog_config.DD_SERVICE:
        tags.append(f"service:{datadog_config.DD_SERVICE}")
    if datadog_config.DD_ENV:
        tags.append(f"env:{datadog_config.DD_ENV}")
    if datadog_config.DD_VERSION:
        tags.append(f"version:{datadog_config.DD_VERSION}")
    return tags


def _build_series_payload(
    metrics: dict[str, float], *, tags: Iterable[str], timestamp: datetime | None = None
) -> MetricPayload:
    point_time = int((timestamp or datetime.now(timezone.utc)).timestamp())
    tag_list = list(tags)

    series = [
        MetricSeries(
            metric=f"softmax.{metric_name}",
            type=MetricIntakeType.GAUGE,
            points=[MetricPoint(timestamp=point_time, value=value)],
            tags=tag_list + [f"metric:{metric_name}"],
        )
        for metric_name, value in metrics.items()
    ]

    return MetricPayload(series=series)


def _submit_to_datadog(payload: MetricPayload, *, api_key: str) -> None:
    configuration = Configuration()
    configuration.server_variables["site"] = datadog_config.DD_SITE
    configuration.api_key["apiKeyAuth"] = api_key

    try:
        with ApiClient(configuration) as api_client:
            MetricsApi(api_client).submit_metrics(body=payload)
    except ApiException as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"Failed to push metrics to Datadog: {exc}") from exc


app = typer.Typer(help="Collect dashboard metrics and optionally push them to Datadog.", add_completion=False)


@app.command()
def report(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Skip Datadog submission and print metrics."),
) -> None:
    metrics = collect_metrics()

    if dry_run or not metrics:
        typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
        return

    api_key = get_secret("datadog/api_key")
    if not api_key:
        raise RuntimeError("Datadog API key not found")
    payload = _build_series_payload(metrics, tags=_base_tags())
    _submit_to_datadog(payload, api_key=api_key)
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
