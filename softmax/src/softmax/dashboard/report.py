from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterable

import typer
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.dashboards_api import DashboardsApi
from datadog_api_client.v1.model.dashboard import Dashboard as DashboardModel
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries

from metta.common.datadog.config import datadog_config
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.log_config import init_logging
from softmax.aws.secrets_manager import get_secretsmanager_secret
from softmax.dashboard.layout import build_dashboard_definition
from softmax.dashboard.registry import collect_metrics

app = typer.Typer(
    help="Collect Softmax dashboard metrics and interact with Datadog.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

DEFAULT_BRANCH = "main"
DEFAULT_WORKFLOW = "checks.yml"
DEFAULT_LOOKBACK_DAYS = 7


def _datadog_repo_tag() -> str:
    return f"repo:{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"


def _base_tags() -> list[str]:
    tags: list[str] = [
        "source:softmax-dashboard",
        _datadog_repo_tag(),
        f"branch:{DEFAULT_BRANCH}",
        f"workflow:{DEFAULT_WORKFLOW}",
    ]
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
            metric=f"metta.{metric_name}",
            type=MetricIntakeType.GAUGE,
            points=[MetricPoint(timestamp=point_time, value=value)],
            tags=tag_list + [f"metric:{metric_name}"],
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
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Skip Datadog submission and print metrics."),
) -> None:
    """Collect registered metrics and optionally send them to Datadog."""

    init_logging()
    metrics = collect_metrics()

    if dry_run or not metrics:
        typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
        return

    configuration = _datadog_configuration()
    tags = _base_tags()
    payload = _build_series_payload(metrics, tags=tags)
    with ApiClient(configuration) as api_client:
        MetricsApi(api_client).submit_metrics(body=payload)
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command()
def sync_dashboard(
    title: str = typer.Option("Softmax System Health", "--title", help="Dashboard title."),
    description: str = typer.Option(
        "Auto-generated from softmax.dashboard.registry METRIC_GOALS.",
        "--description",
        help="Dashboard description.",
    ),
    dashboard_id: str = typer.Option("j79-2k2-5ym", "--dashboard-id", help="Existing dashboard ID to update."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print payload without calling Datadog."),
) -> None:
    """Create or update the Datadog dashboard with one widget per metric."""

    init_logging()
    payload = build_dashboard_definition(title=title, description=description)

    if dry_run:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    configuration = _datadog_configuration()
    print(configuration.api_key)
    dashboard_body = DashboardModel(**payload)

    with ApiClient(configuration) as api_client:
        api = DashboardsApi(api_client)
        response = api.update_dashboard(dashboard_id, body=dashboard_body)
        typer.echo(f"Updated dashboard {response.id}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
