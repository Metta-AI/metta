
import datetime
import json
import os

import datadog_api_client
import datadog_api_client.v2.api.metrics_api
import datadog_api_client.v2.model.metric_intake_type
import datadog_api_client.v2.model.metric_payload
import datadog_api_client.v2.model.metric_point
import datadog_api_client.v2.model.metric_series
import typer

import metta.common.datadog.config
import metta.common.util.log_config
import softmax.aws.secrets_manager
import softmax.dashboard.registry

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
    if metta.common.datadog.config.datadog_config.DD_SERVICE:
        tags.append(f"service:{metta.common.datadog.config.datadog_config.DD_SERVICE}")
    if metta.common.datadog.config.datadog_config.DD_ENV:
        tags.append(f"env:{metta.common.datadog.config.datadog_config.DD_ENV}")
    if metta.common.datadog.config.datadog_config.DD_VERSION:
        tags.append(f"version:{metta.common.datadog.config.datadog_config.DD_VERSION}")
    return tags


def _build_series_payload(
    metrics: dict[str, float],
) -> datadog_api_client.v2.model.metric_payload.MetricPayload:
    point_time = int((datetime.datetime.now(datetime.timezone.utc)).timestamp())

    series = [
        datadog_api_client.v2.model.metric_series.MetricSeries(
            metric=metric_name,
            type=datadog_api_client.v2.model.metric_intake_type.MetricIntakeType.GAUGE,
            points=[datadog_api_client.v2.model.metric_point.MetricPoint(timestamp=point_time, value=value)],
            tags=_base_tags(),
        )
        for metric_name, value in metrics.items()
    ]

    return datadog_api_client.v2.model.metric_payload.MetricPayload(series=series)


def _datadog_configuration() -> datadog_api_client.Configuration:
    configuration = datadog_api_client.Configuration()
    configuration.server_variables["site"] = os.environ.get(
        "DD_SITE", metta.common.datadog.config.datadog_config.DD_SITE
    )
    configuration.api_key["apiKeyAuth"] = softmax.aws.secrets_manager.get_secretsmanager_secret("datadog/api-key")
    configuration.api_key["appKeyAuth"] = softmax.aws.secrets_manager.get_secretsmanager_secret("datadog/app-key")
    return configuration


@app.command()
def report(
    push: bool = typer.Option(False, "--push", "-p", help="Push metrics to Datadog."),
) -> None:
    """Collect registered metrics and optionally send them to Datadog."""
    metrics = softmax.dashboard.registry.collect_metrics()

    typer.echo("Metrics:")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
    if not push:
        typer.echo("Skipping Datadog push")
        return

    typer.echo("Pushing metrics to Datadog...")

    configuration = _datadog_configuration()
    payload = _build_series_payload(metrics)
    with datadog_api_client.ApiClient(configuration) as api_client:
        datadog_api_client.v2.api.metrics_api.MetricsApi(api_client).submit_metrics(body=payload)
    typer.echo("Metrics pushed to Datadog")


def main() -> None:
    metta.common.util.log_config.init_logging()
    app()


if __name__ == "__main__":
    main()
