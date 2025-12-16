"""CLI for Datadog CI metrics collector."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from devops.datadog.collectors.ci_collector import CICollector
from devops.datadog.datadog_client import DatadogMetricsClient
from devops.datadog.models import MetricSample

app = typer.Typer(
    help="Collect CI metrics and push to Datadog",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def collect(
    push: bool = typer.Option(
        False,
        "--push",
        help="Submit collected metrics to Datadog.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print metrics instead of pushing to Datadog.",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Optional path to write JSON payload (useful for debugging).",
    ),
) -> None:
    """Collect CI metrics and optionally push to Datadog."""
    if push and dry_run:
        typer.echo("Error: --push and --dry-run are mutually exclusive")
        raise typer.Exit(1)

    collector = CICollector()
    samples = collector.collect()
    _handle_samples(samples, dry_run=dry_run, output=output)
    if push:
        DatadogMetricsClient().submit(samples)
        typer.echo("Metrics submitted to Datadog")


def _handle_samples(samples: list[MetricSample], *, dry_run: bool, output: Path | None) -> None:
    payload = [sample.to_dict() for sample in samples]
    if dry_run or not samples:
        typer.echo(json.dumps(payload, indent=2))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        typer.echo(f"Wrote payload to {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
