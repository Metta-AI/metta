"""CLI for Datadog utilities.

Note: Training and eval collectors were removed. The stable runner workflow
emits metrics directly. This CLI only supports CI collector for GitHub metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import typer

from devops.datadog.collectors import available_collectors, get_collector
from devops.datadog.datadog_client import DatadogMetricsClient
from devops.datadog.models import MetricSample

logging.basicConfig(level=logging.INFO)

app = typer.Typer(
    help="Metta Datadog utilities",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command("collect")
def collect_command(
    collector: str = typer.Argument(..., help="Collector slug (ci)"),
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
    """Collect metrics using the specified collector (CI only)."""
    if push and dry_run:
        typer.echo("Error: --push and --dry-run are mutually exclusive")
        raise typer.Exit(1)

    collector_instance = get_collector(collector)
    typer.echo(f"[collector] Running '{collector}' collector...")
    samples = collector_instance.collect()
    _handle_samples(samples, dry_run=dry_run, output=output)
    if push:
        typer.echo("[collector] Submitting metrics to Datadog...")
        DatadogMetricsClient().submit(samples)
        typer.echo("[collector] Submission complete.")
    elif not dry_run:
        typer.echo("[collector] Push flag not set; metrics were not sent.")


@app.command("list")
def list_collectors() -> None:
    """List available collectors."""
    typer.echo("Available collectors:")
    for slug in available_collectors():
        typer.echo(f" - {slug}")


def _handle_samples(samples: List[MetricSample], *, dry_run: bool, output: Path | None) -> None:
    payload = [sample.to_dict() for sample in samples]
    if dry_run or not samples:
        typer.echo(json.dumps(payload, indent=2))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        typer.echo(f"[collector] Wrote payload to {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
