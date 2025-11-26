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
    help="Metta Datadog metric collectors",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command("collect")
def collect_command(
    collector: str = typer.Argument(..., help="Collector slug (ci, training, eval)"),
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
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write JSON payload (useful for debugging).",
    ),
) -> None:
    """Collect metrics using the specified collector."""
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


@app.command("list-workflows")
def list_workflows_command(
    repo: str = typer.Option(
        None,
        "--repo",
        help="Repository (e.g., Metta-AI/metta). Defaults to METTA_GITHUB_REPO env var.",
    ),
) -> None:
    """List all workflows in the repository to help identify which ones to monitor."""
    import os
    from devops.datadog.github_client import GitHubClient

    repo = repo or os.environ.get("METTA_GITHUB_REPO", "Metta-AI/metta")
    token = os.environ.get("GITHUB_DASHBOARD_TOKEN") or os.environ.get("GITHUB_TOKEN")
    github = GitHubClient(token=token)

    typer.echo(f"Fetching workflows for {repo}...")
    workflows = github.list_workflows(repo)

    if not workflows:
        typer.echo("No workflows found.")
        return

    typer.echo(f"\nFound {len(workflows)} workflows:\n")
    typer.echo(f"{'ID':<10} {'Name':<50} {'Path':<60} {'State':<10}")
    typer.echo("-" * 130)

    for workflow in workflows:
        workflow_id = workflow.get("id", "")
        name = workflow.get("name", "N/A")
        path = workflow.get("path", "N/A")
        state = workflow.get("state", "N/A")
        typer.echo(f"{workflow_id:<10} {name:<50} {path:<60} {state:<10}")

    typer.echo("\n💡 To configure which workflows to monitor, set these env vars:")
    typer.echo("   CI_TESTS_BLOCKING_MERGE_WORKFLOWS=workflow1,workflow2")
    typer.echo("   CI_BENCHMARKS_WORKFLOWS=workflow1,workflow2")


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
