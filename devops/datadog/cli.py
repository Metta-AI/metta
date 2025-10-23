#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.19.2",
#     "requests",
#     "rich",
#     "datadog-api-client>=2.43.0",
#     "httpx",
#     "boto3",
# ]
# ///
"""Datadog Management CLI.

Unified command-line interface for Datadog dashboard and collector management.

Usage:
    # Dashboard commands
    uv run ./devops/datadog/cli.py dashboard build
    uv run ./devops/datadog/cli.py dashboard push
    uv run ./devops/datadog/cli.py dashboard list

    # Collector commands
    uv run ./devops/datadog/cli.py collect github
    uv run ./devops/datadog/cli.py collect github --push

    # Or via metta (once integrated):
    metta datadog dashboard build
    metta datadog collect github --push
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

# ============================================================================
# Constants
# ============================================================================

DATADOG_DIR = Path(__file__).parent
SCRIPTS_DIR = DATADOG_DIR / "scripts"
DASHBOARDS_DIR = DATADOG_DIR / "dashboards"
TEMPLATES_DIR = DATADOG_DIR / "templates"
COLLECTORS_DIR = DATADOG_DIR / "collectors"

console = Console()

# ============================================================================
# Main App
# ============================================================================

app = typer.Typer(
    name="datadog",
    help="Datadog dashboard and collector management",
    no_args_is_help=True,
)

# Sub-apps for command groups
dashboard_app = typer.Typer(
    name="dashboard",
    help="Dashboard management commands",
    no_args_is_help=True,
)

app.add_typer(dashboard_app, name="dashboard")

# ============================================================================
# Environment Validation
# ============================================================================


def check_datadog_env() -> None:
    """Check that required Datadog environment variables are set."""
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")

    if not api_key or api_key == "your_api_key_here":
        console.print("[red]Error:[/red] DD_API_KEY not set or using placeholder value")
        console.print("Run: [cyan]source ./devops/datadog/load_env.sh[/cyan]")
        raise typer.Exit(1)

    if not app_key or app_key == "your_app_key_here":
        console.print("[red]Error:[/red] DD_APP_KEY not set or using placeholder value")
        console.print("Run: [cyan]source ./devops/datadog/load_env.sh[/cyan]")
        raise typer.Exit(1)


def run_script(script_name: str, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run a script from the scripts directory."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        console.print(f"[red]Error:[/red] Script not found: {script_path}")
        raise typer.Exit(1)

    cmd = [str(script_path)]
    if args:
        cmd.extend(args)

    return subprocess.run(cmd, check=True)


# ============================================================================
# Dashboard Commands
# ============================================================================


@dashboard_app.command("build")
def dashboard_build(
    file: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Build specific dashboard file"),
    ] = None,
) -> None:
    """Build dashboard JSON from Jsonnet sources.

    Compiles .jsonnet files from dashboards/ directory into .json files
    in templates/ directory.
    """
    if file:
        # Build single dashboard
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)

        if not file.suffix == ".jsonnet":
            console.print("[red]Error:[/red] File must have .jsonnet extension")
            raise typer.Exit(1)

        base_name = file.stem
        output_file = TEMPLATES_DIR / f"{base_name}.json"

        console.print(f"Building {base_name}...")

        try:
            result = subprocess.run(
                ["jsonnet", str(file)],
                capture_output=True,
                text=True,
                check=True,
            )

            TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
            output_file.write_text(result.stdout)

            console.print(f"[green]✓[/green] Built {output_file}")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error:[/red] Failed to build {file}")
            console.print(e.stderr)
            raise typer.Exit(1) from None
        except FileNotFoundError:
            console.print("[red]Error:[/red] jsonnet not found. Install with: brew install jsonnet")
            raise typer.Exit(1) from None

    else:
        # Build all dashboards
        if not DASHBOARDS_DIR.exists():
            console.print(f"[red]Error:[/red] Dashboards directory not found: {DASHBOARDS_DIR}")
            console.print("See docs/JSONNET_PROTOTYPE.md to get started")
            raise typer.Exit(1)

        jsonnet_files = list(DASHBOARDS_DIR.glob("*.jsonnet"))

        if not jsonnet_files:
            console.print(f"[yellow]Warning:[/yellow] No .jsonnet files found in {DASHBOARDS_DIR}")
            console.print("See docs/JSONNET_PROTOTYPE.md to get started")
            raise typer.Exit(1)

        console.print("Building dashboards from Jsonnet...")
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for jsonnet_file in jsonnet_files:
            base_name = jsonnet_file.stem
            output_file = TEMPLATES_DIR / f"{base_name}.json"

            try:
                console.print(f"  Building {base_name}...")

                result = subprocess.run(
                    ["jsonnet", str(jsonnet_file)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                output_file.write_text(result.stdout)
                success_count += 1

            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error:[/red] Failed to build {jsonnet_file}")
                console.print(e.stderr)
                continue
            except FileNotFoundError:
                console.print("[red]Error:[/red] jsonnet not found. Install with: brew install jsonnet")
                raise typer.Exit(1) from None

        console.print()
        console.print(f"[green]✓[/green] Built {success_count} dashboard(s)")
        console.print()
        console.print("Next steps:")
        console.print("  - Review: ls -l templates/")
        console.print("  - Diff: git diff templates/")
        console.print("  - Push: uv run ./devops/datadog/cli.py dashboard push")


@dashboard_app.command("push")
def dashboard_push(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be pushed without actually pushing"),
    ] = False,
) -> None:
    """Push dashboards to Datadog.

    Uploads all dashboard JSON files from templates/ directory to Datadog.
    """
    check_datadog_env()

    if not TEMPLATES_DIR.exists() or not list(TEMPLATES_DIR.glob("*.json")):
        console.print(f"[red]Error:[/red] No JSON files found in {TEMPLATES_DIR}")
        console.print("Run: [cyan]uv run ./devops/datadog/cli.py dashboard build[/cyan]")
        raise typer.Exit(1)

    json_files = list(TEMPLATES_DIR.glob("*.json"))

    console.print(f"Pushing {len(json_files)} dashboard(s) to Datadog...")

    args = [str(f) for f in json_files]
    if dry_run:
        args.append("--dry-run")

    try:
        run_script("push_dashboard.py", args)

        if not dry_run:
            console.print()
            console.print("[green]✓[/green] Push complete!")
            console.print()
            console.print("Next steps:")
            console.print("  - Verify in Datadog UI")
            console.print("  - Commit: git add templates/ && git commit")

    except subprocess.CalledProcessError:
        console.print("[red]Error:[/red] Failed to push dashboards")
        raise typer.Exit(1) from None


@dashboard_app.command("pull")
def dashboard_pull() -> None:
    """Export all dashboards from Datadog.

    Downloads all dashboards from Datadog and saves them as JSON files
    in templates/ directory.
    """
    check_datadog_env()

    console.print("Exporting all dashboards from Datadog...")

    try:
        run_script("batch_export.py")

        console.print()
        console.print(f"[green]✓[/green] Dashboards exported to {TEMPLATES_DIR}")
        console.print()
        console.print("Next steps:")
        console.print("  - Review: ls -l templates/")
        console.print("  - Edit: vim templates/my_dashboard.json")
        console.print("  - Push: uv run ./devops/datadog/cli.py dashboard push")

    except subprocess.CalledProcessError:
        console.print("[red]Error:[/red] Failed to export dashboards")
        raise typer.Exit(1) from None


@dashboard_app.command("list")
def dashboard_list() -> None:
    """List all dashboards in Datadog account."""
    check_datadog_env()

    console.print("Fetching dashboard list from Datadog...")

    try:
        run_script("fetch_dashboards.py", ["--format=summary"])

    except subprocess.CalledProcessError:
        console.print("[red]Error:[/red] Failed to fetch dashboards")
        raise typer.Exit(1) from None


@dashboard_app.command("export")
def dashboard_export(
    dashboard_id: Annotated[str, typer.Argument(help="Dashboard ID to export")],
) -> None:
    """Export a specific dashboard by ID."""
    check_datadog_env()

    console.print(f"Exporting dashboard {dashboard_id}...")

    output_file = TEMPLATES_DIR / f"dashboard_{dashboard_id}.json"
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [str(SCRIPTS_DIR / "export_dashboard.py"), dashboard_id],
            capture_output=True,
            text=True,
            check=True,
        )

        output_file.write_text(result.stdout)
        console.print(f"[green]✓[/green] Exported to {output_file}")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Failed to export dashboard {dashboard_id}")
        console.print(e.stderr)
        raise typer.Exit(1) from None


@dashboard_app.command("delete")
def dashboard_delete(
    dashboard_ids: Annotated[list[str], typer.Argument(help="Dashboard ID(s) to delete")],
) -> None:
    """Delete dashboard(s) by ID."""
    check_datadog_env()

    console.print(f"Deleting {len(dashboard_ids)} dashboard(s)...")

    try:
        run_script("delete_dashboard.py", dashboard_ids)
        console.print(f"[green]✓[/green] Deleted {len(dashboard_ids)} dashboard(s)")

    except subprocess.CalledProcessError:
        console.print("[red]Error:[/red] Failed to delete dashboards")
        raise typer.Exit(1) from None


@dashboard_app.command("diff")
def dashboard_diff() -> None:
    """Show git diff of dashboard changes."""
    if not TEMPLATES_DIR.exists() or not list(TEMPLATES_DIR.glob("*.json")):
        console.print("No dashboard files to diff")
        return

    console.print("Dashboard changes:")
    console.print("=" * 60)

    try:
        subprocess.run(
            ["git", "diff", str(TEMPLATES_DIR)],
            check=False,  # git diff returns 1 if there are changes
        )
    except subprocess.CalledProcessError:
        pass


@dashboard_app.command("metrics")
def dashboard_list_metrics() -> None:
    """List available metrics from Datadog (for building widgets)."""
    check_datadog_env()

    console.print("Fetching available metrics from Datadog...")

    try:
        run_script("list_metrics.py")

    except subprocess.CalledProcessError:
        console.print("[red]Error:[/red] Failed to fetch metrics")
        raise typer.Exit(1) from None


@dashboard_app.command("clean")
def dashboard_clean() -> None:
    """Remove generated JSON files."""
    if not TEMPLATES_DIR.exists():
        console.print("Nothing to clean")
        return

    json_files = list(TEMPLATES_DIR.glob("*.json"))
    other_files = list(DATADOG_DIR.glob("dashboards*.json"))

    if not json_files and not other_files:
        console.print("Nothing to clean")
        return

    console.print("Cleaning up...")

    for file in json_files:
        file.unlink()

    for file in other_files:
        file.unlink()

    console.print(f"[green]✓[/green] Cleaned up {len(json_files) + len(other_files)} file(s)")


# ============================================================================
# Collector Commands
# ============================================================================


@app.command("collect")
def collect(
    collector: Annotated[
        str,
        typer.Argument(help="Collector name (e.g., 'github', 'skypilot')"),
    ],
    push: Annotated[
        bool,
        typer.Option("--push", help="Push metrics to Datadog (default is dry-run)"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Run a collector to gather metrics.

    By default, collectors run in dry-run mode (no metrics pushed).
    Use --push to actually submit metrics to Datadog.

    Examples:
        # Dry run (show what would be collected)
        uv run ./devops/datadog/cli.py collect github

        # Actually push metrics to Datadog
        uv run ./devops/datadog/cli.py collect github --push
    """
    import json
    import logging

    # Setup logging level
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if collector == "github":
        # Run the collector using the standalone runner script
        # This ensures we have access to all project dependencies
        runner_script = DATADOG_DIR / "run_collector.py"

        cmd = ["uv", "run", "python", str(runner_script), collector]

        if push:
            # Check Datadog env vars before running
            check_datadog_env()
            cmd.append("--push")

        if verbose:
            cmd.append("--verbose")

        # Add --json flag to get structured output
        cmd.append("--json")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse JSON output
            metrics = json.loads(result.stdout)

            # Display metrics
            console.print(f"\n[green]✓[/green] Collected {len(metrics)} metrics:\n")

            if verbose:
                # Detailed JSON output
                console.print(json.dumps(metrics, indent=2, sort_keys=True))
            else:
                # Summary table
                table = Table(title="GitHub Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green", justify="right")

                for key, value in sorted(metrics.items()):
                    if value is not None:
                        # Format numbers nicely
                        if isinstance(value, float):
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)
                        table.add_row(key, formatted_value)

                console.print(table)

            if push:
                console.print("\n[green]✓[/green] Successfully pushed metrics to Datadog")
            else:
                console.print("\n[dim]Dry-run mode. Use --push to submit metrics to Datadog.[/dim]")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error:[/red] Failed to run {collector} collector")
            if verbose and e.stderr:
                console.print(f"\n{e.stderr}")
            raise typer.Exit(1) from None

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            raise typer.Exit(130) from None

        except json.JSONDecodeError as e:
            console.print("[red]Error:[/red] Failed to parse collector output")
            if verbose:
                console.print(f"Details: {e}")
                console.print(f"Output: {result.stdout}")
            raise typer.Exit(1) from None

        except Exception as e:
            console.print("[red]Error:[/red] Unexpected error")
            if verbose:
                console.print(f"Details: {e}")
                import traceback

                traceback.print_exc()
            raise typer.Exit(1) from None

    else:
        console.print(f"[yellow]Warning:[/yellow] Collector '{collector}' not yet implemented")
        console.print("Available collectors: github")
        console.print("Planned collectors: skypilot, wandb, ec2, asana")
        raise typer.Exit(1)


@app.command("list-collectors")
def list_collectors() -> None:
    """List available collectors."""
    table = Table(title="Available Collectors")

    table.add_column("Collector", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    # Check which collectors exist
    collectors = [
        ("github", "✅ Implemented", "PRs, commits, CI/CD, branches, developers"),
        ("skypilot", "📋 Planned", "Jobs, clusters, compute costs"),
        ("wandb", "📋 Planned", "Training runs, experiments, GPU hours"),
        ("ec2", "📋 Planned", "Instances, costs, utilization"),
        ("asana", "📋 Planned", "Tasks, projects, velocity"),
    ]

    for name, status, description in collectors:
        table.add_row(name, status, description)

    console.print(table)
    console.print()
    console.print("Usage: [cyan]uv run ./devops/datadog/cli.py collect <collector>[/cyan]")
    console.print("Example: [cyan]uv run ./devops/datadog/cli.py collect github --push[/cyan]")


# ============================================================================
# Utility Commands
# ============================================================================


@app.command("env")
def check_env() -> None:
    """Check environment variables."""
    console.print("Checking Datadog environment variables...")

    api_key = os.getenv("DD_API_KEY", "")
    app_key = os.getenv("DD_APP_KEY", "")
    site = os.getenv("DD_SITE", "datadoghq.com")

    table = Table()
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Value")

    def check_var(name: str, value: str, placeholder: str = "") -> tuple[str, str]:
        if not value:
            return "❌ Not set", ""
        if value == placeholder:
            return "⚠️ Placeholder", value[:10] + "..."
        return "✅ Set", value[:10] + "..."

    api_status, api_preview = check_var("DD_API_KEY", api_key, "your_api_key_here")
    app_status, app_preview = check_var("DD_APP_KEY", app_key, "your_app_key_here")
    site_status, site_preview = check_var("DD_SITE", site)

    table.add_row("DD_API_KEY", api_status, api_preview)
    table.add_row("DD_APP_KEY", app_status, app_preview)
    table.add_row("DD_SITE", site_status or "✅ Set", site)

    console.print(table)

    if not api_key or not app_key:
        console.print()
        console.print("To set environment variables, run:")
        console.print("  [cyan]source ./devops/datadog/load_env.sh[/cyan]")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app()
