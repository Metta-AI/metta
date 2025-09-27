#!/usr/bin/env uv run
"""Live run monitor with rich terminal display.

This utility provides a real-time, color-coded view of runs with automatic refresh.

Features:
- Color-coded status: Completed (blue), In Training (green), Pending (gray),
  Training Done No Eval (orange), Failed (red)
- Live progress updates in Gsteps (billions of steps)
- Cost tracking in USD format
- Auto-refresh every 30 seconds (configurable)
- In-place updates (no scrolling output)
- Runs sorted by most recent first
- Configurable fetch and display limits to prevent excessive WandB queries
- Default: fetch 50 runs, display 10 most recent

Usage:
    ./metta/adaptive/live_run_monitor.py --group my_group_name
    ./metta/adaptive/live_run_monitor.py --name-filter "axel.*"
    ./metta/adaptive/live_run_monitor.py --group my_group --name-filter "experiment.*"
    ./metta/adaptive/live_run_monitor.py --refresh 15 --entity myteam --project myproject
    ./metta/adaptive/live_run_monitor.py --fetch-limit 100 --display-limit 20
    ./metta/adaptive/live_run_monitor.py  # Monitor last 10 runs (fetch 50, display 10)
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT

if TYPE_CHECKING:
    from metta.adaptive.models import JobStatus, RunInfo

logger = logging.getLogger(__name__)

# Display limit for runs
DISPLAY_LIMIT = 10

# Typer application for CLI usage
app = typer.Typer(
    help="Live run monitor with rich terminal display.",
    no_args_is_help=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _get_status_color(status: "JobStatus") -> str:
    """Get color for run status."""
    from metta.adaptive.models import JobStatus

    if status == JobStatus.COMPLETED:
        return "bright_blue"
    elif status == JobStatus.IN_TRAINING:
        return "bright_green"
    elif status == JobStatus.PENDING:
        return "bright_black"
    elif status == JobStatus.TRAINING_DONE_NO_EVAL:
        return "green"
    elif status == JobStatus.IN_EVAL:
        return "bright_cyan"
    elif status == JobStatus.FAILED:
        return "bright_red"
    elif status == JobStatus.STALE:
        return "bright_black"
    else:
        return "white"


def make_rich_monitor_table(runs: list["RunInfo"], score_metric: str = "env_agent/heart.get") -> Table:
    """Create rich table for run monitoring."""

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Progress", style="yellow", width=25)
    table.add_column("Score", style="blue", width=15)
    table.add_column("Cost", style="green")

    for run in runs:
        # Format run ID with clickable link to WandB
        wandb_url = f"https://wandb.ai/metta-research/metta/runs/{run.run_id}"
        run_id_text = Text(run.run_id, style="link " + wandb_url)

        # Format progress in Gsteps with percentage
        if run.total_timesteps and run.current_steps is not None:
            total_gsteps = run.total_timesteps / 1_000_000_000
            current_gsteps = run.current_steps / 1_000_000_000
            percentage = (run.current_steps / run.total_timesteps) * 100
            # Show a space before the unit and format total to 2 decimals
            progress_str = f"{current_gsteps:.2f}/{total_gsteps:.2f} Gsteps ({percentage:.0f}%)"
        else:
            progress_str = "N/A"

        # Format score from run.summary
        if run.summary and score_metric in run.summary:
            score_value = run.summary[score_metric]
            if score_value is not None:
                if isinstance(score_value, (int, float)):
                    score_str = f"{score_value:.3f}"
                else:
                    score_str = str(score_value)
            else:
                score_str = "N/A"
        else:
            score_str = "N/A"

        # Format cost
        cost_str = f"${run.cost:.2f}" if run.cost else "$0.00"

        # Status with color
        status = run.status
        status_color = _get_status_color(status)
        status_text = Text(status.value, style=status_color)

        table.add_row(
            run_id_text,
            status_text,
            progress_str,
            score_str,
            cost_str,
        )

    return table


def create_run_banner(
    group: Optional[str],
    name_filter: Optional[str],
    runs: list["RunInfo"],
    display_limit: int = 10,
    score_metric: str = "env_agent/heart.get",
):
    """Create a banner with run information."""

    # Calculate runtime from earliest run created_at
    earliest_created = None
    for run in runs:
        if run.created_at:
            # Parse created_at if it's a string from WandB
            created_at = run.created_at
            if isinstance(created_at, str):
                from dateutil import parser

                created_at = parser.parse(created_at)

            if earliest_created is None or created_at < earliest_created:
                earliest_created = created_at

    if earliest_created:
        # Use timezone-aware current time to match WandB timestamps
        from datetime import timezone

        current_time = datetime.now(timezone.utc) if earliest_created.tzinfo else datetime.now()
        runtime = current_time - earliest_created
        runtime_hours = runtime.total_seconds() / 3600.0
        runtime_str = f"{runtime_hours:.1f} hours"
    else:
        runtime_str = "Unknown"

    # Count runs by status - handle JobStatus enum

    status_counts = {}
    for run in runs:
        # Get the enum value name without the prefix
        status = run.status.name if hasattr(run.status, "name") else str(run.status)
        status_counts[status] = status_counts.get(status, 0) + 1

    total_runs = len(runs)
    completed_runs = status_counts.get("COMPLETED", 0)
    in_training = status_counts.get("IN_TRAINING", 0)
    pending = status_counts.get("PENDING", 0)
    failed = status_counts.get("FAILED", 0)

    # Calculate total cost
    total_cost = sum(run.cost for run in runs)

    # Create filter description
    filter_parts = []
    if group:
        filter_parts.append(f"group={group}")
    if name_filter:
        filter_parts.append(f"name~{name_filter}")

    if filter_parts:
        filter_desc = " | ".join(filter_parts)
    else:
        filter_desc = "all runs"

    # Create inline banner with styled parts
    from rich.text import Text as RichText

    # First line with fetch/display info
    line1 = RichText(
        f"🔄 LIVE RUN MONITOR: {filter_desc} | Fetched: {total_runs} runs, "
        f"displaying at most {display_limit} runs | Score: {score_metric}. "
    )
    line1.append("Use --help to change limits.", style="dim")

    # Cost line with warning
    cost_line = RichText(f"💰 Total Cost: ${total_cost:.2f} ")

    banner_lines = [
        line1,
        f"⏱️  Runtime: {runtime_str}",
        f"📊 Runs: {total_runs} total | ✅ {completed_runs} completed | 🔄 {in_training} training"
        f" | ⏳ {pending} pending | ❌ {failed} failed",
        cost_line,
        f"🔄 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "─" * 100,
    ]

    from rich.console import Group

    return Group(*[Text(line) if isinstance(line, str) else line for line in banner_lines])


def live_monitor_runs(
    group: Optional[str] = None,
    name_filter: Optional[str] = None,
    refresh_interval: int = 30,
    entity: str = "metta-research",
    project: str = "metta",
    clear_screen: bool = True,
    display_limit: int = 10,
    fetch_limit: int = 50,
    score_metric: str = "env_agent/heart.get",
) -> None:
    """Live monitor runs with rich terminal display.

    Args:
        fetch_limit: Maximum number of runs to fetch from WandB (default: 50)
        display_limit: Maximum number of runs to display in table (default: 10)
    """

    console = Console()

    # Always use adaptive store
    try:
        from metta.adaptive.stores.wandb import WandbStore

        store = WandbStore(entity=entity, project=project)
    except ImportError:
        print("Error: Cannot import adaptive WandbStore. Make sure dependencies are installed.")
        sys.exit(1)

    def generate_display():
        try:
            # Build filters
            filters = {}
            if group:
                filters["group"] = group
            if name_filter:
                filters["name"] = {"regex": name_filter}

            # Fetch runs (already sorted by created_at newest first from WandB)
            all_runs = store.fetch_runs(filters, limit=fetch_limit)
            # Take only the display_limit for the table
            runs = all_runs[:display_limit]

            if not runs:
                # Show warning but keep monitoring
                warning_msg = "⚠️  No runs found matching filters"
                if group or name_filter:
                    filter_parts = []
                    if group:
                        filter_parts.append(f"group={group}")
                    if name_filter:
                        filter_parts.append(f"name~{name_filter}")
                    filter_desc = " | ".join(filter_parts)
                    warning_msg += f": {filter_desc}"

                warning_msg += "\n   Waiting for runs to appear..."

                return Text(warning_msg, style="bright_yellow")

            # Create banner using all fetched runs for accurate statistics
            banner = create_run_banner(group, name_filter, all_runs, display_limit, score_metric)

            # Create table
            table = make_rich_monitor_table(runs, score_metric)

            # Create a renderable group
            display = Group(
                banner,
                Text(""),  # Empty line
                table,
                Text(""),  # Empty line
                Text("(CMD + Click to see run in WandB)", style="dim"),
            )
            return display

        except Exception as e:
            error_msg = f"Error fetching run data: {str(e)}"
            logger.error(error_msg)
            return Text(error_msg, style="bright_red")

    # Start monitoring
    filter_desc_parts = []
    if group:
        filter_desc_parts.append(f"group={group}")
    if name_filter:
        filter_desc_parts.append(f"name~{name_filter}")

    if filter_desc_parts:
        filter_desc = " | ".join(filter_desc_parts)
        console.print(f"Starting live monitor for runs matching: {filter_desc}")
    else:
        console.print("Starting live monitor for most recent runs")

    console.print(f"Refresh interval: {refresh_interval} seconds")
    console.print(f"Fetch limit: {fetch_limit} runs, Display limit: {display_limit} runs")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(generate_display(), console=console, refresh_per_second=1, screen=clear_screen) as live:
            while True:
                time.sleep(refresh_interval)
                live.update(generate_display())

    except KeyboardInterrupt:
        console.print("\n\nMonitoring stopped by user.")
    except Exception as e:
        console.print(f"\n\nError during monitoring: {e}")
        raise


def live_monitor_runs_test(
    group: Optional[str] = None, refresh_interval: int = 30, clear_screen: bool = True, display_limit: int = 10
) -> None:
    """Test mode for live run monitoring with mock data."""
    from datetime import datetime, timedelta

    from metta.adaptive.models import JobStatus, RunInfo

    console = Console()

    def generate_test_runs():
        """Generate mock runs for testing."""
        runs = []
        base_time = datetime.now() - timedelta(hours=2)

        for i in range(min(8, display_limit + 2)):  # Generate a few more than display limit
            run_id = f"test_run_{i:03d}"

            # Vary the status
            if i < 2:
                status = JobStatus.COMPLETED
                summary = {"env_agent/heart.get": 0.85 + i * 0.05}
                current_steps = 1000000000
                total_timesteps = 1000000000
            elif i < 4:
                status = JobStatus.IN_TRAINING
                summary = {"env_agent/heart.get": 0.75 + i * 0.05}
                current_steps = 500000000 + i * 100000000
                total_timesteps = 1000000000
            elif i < 6:
                status = JobStatus.PENDING
                summary = None
                current_steps = None
                total_timesteps = None
            else:
                status = JobStatus.FAILED
                summary = None
                current_steps = None
                total_timesteps = None

            run = RunInfo(
                run_id=run_id,
                has_started_training=status != JobStatus.PENDING,
                has_completed_training=status in [JobStatus.COMPLETED, JobStatus.FAILED],
                has_started_eval=status == JobStatus.COMPLETED,
                has_been_evaluated=status == JobStatus.COMPLETED,
                has_failed=status == JobStatus.FAILED,
                summary=summary,
                cost=4.50 + i if status != JobStatus.PENDING else 0.0,
                created_at=base_time + timedelta(minutes=i * 15),
                current_steps=current_steps,
                total_timesteps=total_timesteps,
            )
            runs.append(run)

        return runs

    def generate_display():
        runs = generate_test_runs()

        # Create banner
        banner = create_run_banner(group, None, runs, display_limit)

        # Create table
        table = make_rich_monitor_table(runs)

        # Create a renderable group
        display = Group(
            Text(banner),
            Text(""),  # Empty line
            table,
            Text(""),  # Empty line
            Text("🧪 TEST MODE - Mock data displayed above", style="bright_magenta"),
        )
        return display

    console.print(f"Starting TEST live monitor for group: {group or 'test_group'}")
    console.print(f"Refresh interval: {refresh_interval} seconds")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(generate_display(), console=console, refresh_per_second=1, screen=clear_screen) as live:
            while True:
                time.sleep(refresh_interval)
                live.update(generate_display())

    except KeyboardInterrupt:
        console.print("\n\nTest monitoring stopped by user.")
    except Exception as e:
        console.print(f"\n\nError during test monitoring: {e}")
        raise


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    group: Annotated[str | None, typer.Option("--group", "-g", help="WandB group to monitor")] = None,
    name_filter: Annotated[
        str | None, typer.Option("--name-filter", help=f"Regex filter for run names (e.g., '{os.getenv('USER')}.*')")
    ] = None,
    refresh: Annotated[int, typer.Option("--refresh", "-r", help="Refresh interval in seconds")] = 30,
    entity: Annotated[str, typer.Option("--entity", "-e", help="WandB entity")] = METTA_WANDB_ENTITY,
    project: Annotated[str, typer.Option("--project", "-p", help="WandB project")] = METTA_WANDB_PROJECT,
    test: Annotated[
        bool, typer.Option("--test", help="Run in test mode with mock data (no WandB connection required)")
    ] = False,
    no_clear: Annotated[bool, typer.Option("--no-clear", help="Don't clear screen, append output instead")] = False,
    fetch_limit: Annotated[int, typer.Option("--fetch-limit", help="Maximum number of runs to fetch from WandB")] = 50,
    display_limit: Annotated[
        int, typer.Option("--display-limit", help="Maximum number of runs to display in table")
    ] = 10,
    score_metric: Annotated[
        str,
        typer.Option(
            "--score-metric",
            help="Metric key in run.summary to use for score",
        ),
    ] = "env_agent/heart.get",
) -> None:
    """Default command for the live run monitor app."""
    # If a subcommand is provided, do nothing here
    if ctx.invoked_subcommand:
        return

    # Validate refresh interval
    if refresh < 1:
        typer.echo("Error: Refresh interval must be at least 1 second")
        raise typer.Exit(1)

    # Validate limits
    if fetch_limit < 1:
        typer.echo("Error: Fetch limit must be at least 1")
        raise typer.Exit(1)
    if display_limit < 1:
        typer.echo("Error: Display limit must be at least 1")
        raise typer.Exit(1)
    if display_limit > fetch_limit:
        typer.echo("Warning: Display limit is greater than fetch limit, some runs may not be shown")
        typer.echo(f"  Fetch limit: {fetch_limit}, Display limit: {display_limit}")

    # Test mode with mock data
    if test:
        try:
            live_monitor_runs_test(
                group=group or "test_group",
                refresh_interval=refresh,
                clear_screen=not no_clear,
                display_limit=display_limit,
            )
        except KeyboardInterrupt:
            typer.echo("\nTest monitoring stopped by user.")
            raise typer.Exit(0) from None
        return

    import wandb

    if not wandb.api.api_key:
        typer.echo("Error: WandB API key not found. Please run 'metta install wandb' first.")
        raise typer.Exit(1)

    try:
        live_monitor_runs(
            group=group,
            name_filter=name_filter,
            refresh_interval=refresh,
            entity=entity,
            project=project,
            clear_screen=not no_clear,
            display_limit=display_limit,
            fetch_limit=fetch_limit,
            score_metric=score_metric,
        )
    except KeyboardInterrupt:
        typer.echo("\nMonitoring stopped by user.")
        raise typer.Exit(0) from None
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
