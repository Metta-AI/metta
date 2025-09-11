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
    ./metta/util/live_run_monitor.py --group my_group_name
    ./metta/util/live_run_monitor.py --name-filter "axel.*"
    ./metta/util/live_run_monitor.py --group my_group --name-filter "experiment.*"
    ./metta/util/live_run_monitor.py --refresh 15 --entity myteam --project myproject
    ./metta/util/live_run_monitor.py --fetch-limit 100 --display-limit 20
    ./metta/util/live_run_monitor.py  # Monitor last 10 runs (fetch 50, display 10)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from metta.adaptive.models import RunInfo

logger = logging.getLogger(__name__)

# Display limit for runs
DISPLAY_LIMIT = 10


def _get_status_color(status: str) -> str:
    """Get color for run status."""
    if status == "COMPLETED":
        return "bright_blue"
    elif status == "IN_TRAINING":
        return "bright_green"
    elif status == "PENDING":
        return "bright_black"
    elif status == "TRAINING_DONE_NO_EVAL":
        return "bright_yellow"
    elif status == "IN_EVAL":
        return "bright_cyan"
    elif status == "FAILED":
        return "bright_red"
    else:
        return "white"


def make_rich_monitor_table(runs: List[RunInfo]) -> Table:
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
            progress_str = f"{current_gsteps:.2f}/{total_gsteps:.0f}Gsteps ({percentage:.0f}%)"
        else:
            progress_str = "N/A"

        # Format score
        if run.observation and run.observation.score is not None:
            score_str = f"{run.observation.score:.3f}"
        else:
            score_str = "N/A"

        # Format cost
        cost_str = f"${run.cost:.2f}" if run.cost else "$0.00"

        # Status with color
        status_name = run.status.name if hasattr(run.status, "name") else str(run.status)
        status_color = _get_status_color(status_name)
        status_text = Text(status_name, style=status_color)

        table.add_row(
            run_id_text,
            status_text,
            progress_str,
            score_str,
            cost_str,
        )

    return table


def create_run_banner(group: Optional[str], name_filter: Optional[str], runs: List[RunInfo], display_limit: int = 10):
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
        f"🔄 LIVE RUN MONITOR: {filter_desc} | Fetched: {total_runs} runs, displaying at most {display_limit} runs. "
    )
    line1.append("Use --help to change limits.", style="dim")

    # Cost line with warning
    cost_line = RichText(f"💰 Total Cost: ${total_cost:.2f} ")
    cost_line.append("(Warning: cost is shown for a L4:4 instance until cost monitoring is fixed)", style="orange3")

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
            banner = create_run_banner(group, name_filter, all_runs, display_limit)

            # Create table
            table = make_rich_monitor_table(runs)

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

    from metta.adaptive.models import JobStatus, Observation, RunInfo

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
                observation = Observation(score=0.85 + i * 0.05, cost=4.50 + i, suggestion={})
                current_steps = 1000000000
                total_timesteps = 1000000000
            elif i < 4:
                status = JobStatus.IN_TRAINING
                observation = None
                current_steps = 500000000 + i * 100000000
                total_timesteps = 1000000000
            elif i < 6:
                status = JobStatus.PENDING
                observation = None
                current_steps = None
                total_timesteps = None
            else:
                status = JobStatus.FAILED
                observation = None
                current_steps = None
                total_timesteps = None

            run = RunInfo(
                run_id=run_id,
                has_started_training=status != JobStatus.PENDING,
                has_completed_training=status in [JobStatus.COMPLETED, JobStatus.FAILED],
                has_started_eval=status == JobStatus.COMPLETED,
                has_been_evaluated=status == JobStatus.COMPLETED,
                has_failed=status == JobStatus.FAILED,
                observation=observation,
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


def main():
    """CLI entry point for live run monitoring."""
    parser = argparse.ArgumentParser(
        description="Live monitor runs with rich terminal display",
        epilog="""
Examples:
  %(prog)s --group my_group_name
  %(prog)s --name-filter "axel.*"
  %(prog)s --group my_group --name-filter "experiment.*"
  %(prog)s --refresh 15 --entity myteam --project myproject
  %(prog)s --fetch-limit 100 --display-limit 20  # Fetch more, display more
  %(prog)s  # Monitor last 10 runs (fetch 50, display 10)

The monitor will display a live table with color-coded statuses:
  - Completed runs (blue)
  - In training runs (green)
  - Pending runs (gray)
  - Training done, no eval (orange)
  - Failed runs (red)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", "-g", help="WandB group to monitor (optional)")
    parser.add_argument("--name-filter", help="Regex filter for run names (e.g., 'axel.*')")
    parser.add_argument("--refresh", "-r", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument(
        "--entity", "-e", type=str, default="metta-research", help="WandB entity (default: metta-research)"
    )
    parser.add_argument("--project", "-p", type=str, default="metta", help="WandB project (default: metta)")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with mock data (no WandB connection required)"
    )
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen, append output instead")
    parser.add_argument(
        "--fetch-limit", type=int, default=50, help="Maximum number of runs to fetch from WandB (default: 50)"
    )
    parser.add_argument(
        "--display-limit", type=int, default=10, help="Maximum number of runs to display in table (default: 10)"
    )

    args = parser.parse_args()

    # Validate refresh interval
    if args.refresh < 1:
        print("Error: Refresh interval must be at least 1 second")
        sys.exit(1)

    # Validate limits
    if args.fetch_limit < 1:
        print("Error: Fetch limit must be at least 1")
        sys.exit(1)
    if args.display_limit < 1:
        print("Error: Display limit must be at least 1")
        sys.exit(1)
    if args.display_limit > args.fetch_limit:
        print("Warning: Display limit is greater than fetch limit, some runs may not be shown")
        print(f"  Fetch limit: {args.fetch_limit}, Display limit: {args.display_limit}")

    # Test mode with mock data
    if args.test:
        try:
            live_monitor_runs_test(
                group=args.group or "test_group",
                refresh_interval=args.refresh,
                clear_screen=not args.no_clear,
                display_limit=args.display_limit,
            )
        except KeyboardInterrupt:
            print("\nTest monitoring stopped by user.")
            sys.exit(0)
        return

    # Validate WandB access
    try:
        import wandb

        if not wandb.api.api_key:
            print("Error: WandB API key not found. Please run 'wandb login' first.")
            sys.exit(1)
    except ImportError:
        print("Error: WandB not installed. Please install with 'pip install wandb'.")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not validate WandB credentials: {e}")

    try:
        live_monitor_runs(
            group=args.group,
            name_filter=args.name_filter,
            refresh_interval=args.refresh,
            entity=args.entity,
            project=args.project,
            clear_screen=not args.no_clear,
            display_limit=args.display_limit,
            fetch_limit=args.fetch_limit,
        )
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
