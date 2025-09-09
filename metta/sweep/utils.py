"""Utility functions for sweep orchestration."""

import argparse
import hashlib
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from metta.sweep.models import JobDefinition, JobTypes, RunInfo, SweepMetadata
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text


logger = logging.getLogger(__name__)


def make_monitor_table(
    runs: list["RunInfo"],
    title: str = "Run Status Table",
    logger_prefix: str = "",
    include_score: bool = True,
    truncate_run_id: bool = True,
) -> list[str]:
    """Create a formatted table showing run status.

    Args:
        runs: List of RunInfo objects to display
        title: Title for the table
        logger_prefix: Prefix to add to each log line (e.g., "[OptimizingScheduler]")
        include_score: Whether to include the score column
        truncate_run_id: Whether to truncate run IDs to just show trial numbers

    Returns:
        List of formatted lines that can be logged
    """
    lines = []
    prefix = f"{logger_prefix} " if logger_prefix else ""

    # Title
    lines.append(f"{prefix}{title}:")
    lines.append(f"{prefix}{'=' * 110}")

    # Header
    if include_score:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30} {'Score':<15} {'Cost':<10}")
    else:
        lines.append(f"{prefix}{'Run ID':<25} {'Status':<25} {'Progress':<30}")
    lines.append(f"{prefix}{'-' * 110}")

    # Rows
    for run in runs:
        # Format run ID
        display_id = get_display_id(run.run_id) if truncate_run_id else run.run_id

        # Format progress in Gsteps
        if run.total_timesteps and run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            total_gsteps = run.total_timesteps / 1_000_000_000
            progress_pct = (run.current_steps / run.total_timesteps) * 100
            progress_str = f"{current_gsteps:.3g}/{total_gsteps:.3g} Gsteps ({progress_pct:.1f}%)"
        elif run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            progress_str = f"{current_gsteps:.3g}/? Gsteps"
        else:
            progress_str = "-"

        # Format score and cost
        if include_score:
            score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
            cost_str = f"${run.observation.cost:.2f}" if run.observation else "N/A"
            lines.append(
                f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30} {score_str:<15} {cost_str:<10}"
            )
        else:
            lines.append(f"{prefix}{display_id:<25} {str(run.status):<25} {progress_str:<30}")

    lines.append(f"{prefix}{'=' * 110}")

    return lines


def get_display_id(run_id: str) -> str:
    """Extract clean display ID from run ID.

    Args:
        run_id: Full run ID (e.g., "sweep_name_trial_0001_a1b2c3")

    Returns:
        Cleaned display ID (e.g., "trial_0001")
    """
    if "_trial_" in run_id:
        # Extract everything after "_trial_"
        trial_part = run_id.split("_trial_")[-1]
        run_id = trial_part
    return run_id


def build_eval_overrides(
    run_id: str,
    sweep_id: str,
    stats_server_uri: Optional[str] = None,
    additional_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build evaluation override parameters.

    Args:
        run_id: The run ID for WandB tracking
        sweep_id: The sweep ID for grouping
        stats_server_uri: Optional stats server URI
        additional_overrides: Optional additional overrides to merge

    Returns:
        Dictionary of evaluation overrides
    """
    eval_overrides = additional_overrides.copy() if additional_overrides else {}

    # WandB configuration
    eval_overrides["push_metrics_to_wandb"] = "True"
    eval_overrides["wandb.name"] = run_id
    eval_overrides["wandb.run_id"] = run_id
    eval_overrides["wandb.group"] = sweep_id

    # Stats server configuration
    if stats_server_uri:
        eval_overrides["stats_server_uri"] = stats_server_uri

    return eval_overrides


def build_train_overrides(
    stats_server_uri: Optional[str] = None,
    additional_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build training override parameters.

    Args:
        stats_server_uri: Optional stats server URI
        additional_overrides: Optional additional overrides to merge

    Returns:
        Dictionary of training overrides
    """
    overrides = additional_overrides.copy() if additional_overrides else {}

    if stats_server_uri:
        overrides["stats_server_uri"] = stats_server_uri
        overrides["trainer.evaluation.evaluate_remote"] = "True"
        overrides["trainer.evaluation.evaluate_local"] = "False"
        overrides["trainer.evaluation.skip_git_check"] = "True"

    return overrides


def create_eval_job(
    run_id: str,
    sweep_id: str,
    recipe_module: str,
    eval_entrypoint: str,
    stats_server_uri: Optional[str] = None,
    eval_args: Optional[List[str]] = None,
    eval_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create an evaluation job definition.

    Args:
        run_id: The run ID to evaluate
        sweep_id: The sweep ID for grouping
        recipe_module: Module containing the evaluation function
        eval_entrypoint: Name of the evaluation function
        stats_server_uri: Optional stats server URI
        eval_args: Optional positional arguments for evaluation
        eval_overrides: Optional additional overrides

    Returns:
        JobDefinition for evaluation
    """
    from metta.sweep.models import JobDefinition, JobTypes

    overrides = build_eval_overrides(
        run_id=run_id,
        sweep_id=sweep_id,
        stats_server_uri=stats_server_uri,
        additional_overrides=eval_overrides,
    )

    return JobDefinition(
        run_id=run_id,
        cmd=f"{recipe_module}.{eval_entrypoint}",
        type=JobTypes.LAUNCH_EVAL,
        args=eval_args or [],
        overrides=overrides,
        metadata={"policy_uri": f"wandb://metta/{run_id}"},
    )


def create_training_job(
    run_id: str,
    sweep_id: str,
    recipe_module: str,
    train_entrypoint: str,
    config: Dict[str, Any],
    gpus: int = 1,
    nodes: int = 1,
    stats_server_uri: Optional[str] = None,
    train_overrides: Optional[Dict[str, Any]] = None,
) -> "JobDefinition":
    """Create a training job definition.

    Args:
        run_id: The unique run ID
        sweep_id: The sweep ID for grouping
        recipe_module: Module containing the training function
        train_entrypoint: Name of the training function
        config: Hyperparameter configuration from optimizer
        gpus_per_job: Number of GPUs per job
        stats_server_uri: Optional stats server URI
        train_overrides: Optional additional overrides

    Returns:
        JobDefinition for training
    """

    overrides = build_train_overrides(
        stats_server_uri=stats_server_uri,
        additional_overrides=train_overrides,
    )

    return JobDefinition(
        run_id=run_id,
        cmd=f"{recipe_module}.{train_entrypoint}",
        type=JobTypes.LAUNCH_TRAINING,
        gpus=gpus,
        nodes=nodes,
        config=config,
        overrides=overrides,
        metadata={"group": sweep_id},
    )


def generate_run_id(sweep_id: str, trial_num: int) -> str:
    """Generate a standardized run ID with hash to avoid collisions.

    Args:
        sweep_id: The sweep identifier
        trial_num: The trial number (1-based)

    Returns:
        Formatted run ID like "sweep_id_trial_0001_a1b2c3"
    """
    # Generate a short hash to avoid name collisions
    # Use sweep_id, trial_num, and current time to ensure uniqueness
    hash_input = f"{sweep_id}_{trial_num}_{time.time()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{sweep_id}_trial_{trial_num:04d}_{short_hash}"


def _get_status_color(status: str) -> str:
    """Get color for different run statuses."""
    status_colors = {
        "COMPLETED": "bright_blue",
        "IN_TRAINING": "bright_green",
        "PENDING": "bright_black",
        "TRAINING_DONE_NO_EVAL": "bright_yellow",
        "IN_EVAL": "bright_cyan",
        "EVAL_DONE_NOT_COMPLETED": "bright_magenta",
        "FAILED": "bright_red",
    }
    return status_colors.get(status, "white")


def _sort_runs_for_display(runs: List["RunInfo"]) -> List["RunInfo"]:
    """Sort runs for display by created_at time, newest first."""
    # Sort by created_at time (descending), with None values at the end
    return sorted(runs, key=lambda r: r.created_at if r.created_at else datetime.max, reverse=True)


def make_rich_monitor_table(runs: List["RunInfo"], sweep_metadata: Optional["SweepMetadata"] = None) -> Table:
    """Create a rich table with color-coded run status."""

    # Sort runs with completed at bottom
    sorted_runs = _sort_runs_for_display(runs)

    # Create table
    table = Table(show_header=True, header_style="bold magenta", border_style="bright_blue")
    table.add_column("Run ID", style="cyan", width=25)
    table.add_column("Status", width=25)
    table.add_column("Progress", style="yellow", width=30)
    table.add_column("Score", style="magenta", width=15)
    table.add_column("Cost", style="green", width=10)

    for run in sorted_runs:
        # Format run ID with clickable link to WandB
        display_id = get_display_id(run.run_id)
        wandb_url = f"https://wandb.ai/metta-research/metta/runs/{run.run_id}"
        run_id_text = Text(display_id, style="link " + wandb_url)

        # Format progress in Gsteps
        if run.total_timesteps and run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            total_gsteps = run.total_timesteps / 1_000_000_000
            progress_pct = (run.current_steps / run.total_timesteps) * 100
            progress_str = f"{current_gsteps:.3g}/{total_gsteps:.3g} Gsteps ({progress_pct:.1f}%)"
        elif run.current_steps is not None:
            current_gsteps = run.current_steps / 1_000_000_000
            progress_str = f"{current_gsteps:.3g}/? Gsteps"
        else:
            progress_str = "-"

        # Format score and cost
        score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
        cost_str = f"${run.cost:.2f}" if run.cost > 0 else "$0.00"

        # Get status color and create colored status text
        status_color = _get_status_color(str(run.status))
        status_text = Text(str(run.status), style=status_color)

        table.add_row(run_id_text, status_text, progress_str, score_str, cost_str)

    return table


def create_sweep_banner(sweep_id: str, runs: List["RunInfo"], start_time: Optional[datetime] = None) -> str:
    """Create a banner with sweep information."""

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

    banner_lines = [
        f"🔄 LIVE SWEEP MONITOR: {sweep_id}",
        f"⏱️  Runtime: {runtime_str}",
        f"📊 Runs: {total_runs} total | ✅ {completed_runs} completed | 🔄 {in_training} training | ⏳ {pending} pending | ❌ {failed} failed",
        f"💰 Total Cost: ${total_cost:.2f}",
        f"🔄 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "─" * 100,
    ]

    return "\n".join(banner_lines)


def live_monitor_sweep(
    sweep_id: str,
    refresh_interval: int = 30,
    entity: str = "metta-research",
    project: str = "metta",
    clear_screen: bool = True,
) -> None:
    """Live monitor a sweep with rich terminal display."""

    try:
        from metta.sweep.stores.wandb import WandbStore
    except ImportError:
        print("Error: Cannot import WandbStore. Make sure wandb is installed and configured.")
        sys.exit(1)

    console = Console()
    store = WandbStore(entity=entity, project=project)
    start_time = datetime.now()

    def generate_display():
        try:
            # Fetch runs
            runs = store.fetch_runs({"sweep_id": sweep_id})

            if not runs:
                return Text(f"No runs found for sweep: {sweep_id}", style="bright_red")

            # Create banner
            banner = create_sweep_banner(sweep_id, runs)

            # Create table
            table = make_rich_monitor_table(runs)

            # Create a renderable group
            display = Group(
                Text(banner),
                Text(""),  # Empty line
                table,
            )
            return display

        except Exception as e:
            error_msg = f"Error fetching sweep data: {str(e)}"
            logger.error(error_msg)
            return Text(error_msg, style="bright_red")

    console.print(f"Starting live monitor for sweep: {sweep_id}")
    console.print(f"Refresh interval: {refresh_interval} seconds")
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


def live_monitor_sweep_test(sweep_id: str, refresh_interval: int = 30, clear_screen: bool = True) -> None:
    """Test mode for live sweep monitoring with mock data."""
    from datetime import datetime

    from metta.sweep.models import Observation, RunInfo

    console = Console()
    start_time = datetime.now()

    def create_mock_runs():
        """Create mock run data for testing."""
        mock_runs = []

        # Create various types of mock runs
        mock_data = [
            ("trial_0001", "COMPLETED", 2500000000, 2500000000, 0.9234, 12.45),
            ("trial_0002", "IN_TRAINING", 1890000000, 2500000000, None, 8.76),
            ("trial_0003", "PENDING", None, 2500000000, None, 0.0),
            ("trial_0004", "TRAINING_DONE_NO_EVAL", 2500000000, 2500000000, None, 11.20),
            ("trial_0005", "FAILED", 450000000, 2500000000, None, 3.45),
            ("trial_0006", "IN_EVAL", 2500000000, 2500000000, None, 9.88),
        ]

        for i, (run_id, status, current_steps, total_steps, score, cost) in enumerate(mock_data):
            observation = None
            if score is not None:
                observation = Observation(score=score, cost=cost, suggestion={})

            run = RunInfo(
                run_id=f"{sweep_id}_{run_id}_mock{i:03d}",
                current_steps=current_steps,
                total_timesteps=total_steps,
                cost=cost,
                observation=observation,
            )
            # Manually set status since it's a property
            run.has_failed = status == "FAILED"
            run.has_started_training = status != "PENDING"
            run.has_completed_training = status in ["COMPLETED", "TRAINING_DONE_NO_EVAL", "IN_EVAL"]
            run.has_started_eval = status in ["IN_EVAL", "COMPLETED"]
            run.has_been_evaluated = status == "COMPLETED"

            mock_runs.append(run)

        return mock_runs

    def generate_display():
        try:
            # Get mock runs
            runs = create_mock_runs()

            # Simulate some progress over time
            elapsed = (datetime.now() - start_time).total_seconds()
            training_run = next((r for r in runs if "trial_0002" in r.run_id), None)
            if training_run and training_run.current_steps:
                # Simulate training progress
                progress_boost = int(elapsed * 1000000)  # 1M steps per second
                training_run.current_steps = min(
                    training_run.current_steps + progress_boost,
                    training_run.total_timesteps or training_run.current_steps + progress_boost,
                )

            # Create banner
            banner = create_sweep_banner(f"{sweep_id} (TEST MODE)", runs)

            # Create table
            table = make_rich_monitor_table(runs)

            # Create display
            display = Group(
                Text(banner),
                Text(""),
                table,
                Text("\n[TEST MODE] This is mock data for testing purposes", style="bright_yellow"),
            )
            return display

        except Exception as e:
            error_msg = f"Error generating test display: {str(e)}"
            logger.error(error_msg)
            return Text(error_msg, style="bright_red")

    console.print(f"Starting TEST MODE live monitor for sweep: {sweep_id}")
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
    """CLI entry point for live sweep monitoring."""
    parser = argparse.ArgumentParser(description="Live monitor a sweep with rich terminal display")
    parser.add_argument("sweep_id", help="Sweep ID to monitor")
    parser.add_argument("--refresh", "-r", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument("--entity", "-e", type=str, default="metta", help="WandB entity (default: metta)")
    parser.add_argument("--project", "-p", type=str, default="metta", help="WandB project (default: metta)")

    args = parser.parse_args()

    live_monitor_sweep(sweep_id=args.sweep_id, refresh_interval=args.refresh, entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
