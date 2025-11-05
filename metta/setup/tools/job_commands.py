"""CLI commands for managing job groups."""

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root
from metta.jobs.job_display import JobDisplay
from metta.jobs.job_manager import JobManager
from metta.jobs.notebook_generation import generate_experiment_notebook
from metta.setup.utils import error, info, success, warning

app = typer.Typer(
    help="Manage job groups - monitor and control parallel jobs",
    no_args_is_help=True,
)

console = Console()


def _get_job_manager(state_dir: Path | None = None) -> JobManager:
    """Get or create a JobManager instance."""
    if state_dir is None:
        state_dir = get_repo_root() / "job_state"
    return JobManager(base_dir=state_dir)


@app.command(name="monitor", help="Monitor jobs in a group with live updates")
def cmd_monitor(
    group: Annotated[str, typer.Argument(help="Job group to monitor")],
    state_dir: Annotated[
        Optional[Path],
        typer.Option("--state-dir", help="Job state directory"),
    ] = None,
    refresh_interval: Annotated[
        float,
        typer.Option("--refresh", "-r", help="Refresh interval in seconds"),
    ] = 1.0,
    show_logs: Annotated[
        bool,
        typer.Option("--logs/--no-logs", help="Show running job logs"),
    ] = True,
):
    """Monitor all jobs in a group with live updates.

    Displays a live view similar to the stable release system, showing:
    - Failed jobs (at the top for visibility)
    - Active jobs with progress bars and live log tails
    - Completed jobs with artifacts (WandB URLs, checkpoints)
    - Pending jobs with dependency information

    The monitor will continue updating until all jobs in the group are complete.

    Example:
        metta job monitor experiment_abc123
        metta job monitor my-sweep --refresh 2.0 --no-logs
    """
    manager = _get_job_manager(state_dir)

    # Check if group exists
    group_jobs = manager.get_group_jobs(group)
    if not group_jobs:
        error(f"No jobs found in group '{group}'")
        raise typer.Exit(1)

    info(f"Monitoring {len(group_jobs)} jobs in group '{group}'")
    info("Press Ctrl+C to exit (jobs will continue running)\n")

    display = JobDisplay(manager, group=group)

    try:
        all_complete = False
        while not all_complete:
            # Poll for completed jobs
            manager.poll()

            # Display current status
            display.display_status(
                clear_screen=False,
                show_running_logs=show_logs,
            )

            # Check if all jobs are complete
            group_jobs = manager.get_group_jobs(group)
            all_complete = all(manager.get_job_state(job_name).is_terminal() for job_name in group_jobs)

            if not all_complete:
                time.sleep(refresh_interval)

        # Final status
        success("\nAll jobs complete!")

        # Show summary
        failed_jobs = []
        succeeded_jobs = []
        for job_name in manager.get_group_jobs(group):
            state = manager.get_job_state(job_name)
            if state.exit_code == 0:
                succeeded_jobs.append(job_name)
            else:
                failed_jobs.append(job_name)

        if succeeded_jobs:
            success(f"Succeeded: {len(succeeded_jobs)} jobs")
            for job_name in succeeded_jobs:
                info(f"  ✓ {job_name}")

        if failed_jobs:
            error(f"Failed: {len(failed_jobs)} jobs")
            for job_name in failed_jobs:
                warning(f"  ✗ {job_name}")

    except KeyboardInterrupt:
        info("\nMonitoring stopped (jobs are still running)")
        info(f"Run 'metta job monitor {group}' to resume monitoring")
        raise typer.Exit(0) from None


@app.command(name="kill", help="Cancel all jobs in a group")
def cmd_kill(
    group: Annotated[str, typer.Argument(help="Job group to cancel")],
    state_dir: Annotated[
        Optional[Path],
        typer.Option("--state-dir", help="Job state directory"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
):
    """Cancel all jobs in a group.

    This will terminate all running jobs and mark pending jobs as cancelled.
    Completed jobs are not affected.

    Example:
        metta job kill experiment_abc123
        metta job kill my-sweep --force
    """
    manager = _get_job_manager(state_dir)

    # Check if group exists
    group_jobs = manager.get_group_jobs(group)
    if not group_jobs:
        error(f"No jobs found in group '{group}'")
        raise typer.Exit(1)

    # Get job states
    states = [manager.get_job_state(job_name) for job_name in group_jobs]
    running_jobs = [s.name for s in states if s.status == "running"]
    pending_jobs = [s.name for s in states if s.status == "pending"]
    completed_jobs = [s.name for s in states if s.is_terminal()]

    info(f"Group '{group}' has {len(group_jobs)} jobs:")
    if running_jobs:
        warning(f"  Running: {len(running_jobs)} jobs")
    if pending_jobs:
        info(f"  Pending: {len(pending_jobs)} jobs")
    if completed_jobs:
        success(f"  Completed: {len(completed_jobs)} jobs")

    if not running_jobs and not pending_jobs:
        info("No active jobs to cancel")
        return

    # Confirm
    if not force:
        jobs_to_cancel = running_jobs + pending_jobs
        warning(f"\nWill cancel {len(jobs_to_cancel)} jobs:")
        for job_name in jobs_to_cancel:
            warning(f"  - {job_name}")

        if not typer.confirm("\nProceed with cancellation?"):
            info("Cancelled")
            raise typer.Exit(0)

    # Cancel the group
    info(f"\nCancelling jobs in group '{group}'...")
    manager.cancel_group(group)

    success(f"Cancelled all active jobs in group '{group}'")

    # Show final counts
    if running_jobs:
        warning(f"Terminated {len(running_jobs)} running jobs")
    if pending_jobs:
        info(f"Cancelled {len(pending_jobs)} pending jobs")


@app.command(name="notebook", help="Generate a Jupyter notebook for a job group")
def cmd_notebook(
    group: Annotated[str, typer.Argument(help="Job group to generate notebook for")],
    state_dir: Annotated[
        Optional[Path],
        typer.Option("--state-dir", help="Job state directory"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output notebook path"),
    ] = None,
):
    """Generate a Jupyter notebook with reward and SPS graphs for a job group.

    Creates a notebook that fetches metrics from WandB and generates visualizations
    for all training jobs in the group.

    Example:
        metta job notebook my_experiment
        metta job notebook lr_sweep --output ./my_analysis.ipynb
    """
    manager = _get_job_manager(state_dir)

    # Check if group exists
    group_jobs = manager.get_group_jobs(group)
    if not group_jobs:
        error(f"No jobs found in group '{group}'")
        raise typer.Exit(1)

    # Get job states
    job_states = [manager.get_job_state(job_name) for job_name in group_jobs]

    # Determine output path
    if output is None:
        notebooks_dir = get_repo_root() / "experiments" / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        output = notebooks_dir / f"{group}.ipynb"

    info(f"Generating notebook for {len(job_states)} jobs in group '{group}'...")

    # Generate notebook
    generate_experiment_notebook(
        notebook_path=output,
        group_name=group,
        job_states=job_states,
    )

    success(f"Notebook generated: {output}")
    info(f"\nOpen with: jupyter notebook {output}")


@app.command(name="list", help="List all job groups")
def cmd_list(
    state_dir: Annotated[
        Optional[Path],
        typer.Option("--state-dir", help="Job state directory"),
    ] = None,
):
    """List all job groups and their status.

    Shows a summary of all job groups with counts of jobs in each state.

    Example:
        metta job list
    """
    manager = _get_job_manager(state_dir)

    # Get all jobs
    from metta.jobs.job_state import JobState

    with manager._db_session() as session:
        all_jobs = session.query(JobState).all()

    if not all_jobs:
        info("No jobs found")
        return

    # Group by group name
    groups: dict[str, list[JobState]] = {}
    for job in all_jobs:
        group_name = job.config.get("group", "ungrouped")
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(job)

    # Display groups
    from rich.table import Table

    table = Table(title="Job Groups", show_header=True, header_style="bold magenta")
    table.add_column("Group", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Running", justify="right", style="yellow")
    table.add_column("Pending", justify="right", style="blue")
    table.add_column("Completed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")

    for group_name, jobs in sorted(groups.items()):
        total = len(jobs)
        running = sum(1 for j in jobs if j.status == "running")
        pending = sum(1 for j in jobs if j.status == "pending")
        completed = sum(1 for j in jobs if j.status == "completed" and j.exit_code == 0)
        failed = sum(1 for j in jobs if j.status in ("failed", "completed") and j.exit_code != 0)

        table.add_row(
            group_name,
            str(total),
            str(running) if running else "-",
            str(pending) if pending else "-",
            str(completed) if completed else "-",
            str(failed) if failed else "-",
        )

    console.print(table)


if __name__ == "__main__":
    app()
