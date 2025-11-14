"""High-level job workflow API.

Provides convenient wrappers that combine job manager and display functionality.
This module can safely import from both job_manager and job_display without circular imports.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from functools import partial
from typing import Callable

from metta.jobs.job_config import JobConfig
from metta.jobs.job_display import JobDisplay, report_on_jobs
from metta.jobs.job_manager import JobManager
from metta.jobs.job_state import JobStatus

# Global lock for printing to avoid interleaved output from parallel job submissions
_print_lock = threading.Lock()


def state_change_printer(job_manager: JobManager, job_name: str, old_status: str, new_status: str) -> None:
    """Print job state changes with artifacts.

    Used for non-interactive mode to show progress as jobs change state.
    Prints artifacts (WandB URLs, checkpoints, logs) when jobs complete.

    Args:
        job_manager: JobManager instance to query job state
        job_name: Name of job that changed state
        old_status: Previous status
        new_status: New status
    """
    job_state = job_manager.get_job_state(job_name)
    if not job_state:
        return

    # Use lock to prevent interleaved output from parallel callbacks
    with _print_lock:
        # Print status transition
        print(f"{job_name}: {old_status} → {new_status}", flush=True)

        # For completed jobs, show artifacts
        if new_status == JobStatus.COMPLETED:
            # Show WandB URL for training jobs
            if job_state.wandb_url:
                print(f"  🔗 {job_state.wandb_url}", flush=True)

            # Show checkpoint URI
            if job_state.checkpoint_uri:
                print(f"  📦 {job_state.checkpoint_uri}", flush=True)

            # Always show logs path
            if job_state.logs_path:
                print(f"  📝 {job_state.logs_path}", flush=True)


def submit_with_callback(
    job_manager: JobManager,
    config: JobConfig,
    on_state_change: Callable[[str, str, str], None] | None = None,
) -> None:
    """Submit a job and attach a state change callback.

    Args:
        job_manager: JobManager instance
        config: Job configuration
        on_state_change: Callback fired when job status changes
    """
    # Attach callback BEFORE submitting so we catch the pending→running transition
    if on_state_change:
        job_manager.set_state_change_callback(config.name, on_state_change)

    # Submit the job (may immediately transition pending→running)
    job_manager.submit(config)


def _monitor_non_interactive(
    submitted_jobs: set[str] | list[str],
    job_manager: JobManager,
    group: str | None = None,
    title: str | None = None,
    poll_interval: float = 1.0,
) -> None:
    """Monitor jobs in non-interactive mode (callbacks handle output).

    Just polls JobManager until all jobs complete. Callbacks print state changes.
    Used in CI environments where terminal control sequences don't work.

    Args:
        submitted_jobs: Set or list of job names to monitor
        job_manager: JobManager instance
        group: Optional group name for filtering jobs in display
        title: Optional title to show in display
        poll_interval: Seconds between manager polls
    """
    last_poll_time = 0.0
    try:
        while True:
            # Check if all jobs complete
            all_complete = all(
                job_state.status == JobStatus.COMPLETED
                for job_name in submitted_jobs
                if (job_state := job_manager.get_job_state(job_name))
            )

            if all_complete:
                break

            now = time.time()

            # Poll JobManager to start pending jobs
            if now - last_poll_time >= poll_interval:
                job_manager.poll()
                last_poll_time = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        print("   • Killing local jobs...")
        if group:
            print("   • Leaving remote jobs running (they will continue on cluster)")
            cancelled = job_manager.cancel_group(group, local_only=True)
            print(f"   • Killed {cancelled} local job(s)")
            print("\n💡 On restart: stale local jobs will be retried, running remote jobs will be reattached")
        sys.exit(130)


def _monitor_interactive(
    submitted_jobs: set[str] | list[str],
    job_manager: JobManager,
    group: str | None = None,
    title: str | None = None,
    display_interval: float = 3.0,
    poll_interval: float = 1.0,
) -> None:
    """Monitor jobs in interactive mode with live terminal updates.

    Uses in-place terminal updates with ANSI control sequences.
    Clears and redraws the display periodically for a clean UX.

    Args:
        submitted_jobs: Set or list of job names to monitor
        job_manager: JobManager instance
        group: Optional group name for filtering jobs in display
        title: Optional title to show in display
        display_interval: Seconds between display updates
        poll_interval: Seconds between manager polls
    """
    monitor = JobDisplay(job_manager, group=group)
    monitor_line_count = 0
    last_display_update = 0.0
    last_poll_time = 0.0
    first_display = True

    # Clear screen before starting monitoring
    print("\033[2J\033[H", end="", flush=True)

    try:
        while True:
            # Check if all jobs complete
            all_complete = all(
                job_state.status == JobStatus.COMPLETED
                for job_name in submitted_jobs
                if (job_state := job_manager.get_job_state(job_name))
            )

            if all_complete:
                break

            now = time.time()

            # Poll JobManager to start pending jobs
            if now - last_poll_time >= poll_interval:
                job_manager.poll()
                last_poll_time = now

            # Display status periodically
            if now - last_display_update >= display_interval or first_display:
                # Capture monitor output
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    monitor.display_status(
                        clear_screen=False,
                        title=title,
                        highlight_failures=True,
                        show_running_logs=True,
                        log_tail_lines=5,
                    )
                output = buffer.getvalue()
                new_line_count = output.count("\n")

                if not first_display and monitor_line_count > 0:
                    # Move cursor up and clear
                    print(f"\033[{monitor_line_count}A\r", end="", flush=True)
                    print("\033[J", end="", flush=True)

                # Print new output
                print(output, end="", flush=True)
                monitor_line_count = new_line_count
                last_display_update = now
                first_display = False

            time.sleep(0.1)

        # Display final status before returning (normal completion)
        print()  # Move past the live-updating section
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            monitor.display_status(
                clear_screen=False,
                title=title,
                highlight_failures=True,
                show_running_logs=False,
                log_tail_lines=0,
            )
        print(buffer.getvalue())

        # Reset all terminal formatting
        print("\033[0m", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        print("   • Killing local jobs...")
        if group:
            print("   • Leaving remote jobs running (they will continue on cluster)")
            cancelled = job_manager.cancel_group(group, local_only=True)
            print(f"   • Killed {cancelled} local job(s)")
            print("\n💡 On restart: stale local jobs will be retried, running remote jobs will be reattached")

        # Reset all terminal formatting before exit
        print("\033[0m", end="", flush=True)
        sys.exit(130)


def monitor_jobs_until_complete(
    submitted_jobs: set[str] | list[str],
    job_manager: JobManager,
    group: str | None = None,
    title: str | None = None,
    display_interval: float = 3.0,
    poll_interval: float = 1.0,
    no_interactive: bool = False,
) -> None:
    """Monitor jobs via JobDisplay until all complete.

    Shared monitoring logic used by both CI and stable release validation.
    Uses in-place terminal updates for a clean display experience (when no_interactive=False).
    When no_interactive=True, only prints updates when job states change (for CI environments).

    Args:
        submitted_jobs: Set or list of job names to monitor
        job_manager: JobManager instance
        group: Optional group name for filtering jobs in display
        title: Optional title to show in display
        display_interval: Seconds between display updates (ignored in no_interactive mode)
        poll_interval: Seconds between manager polls
        no_interactive: If True, disable live updates and only print on state changes
    """
    if no_interactive:
        _monitor_non_interactive(submitted_jobs, job_manager, group, title, poll_interval)
    else:
        _monitor_interactive(submitted_jobs, job_manager, group, title, display_interval, poll_interval)


def submit_monitor_and_report(
    job_manager: JobManager,
    jobs: list[JobConfig],
    title: str = "Job Results",
    group: str | None = None,
    no_interactive: bool = False,
) -> bool:
    """Submit jobs, monitor until complete, then report results.

    High-level wrapper that combines submit, monitor, and report steps.
    Used by both CI and stable release validation.

    Args:
        job_manager: JobManager instance
        jobs: List of JobConfig objects to run
        title: Title for final report
        group: Optional group name for job filtering
        no_interactive: If True, use callback-based monitoring instead of live display

    Returns:
        True if all jobs succeeded, False otherwise
    """
    if not jobs:
        return True

    # Create state change callback for non-interactive mode
    callback = partial(state_change_printer, job_manager) if no_interactive else None

    # Submit all jobs in parallel (SkyPilot launches can be slow)
    job_names = []

    def submit_job(job: JobConfig) -> str:
        """Submit a single job and return its name."""
        with _print_lock:
            print(f"Submitting: {job.name}")
        if callback:
            submit_with_callback(job_manager, job, on_state_change=callback)
        else:
            job_manager.submit(job)
        return job.name

    # Use ThreadPoolExecutor to parallelize submissions
    with ThreadPoolExecutor(max_workers=min(len(jobs), 10)) as executor:
        # Submit all jobs concurrently
        future_to_job = {executor.submit(submit_job, job): job for job in jobs}

        # Collect results as they complete
        for future in as_completed(future_to_job):
            job_name = future.result()
            job_names.append(job_name)

    print()  # Blank line after submissions

    # Monitor until completion
    monitor_jobs_until_complete(job_names, job_manager, group=group, title=title, no_interactive=no_interactive)

    # Report on results
    print()  # Blank line before final report
    report_on_jobs(job_manager, job_names, title=title)

    # Check if all jobs passed
    all_passed = all(
        job_state.is_successful for job_name in job_names if (job_state := job_manager.get_job_state(job_name))
    )

    return all_passed
