"""High-level job workflow API.

Provides convenient wrappers that combine job manager and display functionality.
This module can safely import from both job_manager and job_display without circular imports.
"""

from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout

from metta.jobs.job_config import JobConfig
from metta.jobs.job_display import JobDisplay, report_on_jobs
from metta.jobs.job_manager import JobManager


def monitor_jobs_until_complete(
    submitted_jobs: set[str] | list[str],
    job_manager: JobManager,
    group: str | None = None,
    title: str | None = None,
    display_interval: float = 3.0,
    poll_interval: float = 1.0,
) -> None:
    """Monitor jobs via JobDisplay until all complete.

    Shared monitoring logic used by both CI and stable release validation.
    Uses in-place terminal updates for a clean display experience.

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
                job_state.status == "completed"
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


def submit_monitor_and_report(
    job_manager: JobManager,
    jobs: list[JobConfig],
    title: str = "Job Results",
    group: str | None = None,
) -> bool:
    """Submit jobs, monitor until complete, then report results.

    High-level wrapper that combines submit, monitor, and report steps.
    Used by both CI and stable release validation.

    Args:
        job_manager: JobManager instance
        jobs: List of JobConfig objects to run
        title: Title for final report
        group: Optional group name for job filtering

    Returns:
        True if all jobs succeeded, False otherwise
    """
    if not jobs:
        return True

    # Submit all jobs in parallel (SkyPilot launches can be slow)
    job_names = []

    def submit_job(job: JobConfig) -> str:
        """Submit a single job and return its name."""
        print(f"Submitting: {job.name}")
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
    monitor_jobs_until_complete(job_names, job_manager, group=group, title=title)

    # Report on results
    report_on_jobs(job_manager, job_names, title=title)

    # Check if all jobs passed
    all_passed = all(
        job_state.is_successful for job_name in job_names if (job_state := job_manager.get_job_state(job_name))
    )

    return all_passed
