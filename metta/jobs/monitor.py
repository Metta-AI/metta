"""Job monitoring utilities for displaying status and progress."""

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Optional

from metta.jobs.runner import Job, JobResult


class JobMonitor:
    """Monitor multiple jobs and display their status."""

    def __init__(self, jobs: list[Job], poll_interval_s: float = 2.0):
        self.jobs = jobs
        self.poll_interval_s = poll_interval_s
        self._start_time: Optional[float] = None
        self._results: dict[str, JobResult] = {}
        self._job_ids: dict[str, str] = {}  # name -> job_id

    def start(self) -> None:
        """Start monitoring jobs (submit all)."""
        self._start_time = time.time()
        for job in self.jobs:
            job.submit()

    def wait_all(
        self,
        on_status_update: Optional[Callable[[dict[str, Any]], None]] = None,
        on_job_complete: Optional[Callable[[str, JobResult], None]] = None,
    ) -> dict[str, JobResult]:
        """Wait for all jobs to complete.

        Args:
            on_status_update: Called periodically with status dict
            on_job_complete: Called when a job completes

        Returns:
            Dict mapping job name to JobResult
        """
        if not self._start_time:
            self.start()

        completed = set()

        while len(completed) < len(self.jobs):
            # Check each job
            for job in self.jobs:
                if job.name in completed:
                    continue

                # Update job_id if available
                if hasattr(job, "_job_id") and job._job_id:
                    self._job_ids[job.name] = str(job._job_id)

                # Check if complete
                if job.is_complete():
                    result = job.get_result()
                    if result:
                        self._results[job.name] = result
                        completed.add(job.name)

                        # Call completion callback
                        if on_job_complete:
                            on_job_complete(job.name, result)

            # Call status update callback
            if on_status_update:
                status = self.get_status()
                on_status_update(status)

            # Sleep before next check
            if len(completed) < len(self.jobs):
                time.sleep(self.poll_interval_s)

        return self._results

    def get_status(self) -> dict[str, Any]:
        """Get current status of all jobs.

        Returns:
            Dict with keys:
            - total: Total number of jobs
            - completed: Number of completed jobs
            - running: Number of running jobs
            - pending: Number of pending jobs
            - succeeded: Number of successful jobs
            - failed: Number of failed jobs
            - elapsed_s: Elapsed time in seconds
            - jobs: List of job status dicts
        """
        total = len(self.jobs)
        completed = len(self._results)
        succeeded = sum(1 for r in self._results.values() if r.success)
        failed = completed - succeeded

        elapsed_s = 0.0
        if self._start_time:
            elapsed_s = time.time() - self._start_time

        # Count running vs pending
        running = 0
        pending = 0
        for job in self.jobs:
            if job.name not in self._results:
                if job._submitted:
                    running += 1
                else:
                    pending += 1

        # Build job status list
        job_statuses = []
        for job in self.jobs:
            result = self._results.get(job.name)
            job_id = self._job_ids.get(job.name)

            if result:
                status = "succeeded" if result.success else "failed"
                job_statuses.append(
                    {
                        "name": job.name,
                        "status": status,
                        "exit_code": result.exit_code,
                        "duration_s": result.duration_s,
                        "job_id": job_id,
                    }
                )
            elif job._submitted:
                job_statuses.append(
                    {
                        "name": job.name,
                        "status": "running",
                        "job_id": job_id,
                    }
                )
            else:
                job_statuses.append(
                    {
                        "name": job.name,
                        "status": "pending",
                    }
                )

        return {
            "total": total,
            "completed": completed,
            "running": running,
            "pending": pending,
            "succeeded": succeeded,
            "failed": failed,
            "elapsed_s": elapsed_s,
            "jobs": job_statuses,
        }

    def cancel_all(self) -> None:
        """Cancel all running jobs."""
        for job in self.jobs:
            if job.name not in self._results:
                try:
                    job.cancel()
                except Exception as e:
                    print(f"Failed to cancel {job.name}: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Examples:
        42.5 -> "42s"
        125.3 -> "2m 5s"
        3725.8 -> "1h 2m 5s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"

    delta = timedelta(seconds=int(seconds))
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if delta.days > 0:
        parts.append(f"{delta.days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display.

    Example: "2024-01-15 14:30:22"
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_status_table(status: dict[str, Any]) -> None:
    """Print job status as a formatted table.

    Args:
        status: Status dict from JobMonitor.get_status()
    """
    print("\nJob Status:")
    print(f"  Total: {status['total']}")
    print(f"  Running: {status['running']}")
    print(f"  Completed: {status['completed']}")
    print(f"  Succeeded: {status['succeeded']}")
    print(f"  Failed: {status['failed']}")
    print(f"  Elapsed: {format_duration(status['elapsed_s'])}")
    print()

    # Print individual job statuses
    print("Jobs:")
    for job_status in status["jobs"]:
        name = job_status["name"]
        status_str = job_status["status"]
        job_id = job_status.get("job_id")

        # Format status with color/symbol
        if status_str == "succeeded":
            symbol = "✓"
            status_display = f"{symbol} {status_str}"
        elif status_str == "failed":
            symbol = "✗"
            status_display = f"{symbol} {status_str}"
        elif status_str == "running":
            symbol = "⋯"
            status_display = f"{symbol} {status_str}"
        else:  # pending
            symbol = "○"
            status_display = f"{symbol} {status_str}"

        # Build line
        line = f"  {name:30s} {status_display:20s}"

        # Add job ID if available
        if job_id:
            line += f" (ID: {job_id})"

        # Add duration if completed
        if "duration_s" in job_status:
            duration = format_duration(job_status["duration_s"])
            line += f" [{duration}]"

        print(line)
