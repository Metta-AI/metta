"""Job monitoring utilities for displaying status and progress.

Query-based monitor that observes JobManager state without managing jobs directly.
JobManager owns job execution and state, JobMonitor only queries and displays.
"""

import time
from datetime import datetime, timedelta
from typing import Any


class JobMonitor:
    """Query-based monitor that displays JobManager state.

    Does not manage Job instances or execution - only queries JobState
    from JobManager and formats it for display.
    """

    def __init__(self, job_manager, group: str | None = None):
        """Initialize monitor.

        Args:
            job_manager: JobManager instance to query
            group: Optional group filter (only show jobs in this group)
        """
        from metta.jobs.job_manager import JobManager

        self.job_manager: JobManager = job_manager
        self.group = group
        self._start_time = time.time()

    def _should_show_training_artifacts(self, job_name: str) -> bool:
        """Check if training artifacts should be shown for this job.

        Only training jobs have WandB URLs and checkpoint URIs.

        Args:
            job_name: Name of the job

        Returns:
            True if this is a training job (determined by JobConfig.is_training_job)
        """
        job_state = self.job_manager.get_job_state(job_name)
        return job_state.config.is_training_job if job_state else False

    def _extract_failure_summary(self, logs_path: str) -> list[str]:
        """Extract failure summary from logs if available.

        Looks for common failure summary patterns like 'Failing tests:' or 'FAILED' sections.

        Args:
            logs_path: Path to log file

        Returns:
            List of relevant log lines (empty if no summary found)
        """
        try:
            from pathlib import Path

            log_file = Path(logs_path)
            if not log_file.exists():
                return []

            lines = log_file.read_text(errors="ignore").splitlines()

            # Look for "Failing tests:" section (pytest output)
            for i, line in enumerate(lines):
                if "Failing tests:" in line:
                    # Return everything from "Failing tests:" to end of log (or next section)
                    summary_lines = []
                    for j in range(i, min(len(lines), i + 50)):  # Limit to 50 lines
                        summary_lines.append(lines[j])
                        # Stop at blank line followed by another section
                        if j > i + 1 and not lines[j].strip() and j + 1 < len(lines):
                            next_line = lines[j + 1].strip()
                            if next_line and not next_line.startswith(" "):
                                break
                    return summary_lines

            # Look for other failure indicators
            # If no summary found, return empty
            return []

        except Exception:
            return []

    def _display_log_tail(self, logs_path: str, num_lines: int) -> bool:
        """Display last N lines of a log file.

        Args:
            logs_path: Path to log file
            num_lines: Number of lines to show

        Returns:
            True if logs were displayed, False if no logs available
        """
        try:
            from pathlib import Path

            log_file = Path(logs_path)
            if log_file.exists():
                lines = log_file.read_text(errors="ignore").splitlines()
                if lines:
                    # Get last N non-empty lines
                    relevant_lines = [line for line in lines if line.strip()][-num_lines:]
                    if relevant_lines:
                        for line in relevant_lines:
                            # Truncate and indent with border
                            print(f"│      │ {line[:95]}")
                        return True
        except Exception:
            pass  # Silently fail if we can't read logs
        return False

    def get_status(self) -> dict[str, Any]:
        """Get current status snapshot (non-blocking).

        Queries JobManager's database for current state of all jobs.

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
        # Query jobs from JobManager
        if self.group:
            job_states = self.job_manager.get_group_jobs(self.group)
        else:
            job_states = self.job_manager.get_all_jobs()

        # Count statuses
        total = len(job_states)
        completed = sum(1 for js in job_states.values() if js.status == "completed")
        running = sum(1 for js in job_states.values() if js.status == "running")
        pending = sum(1 for js in job_states.values() if js.status == "pending")

        # Count success/failure
        succeeded = sum(1 for js in job_states.values() if js.status == "completed" and js.exit_code == 0)
        failed = sum(1 for js in job_states.values() if js.status == "completed" and js.exit_code != 0)

        elapsed_s = time.time() - self._start_time

        # Build job status list
        job_statuses = []
        for name, job_state in job_states.items():
            status_dict = {
                "name": name,
                "status": job_state.status,
                "request_id": job_state.request_id,
                "job_id": job_state.job_id,
                "skypilot_status": job_state.skypilot_status,
            }

            # Add logs path for all jobs (if available)
            if job_state.logs_path:
                status_dict["logs_path"] = job_state.logs_path

            # Add artifacts for all jobs (if available) - show as soon as they exist
            if job_state.wandb_url:
                status_dict["wandb_url"] = job_state.wandb_url
            if job_state.checkpoint_uri:
                status_dict["checkpoint_uri"] = job_state.checkpoint_uri

            if job_state.status == "completed":
                status_dict["exit_code"] = job_state.exit_code
                status_dict["success"] = job_state.exit_code == 0

                # Calculate duration
                if job_state.started_at and job_state.completed_at:
                    try:
                        started = datetime.fromisoformat(job_state.started_at)
                        completed_at = datetime.fromisoformat(job_state.completed_at)
                        status_dict["duration_s"] = (completed_at - started).total_seconds()
                    except Exception:
                        pass

            job_statuses.append(status_dict)

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

    def display_status(
        self,
        clear_screen: bool = True,
        show_artifacts: bool = False,
        title: str | None = None,
        highlight_failures: bool = True,
        show_running_logs: bool = False,
        log_tail_lines: int = 3,
    ) -> None:
        """Display current status table (non-blocking).

        Args:
            clear_screen: Clear screen before displaying (default: True)
            show_artifacts: Show WandB URLs and checkpoints (default: False)
            title: Optional title to display above status
            highlight_failures: Highlight failed jobs in red (default: True)
            show_running_logs: Show last N lines of logs for running jobs (default: False)
            log_tail_lines: Number of log lines to show for running jobs (default: 3)
        """
        if clear_screen:
            # Clear screen: move cursor to home, clear from cursor to end of screen
            print("\033[H\033[J", end="", flush=True)

        status = self.get_status()

        # Print title if provided
        if title:
            print(f"\n{title}")
            print("=" * len(title))

        # Print summary with progress percentage
        print("\n┌─ Summary " + "─" * 50)
        print(f"│  Total: {status['total']}")

        # Calculate and display progress
        if status["total"] > 0:
            progress_pct = (status["completed"] / status["total"]) * 100
            # Create simple progress bar
            bar_width = 30
            filled = int(bar_width * progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"│  Progress: {progress_pct:.0f}% │{bar}│ ({status['completed']}/{status['total']})")

        print(f"│  Running: {status['running']}  •  Pending: {status['pending']}")

        # Color-code success/failure counts on same line
        success_str = f"\033[92m{status['succeeded']}\033[0m" if status["succeeded"] > 0 else f"{status['succeeded']}"
        fail_str = f"\033[91m{status['failed']}\033[0m" if status["failed"] > 0 else f"{status['failed']}"
        print(f"│  Succeeded: {success_str}  •  Failed: {fail_str}")

        print(f"│  Elapsed: {format_duration(status['elapsed_s'])}")
        print("└" + "─" * 60)
        print()

        # Print individual job statuses
        print("┌─ Jobs " + "─" * 53)
        print("│")
        for job_status in status["jobs"]:
            name = job_status["name"]
            # Strip version prefix for cleaner display (e.g., "v2025.10.27-1726_cpp_ci" → "cpp_ci")
            if "_" in name:
                display_name = name.split("_", 1)[1]
            else:
                display_name = name
            status_str = job_status["status"]
            job_id = job_status.get("job_id")

            # Format status with symbol and color
            symbol = get_status_symbol(status_str)
            if status_str == "completed":
                # Use success/failure for completed jobs
                if job_status.get("success"):
                    symbol = "✓"
                    status_display = f"\033[92m{symbol} succeeded\033[0m"  # Green
                else:
                    symbol = "✗"
                    if highlight_failures:
                        status_display = f"\033[91m{symbol} failed\033[0m"  # Red
                    else:
                        status_display = f"{symbol} failed"
            elif status_str == "running":
                status_display = f"\033[93m{symbol} {status_str}\033[0m"  # Yellow
            elif status_str == "pending":
                status_display = f"\033[90m{symbol} {status_str}\033[0m"  # Gray
            else:
                status_display = f"{symbol} {status_str}"

            # Build line with display name and indentation
            line = f"│  {display_name:30s} {status_display:20s}"

            # Show request_id → job_id progression for remote jobs
            request_id = job_status.get("request_id")
            if request_id and not job_id:
                # Remote job launching: have request_id but not job_id yet
                line += f" (Request: {request_id[:8]}... → waiting for job ID)"
            elif job_id:
                # Have job_id (local or remote)
                if request_id:
                    # Remote job: show request -> job progression
                    line += f" (Request: {request_id[:8]}... → Job: {job_id})"
                else:
                    # Local job: just show ID (PID)
                    line += f" (PID: {job_id})"

            # Add duration if completed
            if "duration_s" in job_status:
                duration = format_duration(job_status["duration_s"])
                line += f" [{duration}]"

            print(line)

            # Show additional details for completed jobs
            if status_str == "completed":
                # For failures, always show exit code, logs, and failure context
                if not job_status.get("success"):
                    exit_code = job_status.get("exit_code", "unknown")
                    print(f"│    ⚠️  Exit code: {exit_code}")

                    # Show log path if available
                    if "logs_path" in job_status:
                        print(f"│    📝 Logs: {job_status['logs_path']}")

                        # Try to extract failure summary first
                        summary = self._extract_failure_summary(job_status["logs_path"])
                        if summary:
                            print("│    📜 Failure summary:")
                            for line in summary[:20]:  # Limit to 20 lines
                                print(f"│      │ {line[:95]}")
                        else:
                            # Fallback: show last 10 lines
                            print("│    📜 Failure context:")
                            has_logs = self._display_log_tail(job_status["logs_path"], 10)
                            if not has_logs:
                                print("│      │ (no logs available)")

                # Show artifacts if requested
                if show_artifacts:
                    if "wandb_url" in job_status:
                        print(f"│    📊 WandB: {job_status['wandb_url']}")
                    if "checkpoint_uri" in job_status:
                        print(f"│    💾 Checkpoint: {job_status['checkpoint_uri']}")

                    # Show logs for successful jobs too when showing artifacts
                    if job_status.get("success") and "logs_path" in job_status:
                        print(f"│    📝 Logs: {job_status['logs_path']}")

            # Show live logs for running or succeeded jobs
            if show_running_logs and "logs_path" in job_status:
                if status_str == "running":
                    skypilot_status = job_status.get("skypilot_status")
                    # Check if job is actually running on SkyPilot or just pending
                    if request_id and skypilot_status == "PENDING":
                        print("│    🕐 Job queued on cluster, waiting to start...")
                    else:
                        # Show WandB URL if this is a training job
                        if "wandb_url" in job_status and self._should_show_training_artifacts(name):
                            print(f"│    📊 WandB: {job_status['wandb_url']}")
                        print(f"│    📝 {job_status['logs_path']}")
                        print("│    📜 Live output:")
                        has_logs = self._display_log_tail(job_status["logs_path"], log_tail_lines)
                        if not has_logs and request_id:
                            # Remote job running but no logs yet - just started
                            print("│      │ 🕐 Starting...")
                elif status_str == "completed" and job_status.get("success"):
                    # Show last few lines for succeeded jobs too
                    if "wandb_url" in job_status and self._should_show_training_artifacts(name):
                        print(f"│    📊 WandB: {job_status['wandb_url']}")
                    print(f"│    📝 {job_status['logs_path']}")
                    print("│    📜 Output:")
                    self._display_log_tail(job_status["logs_path"], log_tail_lines)

            # Add blank line between jobs for readability
            print("│")

        # Close the jobs section
        print("└" + "─" * 60)


def get_status_symbol(status: str) -> str:
    """Get symbol for job status.

    Args:
        status: Status string (completed, running, pending)

    Returns:
        Unicode symbol representing the status
    """
    symbols = {
        "completed": "✓",
        "succeeded": "✓",
        "failed": "✗",
        "running": "⋯",
        "pending": "○",
    }
    return symbols.get(status.lower(), "○")


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
    minutes, secs = divmod(remainder, 60)

    parts = []
    if delta.days > 0:
        parts.append(f"{delta.days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Format progress bar for display.

    Args:
        completed: Number of completed items
        total: Total number of items
        width: Width of progress bar in characters

    Returns:
        Progress bar string like "█████████░░░░░"
    """
    if total == 0:
        return "░" * width
    filled = int(width * completed / total)
    return "█" * filled + "░" * (width - filled)


def extract_log_tail(logs_path: str, num_lines: int = 5) -> list[str]:
    """Extract last N non-empty lines from log file.

    Args:
        logs_path: Path to log file
        num_lines: Number of lines to extract

    Returns:
        List of log lines (empty if file doesn't exist or can't be read)
    """
    try:
        from pathlib import Path

        log_file = Path(logs_path)
        if not log_file.exists():
            return []

        lines = log_file.read_text(errors="ignore").splitlines()
        # Get last N non-empty lines
        relevant_lines = [line for line in lines if line.strip()][-num_lines:]
        # Truncate long lines
        return [line[:100] for line in relevant_lines]
    except Exception:
        return []


def format_artifact_link(uri: str) -> str:
    """Format artifact URI with appropriate icon and styling.

    Args:
        uri: Artifact URI (wandb://, s3://, file://, http://)

    Returns:
        Formatted string with icon
    """
    if uri.startswith("wandb://"):
        return f"📦 {uri}"
    elif uri.startswith("s3://"):
        return f"📦 {uri}"
    elif uri.startswith("file://"):
        return f"📦 {uri}"
    elif uri.startswith("http"):
        return f"🔗 {uri}"
    return uri


def format_job_status_line(job_dict: dict, show_duration: bool = True) -> str:
    """Format a single job status line.

    Args:
        job_dict: Job dict from get_status_summary()
        show_duration: Whether to show duration for completed jobs

    Returns:
        Formatted status line like "✓ job_name succeeded [2m 30s]"
    """
    name = job_dict["name"]
    status = job_dict["status"]

    # Get status symbol and text
    symbol = get_status_symbol(status)
    if status == "completed":
        success = job_dict.get("exit_code") == 0
        status_text = "succeeded" if success else "failed"
        symbol = "✓" if success else "✗"
    else:
        status_text = status

    line = f"{symbol} {name} {status_text}"

    # Add duration for completed jobs
    if show_duration and status == "completed":
        started = job_dict.get("started_at")
        completed = job_dict.get("completed_at")
        if started and completed:
            try:
                from datetime import datetime

                start_dt = datetime.fromisoformat(started)
                end_dt = datetime.fromisoformat(completed)
                duration = (end_dt - start_dt).total_seconds()
                line += f" [{format_duration(duration)}]"
            except Exception:
                pass

    return line
