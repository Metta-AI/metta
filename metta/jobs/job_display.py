"""Job display utilities for showing status and progress.

Query-based display that observes JobManager state without managing jobs directly.
JobManager owns job execution and state, JobDisplay only queries and displays.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

from sky.server.common import get_server_url
from sqlmodel import Session

from metta.jobs.job_manager import JobManager
from metta.jobs.job_state import JobState


class JobDisplay:
    def __init__(self, job_manager: JobManager, group: str | None = None):
        self.job_manager = job_manager
        self.group = group
        self._start_time = time.time()

    def _should_show_training_artifacts(self, job_name: str) -> bool:
        job_state = self.job_manager.get_job_state(job_name)
        return job_state.config.is_training_job if job_state else False

    def _get_display_name(self, job_name: str) -> str:
        """Strip version prefix for cleaner display.

        Example: "v2025.10.27-1726_cpp_ci" → "cpp_ci"
        """
        if "_" in job_name:
            return job_name.split("_", 1)[1]
        return job_name

    def _format_status_with_color(self, status: str) -> str:
        """Format status string with color and symbol."""
        symbol = get_status_symbol(status)
        if status == "running":
            return f"\033[93m{symbol} {status}\033[0m"  # Yellow
        elif status == "pending":
            return f"\033[90m{symbol} {status}\033[0m"  # Gray
        else:
            return f"{symbol} {status}"

    def _extract_failure_summary(self, logs_path: str) -> list[str]:
        """Extract failure context from logs.

        Looks for:
        1. Pytest "Failing tests:" sections
        2. Error/exception patterns (Error:, Exception:, Traceback, FAILED)
        3. Last few non-empty lines if no specific error found
        """
        try:
            log_file = Path(logs_path)
            if not log_file.exists():
                return []

            lines = log_file.read_text(errors="ignore").splitlines()
            if not lines:
                return []

            # Strategy 1: Look for "Failing tests:" section (pytest output)
            for i, line in enumerate(lines):
                if "Failing tests:" in line:
                    summary_lines = []
                    for j in range(i, min(len(lines), i + 50)):
                        summary_lines.append(lines[j])
                        if j > i + 1 and not lines[j].strip() and j + 1 < len(lines):
                            next_line = lines[j + 1].strip()
                            if next_line and not next_line.startswith(" "):
                                break
                    return summary_lines

            # Strategy 2: Look for error patterns (scan last 100 lines)
            error_patterns = ["Error:", "Exception:", "Traceback", "FAILED", "FAILED:", "RuntimeError"]
            search_start = max(0, len(lines) - 100)
            for i in range(len(lines) - 1, search_start - 1, -1):
                line = lines[i]
                if any(pattern in line for pattern in error_patterns):
                    # Found error - return context (5 lines before and 10 lines after)
                    context_start = max(0, i - 5)
                    context_end = min(len(lines), i + 10)
                    return lines[context_start:context_end]

            # Strategy 3: Job failed but no explicit error - show last 5 non-empty lines
            non_empty = [line for line in lines if line.strip()]
            if non_empty:
                return non_empty[-5:]

            return []

        except Exception:
            return []

    def _display_log_tail(self, logs_path: str, num_lines: int) -> bool:
        try:
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

        # Query jobs from JobManager
        if self.group:
            jobs_dict = self.job_manager.get_group_jobs(self.group)
        else:
            jobs_dict = self.job_manager.get_all_jobs()

        jobs = list(jobs_dict.values())

        # Compute counts
        total = len(jobs)
        running = sum(1 for j in jobs if j.status == "running")
        pending = sum(1 for j in jobs if j.status == "pending")
        succeeded = sum(1 for j in jobs if j.status == "completed" and j.exit_code == 0)
        failed = sum(1 for j in jobs if j.status == "completed" and j.exit_code != 0)
        elapsed_s = time.time() - self._start_time

        # Print title if provided
        if title:
            print(f"\n{title}")
            print("=" * len(title))

        # Print summary
        print("\n┌─ Summary " + "─" * 50)
        print(f"│  Total: {total}")
        print(f"│  Running: {running}  •  Pending: {pending}")

        # Color-code success/failure counts on same line
        success_str = f"\033[92m{succeeded}\033[0m" if succeeded > 0 else f"{succeeded}"
        fail_str = f"\033[91m{failed}\033[0m" if failed > 0 else f"{failed}"
        print(f"│  Succeeded: {success_str}  •  Failed: {fail_str}")

        print(f"│  Elapsed: {format_duration(elapsed_s)}")
        print("└" + "─" * 60)
        print()

        # Separate jobs into three categories
        active_jobs = [j for j in jobs if j.status in ("running", "pending")]
        completed_jobs = [j for j in jobs if j.status == "completed" and j.exit_code == 0]
        failed_jobs = [j for j in jobs if j.status == "completed" and j.exit_code != 0]

        # Print failed jobs FIRST (at the top for maximum visibility)
        if failed_jobs:
            print("┌─ Failed Jobs " + "─" * 46)
            print("│")
            for job_state in failed_jobs:
                self._display_failed_job(job_state)

            # Close the failed jobs section
            print("└" + "─" * 60)
            print()

        # Print active jobs with full details
        if active_jobs:
            print("┌─ Active Jobs " + "─" * 46)
            print("│")
            for job_state in active_jobs:
                self._display_active_job(job_state, show_running_logs, log_tail_lines)

            # Close the active jobs section
            print("└" + "─" * 60)
            print()

        # Print completed (succeeded) jobs in condensed format at the bottom
        if completed_jobs:
            print("┌─ Completed Jobs " + "─" * 43)
            print("│")
            for job_state in completed_jobs:
                self._display_completed_job(job_state, show_artifacts)

            # Close the completed jobs section
            print("└" + "─" * 60)

    def _get_pending_reason(self, job_state: JobState) -> str:
        """Get human-readable reason why a job is pending."""

        # Check if job has dependencies
        if job_state.config.dependency_names:
            # Check which dependencies are not satisfied
            unsatisfied = []
            with Session(self.job_manager._engine) as session:
                for dep_name in job_state.config.dependency_names:
                    dep_state = session.get(JobState, dep_name)
                    if not dep_state:
                        unsatisfied.append(dep_name)
                    elif dep_state.status != "completed":
                        unsatisfied.append(dep_name)
                    elif dep_state.exit_code != 0 or dep_state.acceptance_passed is False:
                        unsatisfied.append(f"{dep_name} (failed)")

            if unsatisfied:
                # Show first dependency for brevity
                first_dep = unsatisfied[0]
                # Strip version prefix for display
                display_dep = first_dep.split("_", 1)[1] if "_" in first_dep else first_dep
                if len(unsatisfied) == 1:
                    return f"(waiting for: {display_dep})"
                else:
                    return f"(waiting for: {display_dep} +{len(unsatisfied) - 1} more)"

        # No dependencies or dependencies satisfied - waiting for worker slot
        return "(waiting for worker slot)"

    def _display_active_job(
        self,
        job_state: JobState,
        show_running_logs: bool = True,
        log_tail_lines: int = 10,
    ) -> None:
        """Display details for an active (running/pending) job."""
        name = job_state.name
        display_name = self._get_display_name(name)
        status_str = job_state.status
        job_id = job_state.job_id

        # Format status with symbol and color
        status_display = self._format_status_with_color(status_str)

        # Build line with display name and indentation
        line = f"│  {display_name:30s} {status_display:20s}"

        # Get request_id for remote job tracking
        request_id = job_state.request_id

        # For pending jobs, show why they're pending
        if status_str == "pending":
            pending_reason = self._get_pending_reason(job_state)
            if pending_reason:
                line += f" {pending_reason}"
        # Show request_id → job_id progression for remote jobs
        elif request_id and not job_id:
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

        print(line)

        # Show progress bar for training jobs
        metrics = job_state.metrics or {}
        if metrics and self._should_show_training_artifacts(name):
            # Check for progress info
            progress_info = metrics.get("_progress")

            # Show progress bar if available
            if progress_info and isinstance(progress_info, dict):
                current = progress_info.get("current_step")
                total = progress_info.get("total_steps")
                if current is not None and total is not None:
                    progress_str = format_progress_bar(current, total)
                    print(f"│    🎯 Progress: {progress_str}")

            # Show other metrics
            if len(metrics) > 1 or "_progress" not in metrics:  # Has metrics other than progress
                print("│    📊 Metrics:")
                for metric_key, metric_value in metrics.items():
                    if metric_key == "_progress":
                        continue

                    # Format value based on magnitude
                    if metric_value >= 1000:
                        value_str = f"{metric_value:,.0f}"
                    elif metric_value >= 1:
                        value_str = f"{metric_value:.2f}"
                    else:
                        value_str = str(metric_value)

                    print(f"│      • {metric_key}: {value_str}")

        # Show live logs for running jobs
        if show_running_logs and job_state.logs_path:
            if status_str == "running":
                skypilot_status = job_state.skypilot_status
                # Check if job is actually running on SkyPilot or just pending
                if request_id and skypilot_status == "PENDING":
                    print("│    🕐 Job queued on cluster, waiting to start...")
                else:
                    # Show SkyPilot dashboard link for remote jobs
                    if request_id and job_id:
                        dashboard_url = f"{get_server_url()}/dashboard/jobs/{job_id}"
                        print(f"│    🚀 SkyPilot: {dashboard_url}")
                    # Show WandB URL if this is a training job
                    if job_state.wandb_url and self._should_show_training_artifacts(name):
                        print(f"│    📊 WandB: {job_state.wandb_url}")
                    print(f"│    📝 {job_state.logs_path}")
                    print("│    📜 Live output:")
                    has_logs = self._display_log_tail(job_state.logs_path, log_tail_lines)
                    if not has_logs and request_id:
                        # Remote job running but no logs yet - just started
                        print("│      │ 🕐 Starting...")

        # Add blank line between jobs for readability
        print("│")

    def _display_completed_job(
        self,
        job_state: JobState,
        show_artifacts: bool = True,
    ) -> None:
        """Display condensed summary for a succeeded job."""
        name = job_state.name
        display_name = self._get_display_name(name)

        symbol = "✓"
        status_display = f"\033[92m{symbol}\033[0m"  # Green checkmark

        # Build condensed line: name, status, duration
        line = f"│  {status_display} {display_name:28s}"

        # Add duration
        if job_state.duration_s is not None:
            duration = format_duration(job_state.duration_s)
            line += f" [{duration}]"

        # Show artifacts for succeeded jobs (WandB, checkpoint)
        artifacts = []
        if show_artifacts:
            if job_state.wandb_url and self._should_show_training_artifacts(name):
                artifacts.append("📊")
            if job_state.checkpoint_uri:
                artifacts.append("💾")
        if artifacts:
            line += f"  {' '.join(artifacts)}"

        print(line)

    def _display_failed_job(
        self,
        job_state: JobState,
    ) -> None:
        """Display condensed summary for a failed job with error context."""
        name = job_state.name
        display_name = self._get_display_name(name)

        symbol = "✗"
        status_display = f"\033[91m{symbol}\033[0m"  # Red X

        # Build condensed line: name, status, duration, exit code
        line = f"│  {status_display} {display_name:28s}"

        # Add duration
        if job_state.duration_s is not None:
            duration = format_duration(job_state.duration_s)
            line += f" [{duration}]"

        # Show exit code inline
        exit_code = job_state.exit_code if job_state.exit_code is not None else "?"
        line += f"  (exit: {exit_code})"

        print(line)

        # Show log path and failure context
        if job_state.logs_path:
            print(f"│    📝 {job_state.logs_path}")
            # Try to extract failure summary (but don't show full tail)
            summary = self._extract_failure_summary(job_state.logs_path)
            if summary:
                # Show first 5 lines of failure summary for better context
                print("│    💥 Error context:")
                for line in summary[:5]:
                    # Strip ANSI color codes for cleaner display
                    clean_line = line.replace("\033[36m", "").replace("\033[0m", "").replace("\033[91m", "")
                    print(f"│      {clean_line[:100]}")


def get_status_symbol(status: str) -> str:
    symbols = {
        "completed": "✓",
        "succeeded": "✓",
        "failed": "✗",
        "running": "⋯",
        "pending": "○",
    }
    return symbols.get(status.lower(), "○")


def format_progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    """Format a progress bar with percentage and step counts.

    Args:
        current: Current step count
        total: Total step count
        bar_width: Width of the progress bar in characters (default: 30)

    Returns:
        Formatted progress string like "42.5% │████████████░░░░░░░░░░░░░░░░░░│ (42,500/100,000 steps)"
    """
    if total == 0:
        bar = "░" * bar_width
        return f"0.0% │{bar}│ (0/0 steps)"

    progress_pct = (current / total) * 100
    filled = int((current / total) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"{progress_pct:.1f}% │{bar}│ ({current:,}/{total:,} steps)"


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


def extract_log_tail(logs_path: str, num_lines: int = 5) -> list[str]:
    try:
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
    """Format URI with appropriate emoji prefix."""
    if uri.startswith(("wandb://", "s3://", "file://")):
        return f"📦 {uri}"
    elif uri.startswith("http"):
        return f"🔗 {uri}"
    return uri


def format_job_status_line(job_state: JobState, show_duration: bool = True) -> str:
    name = job_state.name
    status = job_state.status

    # Get status symbol and text
    symbol = get_status_symbol(status)
    if status == "completed":
        # Use is_successful which checks BOTH exit_code AND acceptance_passed
        success = job_state.is_successful
        status_text = "succeeded" if success else "failed"
        symbol = "✓" if success else "✗"
    else:
        status_text = status

    line = f"{symbol} {name} {status_text}"

    # Add duration for completed jobs
    if show_duration and status == "completed" and job_state.duration_s is not None:
        line += f" [{format_duration(job_state.duration_s)}]"

    return line
