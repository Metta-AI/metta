"""Orchestrates validation task execution with dependency resolution.

The runner handles:
- Running tasks in dependency order via JobManager
- Parallel execution of independent tasks
- Skipping tasks when dependencies fail
- State persistence for cleanup tracking
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout

from devops.stable.display import (
    check_task_passed,
    format_task_result,
    format_task_with_acceptance,
)
from devops.stable.state import ReleaseState
from devops.stable.tasks import Task
from metta.common.util.text_styles import green, red, yellow
from metta.jobs.job_manager import JobManager
from metta.jobs.job_monitor import JobMonitor, format_progress_bar


class TaskRunner:
    """Orchestrates task execution with dependency resolution via JobManager."""

    def __init__(
        self,
        state: ReleaseState,
        job_manager: JobManager,
        retry_failed: bool = False,
        enable_monitor: bool = True,
        show_individual_results: bool = False,
    ):
        """Initialize runner.

        Args:
            state: Release state to track metadata
            job_manager: JobManager for executing jobs
            retry_failed: If True, retry failed tasks; if False, skip them (default: False)
            enable_monitor: If True, display live status updates (default: True)
            show_individual_results: If True, print detailed results when each job completes (default: False)
        """
        self.state = state
        self.job_manager = job_manager
        self.retry_failed = retry_failed
        self.show_individual_results = show_individual_results

        self.monitor = JobMonitor(job_manager, group=state.version) if enable_monitor else None
        self._last_display_update = 0.0
        self._display_interval = 3.0  # Update display every 3 seconds
        self._monitor_line_count = 0  # Track how many lines monitor has printed

    def _get_job_name(self, task: Task) -> str:
        return f"{self.state.version}_{task.name}"

    def _update_monitor_display(self) -> None:
        """Update monitor display in-place without clearing screen.

        Uses ANSI escape codes to move cursor up and rewrite status lines,
        preserving step headers and other output above the monitor.
        """
        if not self.monitor:
            return

        # Capture new output
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.monitor.display_status(
                clear_screen=False,
                title=f"Release Validation: {self.state.version}",
                highlight_failures=True,
                show_running_logs=True,
                log_tail_lines=5,
            )
        output = buffer.getvalue()
        new_line_count = output.count("\n")

        if self._monitor_line_count > 0:
            # Move cursor up to start of previous monitor output
            print(f"\033[{self._monitor_line_count}A\r", end="", flush=True)
            # Clear from cursor to end of screen
            print("\033[J", end="", flush=True)

        # Print new output
        print(output, end="", flush=True)
        self._monitor_line_count = new_line_count

    def _passed(self, job_name: str, task: Task) -> bool:
        """Check if job passed (exit_code 0 + acceptance criteria)."""
        job_state = self.job_manager.get_job_state(job_name)
        if not job_state:
            return False
        return check_task_passed(job_state)

    def _dependencies_satisfied(self, task: Task, task_by_name: dict[str, Task]) -> bool:
        """Check if all dependencies are resolved (completed with any outcome)."""
        for dep_name in task.dependency_names:
            dep_task = task_by_name.get(dep_name)
            if not dep_task:
                return False  # Dependency not found in task list

            dep_job_name = self._get_job_name(dep_task)
            dep_state = self.job_manager.get_job_state(dep_job_name)
            if not dep_state or dep_state.status != "completed":
                return False

        return True

    def _should_skip_due_to_failed_dependency(
        self, task: Task, task_by_name: dict[str, Task]
    ) -> tuple[bool, str | None]:
        """Check if task should be skipped due to failed dependency.

        Returns:
            (should_skip, reason)
        """
        for dep_name in task.dependency_names:
            dep_task = task_by_name.get(dep_name)
            if not dep_task:
                return (True, f"Dependency {dep_name} not found")

            dep_job_name = self._get_job_name(dep_task)
            if not self._passed(dep_job_name, dep_task):
                return (True, f"Dependency {dep_name} did not pass")

        return (False, None)

    def _inject_checkpoint_uri(self, task: Task, task_by_name: dict[str, Task]) -> None:
        """Inject checkpoint_uri from dependencies if needed."""
        if not task.dependency_names or "policy_uri" in task.job_config.args:
            return

        # Look for checkpoint_uri in dependency results
        for dep_name in task.dependency_names:
            dep_task = task_by_name.get(dep_name)
            if not dep_task:
                continue

            dep_job_name = self._get_job_name(dep_task)
            dep_state = self.job_manager.get_job_state(dep_job_name)
            if dep_state and dep_state.checkpoint_uri:
                task.job_config.args["policy_uri"] = dep_state.checkpoint_uri
                break

    def _display_result(self, job_name: str, task: Task) -> None:
        """Display detailed result for a completed job."""
        job_state = self.job_manager.get_job_state(job_name)
        if not job_state:
            return

        # Use job-level acceptance evaluation
        acceptance_passed = job_state.acceptance_passed if job_state.acceptance_passed is not None else True
        acceptance_error = None if acceptance_passed else "Acceptance criteria not met (see logs for details)"

        # Format and print result
        result = format_task_result(job_state, task, acceptance_passed, acceptance_error)
        print(f"\n{result}")

    def run_all(self, tasks: list[Task]) -> None:
        """Run all tasks in parallel, respecting dependencies.

        Execution flow:
        1. Filter stale submitted jobs from previous runs
        2. Submit-poll loop:
           a. Check dependencies satisfied (query JobManager)
           b. Submit ready tasks to JobManager (or skip if cached/failed dep)
           c. Poll JobManager for completions
           d. On completion: display result, save state
        3. Handle Ctrl+C: cancel all jobs in group, save state, exit

        Key behaviors:
        - Tasks with failed dependencies are skipped
        - Completed jobs are reused unless retry_failed=True
        - JobManager handles all parallelism and rate limiting
        - State saved after each completion for cleanup tracking
        """
        # Build task lookup
        task_by_name = {task.name: task for task in tasks}
        current_task_names = set(task_by_name.keys())

        # Filter out stale jobs from previous runs (query JobManager for all jobs in this group)
        all_jobs_in_group = self.job_manager.get_group_jobs(self.state.version)
        stale_job_names = []
        for job_name in all_jobs_in_group:
            # Extract task name from job name (format: {version}_{task_name})
            task_name = job_name.split("_", 1)[1] if "_" in job_name else job_name
            if task_name not in current_task_names:
                stale_job_names.append(job_name)

        if stale_job_names:
            print(
                yellow(f"🧹 Found {len(stale_job_names)} stale job(s) from previous runs: {', '.join(stale_job_names)}")
            )

        # Track which tasks we've already tried to submit (to avoid re-submission)
        submitted_tasks = set()

        # Submit-poll loop with interrupt handling
        first_submission = True
        last_poll_time = 0.0
        poll_interval = 1.0  # Poll JobManager every 1 second for job completions
        try:
            while True:
                # Get current state from JobManager (source of truth)
                group_jobs = self.job_manager.get_group_jobs(self.state.version)

                # Determine which tasks still need work
                incomplete_tasks = []
                for task in tasks:
                    job_name = self._get_job_name(task)

                    # Task needs work if:
                    # 1. Not yet submitted, OR
                    # 2. Submitted but not completed yet
                    if task not in submitted_tasks:
                        incomplete_tasks.append(task)
                    elif job_name in group_jobs and group_jobs[job_name].status != "completed":
                        incomplete_tasks.append(task)

                # Exit if all tasks are complete
                if not incomplete_tasks:
                    break

                # Submit all tasks with satisfied dependencies that we haven't submitted yet
                ready_to_submit = [
                    task
                    for task in tasks
                    if task not in submitted_tasks and self._dependencies_satisfied(task, task_by_name)
                ]

                for task in ready_to_submit:
                    self._submit_task(task, task_by_name)
                    submitted_tasks.add(task)

                # Show initial status immediately after first batch of submissions
                if first_submission and submitted_tasks and self.monitor:
                    first_submission = False
                    # Display monitor for the first time (don't update in-place yet
                    # since we have submission messages above)
                    # Just print normally, then subsequent updates will be in-place
                    self.monitor.display_status(
                        clear_screen=False,
                        title=f"Release Validation: {self.state.version}",
                        highlight_failures=True,
                        show_running_logs=True,
                        log_tail_lines=3,
                    )

                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        self.monitor.display_status(
                            clear_screen=False,
                            title=f"Release Validation: {self.state.version}",
                            highlight_failures=True,
                            show_running_logs=True,
                            log_tail_lines=3,
                        )
                    self._monitor_line_count = buffer.getvalue().count("\n")
                    self._last_display_update = time.time()
                    last_poll_time = time.time()

                now = time.time()

                # Call poll() to start pending jobs and update internal state
                # (we don't use the return value since we query state directly)
                if now - last_poll_time >= poll_interval:
                    self.job_manager.poll()
                    last_poll_time = now

                # Display result for newly completed jobs if requested
                if self.show_individual_results and now - last_poll_time < 0.1:  # Just polled
                    for task in submitted_tasks:
                        job_name = self._get_job_name(task)
                        if job_name in group_jobs and group_jobs[job_name].status == "completed":
                            # Check if we already displayed this one
                            if not hasattr(task, "_result_displayed"):
                                self._display_result(job_name, task)
                                task._result_displayed = True  # Mark as displayed

                # Display status periodically (fast - just reads from database)
                if self.monitor and now - self._last_display_update >= self._display_interval:
                    # Don't clear screen during execution - update in place
                    # This preserves step headers and previous output
                    self._update_monitor_display()
                    self._last_display_update = now

                # Small sleep to avoid tight polling loop
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(yellow("\n\n⚠️  Interrupted by user (Ctrl+C)"))
            print(yellow("   • Killing local jobs..."))
            print(yellow("   • Leaving remote jobs running (they will continue on cluster)"))
            cancelled = self.job_manager.cancel_group(self.state.version, local_only=True)
            print(yellow(f"   • Killed {cancelled} local job(s)"))
            print(yellow("\n💡 On restart: stale local jobs will be retried, running remote jobs will be reattached"))
            sys.exit(130)  # Standard exit code for Ctrl+C

        # Display final summary using composable display
        print("\n" + "=" * 80)
        print(f"Release Validation Complete: {self.state.version}")
        print("=" * 80)

        # Get aggregated summary
        summary = self.job_manager.get_status_summary(group=self.state.version)

        # Show progress bar
        progress = format_progress_bar(summary["completed"], summary["total"])
        pct = (summary["completed"] / summary["total"] * 100) if summary["total"] > 0 else 0
        print(f"\nProgress: {summary['completed']}/{summary['total']} ({pct:.0f}%)")
        print(f"{progress}")
        print(f"Succeeded: {green(str(summary['succeeded']))}  Failed: {red(str(summary['failed']))}")
        print()

        # Show each task with integrated job status + acceptance
        # Build task lookup
        task_by_name = {task.name: task for task in tasks}

        for job_dict in summary["jobs"]:
            # Extract task name from job name (format: {version}_{task_name})
            job_name = job_dict["name"]
            task_name = job_name.split("_", 1)[1] if "_" in job_name else job_name
            task = task_by_name.get(task_name)

            if not task:
                continue

            # Get full job state for metrics
            job_state = self.job_manager.get_job_state(job_name)
            if not job_state:
                continue

            # Use composable display
            display = format_task_with_acceptance(job_dict, job_state)
            print(display)
            print()

    def _submit_task(self, task: Task, task_by_name: dict[str, Task]) -> bool:
        """Submit task to JobManager if not already completed.

        Returns:
            True if task was submitted to JobManager, False if cached/skipped
        """
        job_name = self._get_job_name(task)

        # Check if job already completed
        existing_state = self.job_manager.get_job_state(job_name)
        if existing_state and existing_state.status == "completed":
            # Check if we should retry
            # Exit codes: 0=success, >0=error, -1=abnormal termination, 130=cancelled
            # Auto-retry abnormal terminations (stale local jobs from Ctrl+C)
            # Otherwise respect retry_failed flag
            should_retry = False
            if existing_state.exit_code == -1:
                print(yellow(f"🔄 {task.name} - retrying after abnormal termination"))
                should_retry = True
            elif not self.retry_failed or (existing_state.exit_code == 0 and self._passed(job_name, task)):
                print(f"⏭️  {task.name} - already completed (use --retry to retry)")
                return False
            else:
                print(yellow(f"🔄 {task.name} - retrying previous run"))
                should_retry = True

            # Delete old job state so we can submit a fresh one
            if should_retry:
                self.job_manager.delete_job(job_name)

        # Check if job is already running
        if existing_state and existing_state.status in ("pending", "running"):
            print(f"⏭️  {task.name} - already running (status: {existing_state.status}), attaching")
            return True  # Mark as submitted so we track it

        # Check if any dependency failed
        should_skip, reason = self._should_skip_due_to_failed_dependency(task, task_by_name)
        if should_skip:
            print(yellow(f"⏭️  {task.name} - SKIPPED ({reason})"))
            return False

        # Inject checkpoint_uri from dependencies if needed
        self._inject_checkpoint_uri(task, task_by_name)

        # Set unique job name and group for JobManager
        task.job_config.name = job_name
        task.job_config.group = self.state.version

        # Submit to JobManager (monitor will show status)
        try:
            self.job_manager.submit(task.job_config)
            # Print quick feedback during submission
            print(f"  • Submitted {task.name}")
            return True
        except ValueError as e:
            # This shouldn't happen since we checked above, but handle it gracefully
            print(yellow(f"⚠️  {task.name} - failed to submit: {e}"))
            return False
