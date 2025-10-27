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

from devops.stable.state import ReleaseState
from devops.stable.tasks import AcceptanceRule, Task
from metta.common.util.text_styles import blue, cyan, green, magenta, red, yellow
from metta.jobs.job_manager import JobManager
from metta.jobs.job_monitor import JobMonitor
from metta.jobs.job_state import JobState


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

        # Create monitor for live status display (optional, for tests)
        self.monitor = JobMonitor(job_manager, group=state.version) if enable_monitor else None
        self._last_display_update = 0.0
        self._display_interval = 2.0  # Update display every 2 seconds
        self._monitor_line_count = 0  # Track how many lines monitor has printed

    def _get_job_name(self, task: Task) -> str:
        """Generate globally unique job name for JobManager.

        Format: {version}_{task_name}
        Example: "v2025.10.22_metta_ci"
        """
        return f"{self.state.version}_{task.name}"

    def _update_monitor_display(self) -> None:
        """Update monitor display in-place without clearing screen.

        Uses ANSI escape codes to move cursor up and rewrite status lines,
        preserving step headers and other output above the monitor.
        """
        if not self.monitor:
            return

        # If this is the first update, just print normally
        if self._monitor_line_count == 0:
            # Capture output to count lines
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.monitor.display_status(
                    clear_screen=False,
                    title=f"Release Validation: {self.state.version}",
                    highlight_failures=True,
                )
            output = buffer.getvalue()
            print(output, end="")
            self._monitor_line_count = output.count("\n")
        else:
            # Move cursor up by the number of lines we printed last time
            print(f"\033[{self._monitor_line_count}A", end="", flush=True)

            # Print updated status (will overwrite previous lines)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.monitor.display_status(
                    clear_screen=False,
                    title=f"Release Validation: {self.state.version}",
                    highlight_failures=True,
                )
            output = buffer.getvalue()
            print(output, end="", flush=True)
            self._monitor_line_count = output.count("\n")

    def _evaluate_acceptance(self, job_state: JobState, rules: list[AcceptanceRule]) -> tuple[bool, str | None]:
        """Evaluate acceptance criteria against job metrics.

        Returns:
            (passed, error_message)
        """
        if not rules:
            return (True, None)

        failures: list[str] = []
        for key, op, expected in rules:
            if key not in job_state.metrics:
                failures.append(f"{key}: metric missing (expected {op.__name__} {expected})")
                continue
            if not op(job_state.metrics[key], expected):
                failures.append(f"{key}: expected {op.__name__} {expected}, saw {job_state.metrics[key]}")

        if failures:
            return (False, "; ".join(failures))
        return (True, None)

    def _passed(self, job_name: str, task: Task) -> bool:
        """Check if job passed (exit_code 0 + acceptance criteria)."""
        job_state = self.job_manager.get_job_state(job_name)
        if not job_state or job_state.exit_code != 0:
            return False

        # Check acceptance criteria
        passed, _ = self._evaluate_acceptance(job_state, task.acceptance)
        return passed

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

        print(f"\n{'=' * 80}")
        print(blue(f"📋 TASK RESULT: {task.name}"))
        print(f"{'=' * 80}\n")

        # Outcome
        passed = self._passed(job_name, task)
        if passed:
            print(green("✅ Outcome: PASSED"))
        else:
            print(red("❌ Outcome: FAILED"))

        # Exit code
        if job_state.exit_code != 0:
            print(red(f"⚠️  Exit Code: {job_state.exit_code}"))
        else:
            print(green(f"✓ Exit Code: {job_state.exit_code}"))

        # Acceptance check
        if task.acceptance:
            acceptance_passed, error = self._evaluate_acceptance(job_state, task.acceptance)
            if not acceptance_passed:
                print(red(f"\n❗ Acceptance Criteria Failed: {error}"))

        # Metrics
        if job_state.metrics:
            print("\n📊 Metrics:")
            for key, value in job_state.metrics.items():
                print(f"   • {key}: {value:.4f}")

        # Artifacts with highlighting
        artifacts = {}
        if job_state.wandb_run_id:
            artifacts["wandb_run_id"] = job_state.wandb_run_id
        if job_state.wandb_url:
            artifacts["wandb_url"] = job_state.wandb_url
        if job_state.checkpoint_uri:
            artifacts["checkpoint_uri"] = job_state.checkpoint_uri

        if artifacts:
            print("\n📦 Artifacts:")
            for key, value in artifacts.items():
                highlighted = self._highlight_artifact(value)
                print(f"   • {key}: {highlighted}")

        # Job ID and logs path
        if job_state.job_id:
            print(f"\n🆔 Job ID: {job_state.job_id}")

        if job_state.logs_path:
            print(f"📝 Logs: {job_state.logs_path}")

    def _highlight_artifact(self, value: str) -> str:
        if value.startswith("wandb://"):
            return magenta(f"📦 {value}")
        elif value.startswith("s3://"):
            return magenta(f"📦 {value}")
        elif value.startswith("file://"):
            return magenta(f"📦 {value}")
        elif value.startswith("http"):
            return cyan(f"🔗 {value}")
        return value

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

        # Track task states
        pending = set(tasks)  # Not yet submitted
        submitted = set()  # Submitted to JobManager but not complete
        completed = set()  # Completed (any outcome)

        # Submit-poll loop with interrupt handling
        try:
            while pending or submitted:
                # Submit all tasks with satisfied dependencies
                ready_to_submit = [task for task in pending if self._dependencies_satisfied(task, task_by_name)]

                for task in ready_to_submit:
                    submit_result = self._submit_task(task, task_by_name)
                    if submit_result:  # Task was actually submitted (not skipped/cached)
                        pending.remove(task)
                        submitted.add(task)
                    else:  # Task was skipped or cached
                        completed.add(task)
                        pending.remove(task)

                # Poll for completions
                if submitted:
                    completed_job_names = self.job_manager.poll()
                    for job_name in completed_job_names:
                        # Find the task with this job name
                        task = next((t for t in submitted if self._get_job_name(t) == job_name), None)
                        if task:
                            # Display result only if requested (otherwise monitor shows status)
                            if self.show_individual_results:
                                self._display_result(job_name, task)
                            completed.add(task)
                            submitted.remove(task)

                # Display status periodically (throttled)
                if self.monitor:
                    now = time.time()
                    if now - self._last_display_update >= self._display_interval:
                        # Don't clear screen during execution - update in place
                        # This preserves step headers and previous output
                        self._update_monitor_display()
                        self._last_display_update = now

                # Small sleep to avoid tight polling loop
                if pending or submitted:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print(yellow("\n\n⚠️  Interrupted by user (Ctrl+C)"))
            print(yellow("   • Killing local jobs..."))
            print(yellow("   • Leaving remote jobs running (they will continue on cluster)"))
            cancelled = self.job_manager.cancel_group(self.state.version, local_only=True)
            print(yellow(f"   • Killed {cancelled} local job(s)"))
            print(yellow("\n💡 On restart: stale local jobs will be retried, running remote jobs will be reattached"))
            sys.exit(130)  # Standard exit code for Ctrl+C

        # Display final summary
        if self.monitor:
            print("\n")
            self.monitor.display_status(
                clear_screen=False,
                title=f"Release Validation Complete: {self.state.version}",
                highlight_failures=True,
                show_artifacts=True,
            )

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
            if existing_state.exit_code == -1:
                print(yellow(f"🔄 {task.name} - retrying after abnormal termination"))
            elif not self.retry_failed or (existing_state.exit_code == 0 and self._passed(job_name, task)):
                print(f"⏭️  {task.name} - already completed (use --retry-failed to retry)")
                return False
            else:
                print(yellow(f"🔄 {task.name} - retrying previous run"))

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

        # Submit to JobManager
        print(f"\n{'=' * 80}")
        print(f"🔄 Running: {task.name}")
        print(f"{'=' * 80}")

        try:
            self.job_manager.submit(task.job_config)
            return True
        except ValueError as e:
            # This shouldn't happen since we checked above, but handle it gracefully
            print(yellow(f"⚠️  {task.name} - failed to submit: {e}"))
            return False
