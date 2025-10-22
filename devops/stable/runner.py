"""Orchestrates validation task execution with dependency resolution.

The runner handles:
- Running tasks in dependency order via JobManager
- Parallel execution of independent tasks
- Caching task results
- Retrying failed tasks
- Skipping tasks when dependencies fail
- State persistence
- Interactive verification of task results
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

from devops.stable.state import ReleaseState, save_state
from devops.stable.tasks import Task, TaskResult
from metta.common.util.text_styles import blue, green, red, yellow
from metta.jobs import JobManager


def _prompt_user_verification(result: TaskResult) -> bool:
    """Prompt user to verify task result.

    Returns:
        True if user accepts the result, False if user wants to mark as failed
    """
    print(f"\n{'─' * 80}")

    if result.outcome == "passed":
        prompt = blue("Accept this result? [Y/n/f(ail)] ")
    else:
        prompt = yellow("Override to pass? [y/N/a(ccept as-is)] ")

    while True:
        try:
            response = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted by user")
            sys.exit(1)

        # Handle empty response
        if not response:
            if result.outcome == "passed":
                return True  # Default accept for passed tasks
            else:
                return False  # Default reject for failed tasks

        # Handle responses
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        elif response in ("f", "fail"):
            return False
        elif response in ("a", "accept"):
            return True
        else:
            print(yellow("Please enter 'y' (yes), 'n' (no), 'f' (fail), or 'a' (accept)"))


class TaskRunner:
    """Orchestrates task execution with dependency resolution via JobManager."""

    def __init__(
        self,
        state: ReleaseState,
        job_manager: JobManager,
        interactive: bool = True,
        retry_failed: bool = False,
    ):
        """Initialize runner.

        Args:
            state: Release state to track results
            job_manager: JobManager for executing jobs
            interactive: If True, prompt for user verification after each task (default: True)
            retry_failed: If True, retry failed tasks; if False, skip them (default: False)
        """
        self.state = state
        self.job_manager = job_manager
        self.interactive = interactive
        self.retry_failed = retry_failed
        self._current_task_names: set[str] = set()
        self._batch_id = f"release_{state.version}"

    def run_all(self, tasks: list[Task]) -> None:
        """Run all tasks in parallel, respecting dependencies.

        Submits all ready tasks to JobManager and polls for completions.
        When a task completes, newly-ready tasks are submitted.
        State is saved after each task completion.
        """
        # Build set of current task names for filtering
        self._current_task_names = {task.name for task in tasks}

        # Filter out stale results from state (tasks that no longer exist in current run)
        stale_tasks = [name for name in self.state.results if name not in self._current_task_names]
        if stale_tasks:
            print(
                yellow(
                    f"🧹 Filtering out {len(stale_tasks)} stale task(s) from previous runs: {', '.join(stale_tasks)}"
                )
            )
            for name in stale_tasks:
                del self.state.results[name]
            save_state(self.state)

        # Build task lookup
        task_by_name = {task.name: task for task in tasks}

        # Track task states
        pending = set(tasks)  # Not yet submitted
        submitted = set()  # Submitted to JobManager but not complete
        completed = {}  # name -> TaskResult

        # Submit-poll loop with interrupt handling
        try:
            while pending or submitted:
                # Submit all tasks with satisfied dependencies
                ready_to_submit = [
                    task for task in pending if self._dependencies_satisfied(task, completed, task_by_name)
                ]

                for task in ready_to_submit:
                    submit_result = self._submit_task(task, completed)
                    if submit_result:  # Task was actually submitted (not skipped/cached)
                        pending.remove(task)
                        submitted.add(task)
                    else:  # Task was skipped or cached
                        result = self.state.results[task.name]
                        completed[task.name] = result
                        pending.remove(task)

                # Poll for completions
                if submitted:
                    completed_jobs = self.job_manager.poll()
                    for batch_id, name in completed_jobs:
                        if batch_id == self._batch_id and name in {t.name for t in submitted}:
                            # Get the task object
                            task = next(t for t in submitted if t.name == name)

                            # Complete the task
                            result = self._complete_task(task)
                            completed[task.name] = result
                            self.state.results[task.name] = result
                            submitted.remove(task)
                            save_state(self.state)

                # Small sleep to avoid tight polling loop
                if pending or submitted:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print(yellow("\n\n⚠️  Interrupted by user - cancelling all jobs..."))
            cancelled = self.job_manager.cancel_batch(self._batch_id)
            print(yellow(f"Cancelled {cancelled} job(s)"))
            print(yellow("State has been saved. Re-run to resume from last completed task."))
            sys.exit(130)  # Standard exit code for Ctrl+C

    def _verify_result(self, result: TaskResult) -> TaskResult:
        """Verify task result with user interaction.

        Displays detailed task information and prompts user to accept/reject.
        User can override the outcome if needed (e.g., mark passed task as failed,
        or accept a failed task that's actually okay).

        Returns:
            Modified TaskResult if user overrides, otherwise original result
        """
        # Display detailed verification info
        result.display_detailed()

        # Prompt user
        accepted = _prompt_user_verification(result)

        # Handle user decision
        if result.outcome == "passed" and not accepted:
            # User rejected a passing task - mark as failed
            print(red("\n❌ User rejected result - marking as FAILED"))
            result.outcome = "failed"
            result.error = (result.error or "") + " [User verification failed]"
        elif result.outcome == "failed" and accepted:
            # User accepted a failing task - mark as passed
            print(green("\n✅ User accepted result - marking as PASSED"))
            result.outcome = "passed"
            result.error = None

        return result

    def _dependencies_satisfied(
        self, task: Task, completed: dict[str, TaskResult], task_by_name: dict[str, Task]
    ) -> bool:
        """Check if all dependencies are resolved (ready to process).

        A dependency is resolved if it's completed or cached, regardless of outcome.
        Tasks with failed dependencies will be skipped in _submit_task().
        """
        for dep in task.dependencies:
            # Check completed dict first
            if dep.name in completed:
                # Dependency is resolved, continue
                continue

            # Check cached results
            if dep.name in self.state.results:
                cached = self.state.results[dep.name]
                # Update completed dict with cached result
                completed[dep.name] = cached
                dep.result = cached
                continue

            # Dependency not resolved yet
            return False

        return True

    def _submit_task(self, task: Task, completed: dict[str, TaskResult]) -> bool:
        """Submit task to JobManager if not already cached.

        Returns:
            True if task was submitted to JobManager, False if cached/skipped
        """
        # Check cache first
        if task.name in self.state.results:
            cached = self.state.results[task.name]
            # Skip if already passed or explicitly skipped
            if cached.outcome in ("passed", "skipped"):
                print(f"⏭️  {task.name} - cached ({cached.outcome})")
                task.result = cached
                return False
            # Retry if failed or inconclusive (only if retry_failed=True)
            if cached.outcome in ("failed", "inconclusive"):
                if self.retry_failed:
                    print(yellow(f"🔄 {task.name} - retrying previous {cached.outcome}"))
                else:
                    print(yellow(f"⏭️  {task.name} - skipping (previous {cached.outcome}, use --retry-failed to retry)"))
                    task.result = cached
                    return False

        # Check if any dependency failed
        for dep in task.dependencies:
            dep_result = completed.get(dep.name) or self.state.results.get(dep.name)
            if dep_result and dep_result.outcome != "passed":
                # Skip this task
                result = TaskResult(
                    name=task.name,
                    started_at=datetime.utcnow().isoformat(timespec="seconds"),
                    ended_at=datetime.utcnow().isoformat(timespec="seconds"),
                    exit_code=0,
                    outcome="skipped",
                    error=f"Dependency {dep.name} did not pass",
                )
                print(yellow(f"⏭️  {task.name} - SKIPPED (dependency {dep.name} failed)"))
                self.state.results[task.name] = result
                return False

        # Inject checkpoint_uri from dependencies if needed
        job_config = task.job_config
        if task.dependencies and "policy_uri" not in job_config.args:
            # Look for checkpoint_uri in dependency results
            for dep in task.dependencies:
                dep_result = completed.get(dep.name) or self.state.results.get(dep.name)
                if dep_result and "checkpoint_uri" in dep_result.artifacts:
                    job_config.args["policy_uri"] = dep_result.artifacts["checkpoint_uri"]
                    break

        # Submit to JobManager
        print(f"\n{'=' * 80}")
        print(f"🔄 Running: {task.name}")
        print(f"{'=' * 80}")

        self.job_manager.submit(self._batch_id, job_config)
        return True

    def _complete_task(self, task: Task) -> TaskResult:
        """Complete a task by getting its result from JobManager and evaluating it."""
        try:
            # Get job state from JobManager
            job_state = self.job_manager.get_job_state(self._batch_id, task.name)
            if not job_state:
                raise RuntimeError(f"Job state not found for {task.name}")

            # Evaluate result (business logic)
            result = task.evaluate_result(job_state)
            task.result = result  # Cache for downstream dependencies

            # Interactive verification
            if self.interactive:
                result = self._verify_result(result)

            return result

        except Exception as e:
            result = TaskResult(
                name=task.name,
                started_at=datetime.utcnow().isoformat(timespec="seconds"),
                ended_at=datetime.utcnow().isoformat(timespec="seconds"),
                exit_code=1,
                outcome="failed",
                error=f"Exception: {e}",
            )
            print(red(f"❌ {task.name} - ERROR: {e}"))
            import traceback

            traceback.print_exc()
            if self.interactive:
                result = self._verify_result(result)
            return result
