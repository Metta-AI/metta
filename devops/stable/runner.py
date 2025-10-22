"""Orchestrates validation task execution with dependency resolution.

The runner handles:
- Running tasks in dependency order via JobManager
- Caching task results
- Retrying failed tasks
- Skipping tasks when dependencies fail
- State persistence
- Interactive verification of task results
"""

from __future__ import annotations

import sys
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
    ):
        """Initialize runner.

        Args:
            state: Release state to track results
            job_manager: JobManager for executing jobs
            interactive: If True, prompt for user verification after each task (default: True)
        """
        self.state = state
        self.job_manager = job_manager
        self.interactive = interactive
        self._current_task_names: set[str] = set()
        self._batch_id = f"release_{state.version}"

    def run_all(self, tasks: list[Task]) -> None:
        """Run all tasks, respecting dependencies.

        Tasks are executed in order, with each task waiting for its dependencies to complete.
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

        for task in tasks:
            result = self._run_with_deps(task)
            self.state.results[task.name] = result
            save_state(self.state)

    def _run_with_deps(self, task: Task) -> TaskResult:
        """Run task after ensuring dependencies complete."""

        # Check cache
        if task.name in self.state.results:
            cached = self.state.results[task.name]
            # Skip if already passed or explicitly skipped
            if cached.outcome in ("passed", "skipped"):
                print(f"⏭️  {task.name} - cached ({cached.outcome})")
                task.result = cached  # Restore for downstream dependencies
                return cached
            # Retry if failed or inconclusive
            print(yellow(f"🔄 {task.name} - retrying previous {cached.outcome}"))

        # Run dependencies first
        for dep in task.dependencies:
            dep_result = self._run_with_deps(dep)
            if dep_result.outcome != "passed":
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
                return result

        # Run the task via JobManager
        print(f"\n{'=' * 80}")
        print(f"🔄 Running: {task.name}")
        print(f"{'=' * 80}")

        try:
            # Inject checkpoint_uri from dependencies if needed
            job_config = task.job_config
            if task.dependencies and "policy_uri" not in job_config.args:
                # Look for checkpoint_uri in dependency results
                for dep in task.dependencies:
                    if dep.result and "checkpoint_uri" in dep.result.artifacts:
                        job_config.args["policy_uri"] = dep.result.artifacts["checkpoint_uri"]
                        break

            # Submit to JobManager
            self.job_manager.submit(self._batch_id, job_config)

            # Wait for job to complete
            job_state = self.job_manager.wait_for_job(self._batch_id, task.name)

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
