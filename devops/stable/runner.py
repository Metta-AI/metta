"""Orchestrates validation task execution with dependency resolution.

The runner handles:
- Running tasks in dependency order
- Caching task results
- Retrying failed tasks
- Skipping tasks when dependencies fail
- State persistence
- Interactive verification of task results
"""

from __future__ import annotations

import sys
from datetime import datetime

from devops.stable.state import ReleaseState, get_log_dir, save_state
from devops.stable.tasks import LocalCommandTask, LocalTrainingTask, Task, TaskResult
from metta.common.util.text_styles import blue, green, red, yellow


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
    """Orchestrates task execution with dependency resolution."""

    def __init__(self, state: ReleaseState, interactive: bool = True):
        """Initialize runner.

        Args:
            state: Release state to track results
            interactive: If True, prompt for user verification after each task (default: True)
        """
        self.state = state
        self.interactive = interactive

    def run_all(self, tasks: list[Task]) -> None:
        """Run all tasks, respecting dependencies.

        Tasks are executed in order, with each task waiting for its dependencies to complete.
        State is saved after each task completion.
        """
        for task in tasks:
            result = self._run_with_deps(task)
            self.state.results[task.name] = result
            save_state(self.state)

    def _run_with_deps(self, task: Task) -> TaskResult:
        """Run task after ensuring dependencies complete."""

        # Check cache
        if task.name in self.state.results:
            cached = self.state.results[task.name]
            if cached.outcome in ("passed", "skipped"):
                print(f"⏭️  {task.name} - cached ({cached.outcome})")
                task.result = cached  # Restore to task for downstream dependencies
                return cached
            print(yellow(f"🔄 {task.name} - retrying previous failure"))

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

        # Run the task
        print(f"\n{'=' * 80}")
        print(f"🔄 Running: {task.name}")
        print(f"{'=' * 80}")

        # Inject log directory based on task type
        if hasattr(task, "log_dir") and not task.log_dir:
            # Determine if task is local or remote
            if isinstance(task, (LocalCommandTask, LocalTrainingTask)):
                task.log_dir = str(get_log_dir(self.state.version, "local"))
            else:
                task.log_dir = str(get_log_dir(self.state.version, "remote"))

        try:
            result = task.run()
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
