"""Simple job queue and dispatcher for parallel testing.

Example usage:
    # Create dispatcher
    dispatcher = JobDispatcher(name="recipe_test")

    # Add jobs
    for recipe in recipes:
        job = RemoteJob(name=recipe, module=f"recipes.{recipe}.train", args=["run=test"])
        dispatcher.add_job(job)

    # Run all jobs in parallel
    dispatcher.run_all()

    # Wait for completion
    results = dispatcher.wait_all(timeout_s=3600)

    # Check results
    dispatcher.print_summary()
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from devops.job_runner import Job, JobResult

STATE_DIR = Path("devops/job_dispatcher/state")


@dataclass
class DispatcherState:
    """State of a job dispatcher run."""

    name: str
    created_at: str
    jobs: dict[str, JobStatus] = field(default_factory=dict)
    completed: bool = False


@dataclass
class JobStatus:
    """Status of a job in the dispatcher."""

    name: str
    submitted: bool = False
    complete: bool = False
    result: Optional[dict] = None  # Serialized JobResult


class JobDispatcher:
    """Dispatch and track multiple jobs running in parallel."""

    def __init__(self, name: str, state_dir: str = "devops/job_dispatcher/state"):
        self.name = name
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, Job] = {}
        self._state = DispatcherState(
            name=name,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
        )

    def add_job(self, job: Job) -> None:
        """Add a job to the dispatcher."""
        self._jobs[job.name] = job
        self._state.jobs[job.name] = JobStatus(name=job.name)

    def run_all(self) -> None:
        """Submit all jobs (non-blocking)."""
        for name, job in self._jobs.items():
            job.submit()
            self._state.jobs[name].submitted = True
            print(f"✓ Submitted: {name}")

        self._save_state()

    def is_all_complete(self) -> bool:
        """Check if all jobs are complete."""
        return all(job.is_complete() for job in self._jobs.values())

    def wait_all(self, timeout_s: int = 3600, poll_interval_s: int = 10) -> dict[str, JobResult]:
        """Wait for all jobs to complete.

        Args:
            timeout_s: Total timeout for all jobs
            poll_interval_s: Seconds between status checks

        Returns:
            Dict mapping job name to JobResult
        """
        start_time = time.time()
        results: dict[str, JobResult] = {}

        print(f"\n⏳ Waiting for {len(self._jobs)} jobs to complete...")

        while not self.is_all_complete():
            # Check timeout
            if (time.time() - start_time) > timeout_s:
                print(f"\n⚠️  Timeout after {timeout_s}s - collecting results from completed jobs")
                break

            # Check each job
            for name, job in self._jobs.items():
                if name in results:
                    continue

                if job.is_complete():
                    result = job.get_result()
                    if result:
                        results[name] = result
                        self._state.jobs[name].complete = True
                        self._state.jobs[name].result = asdict(result)

                        icon = "✅" if result.success else "❌"
                        print(f"{icon} Completed: {name} (exit {result.exit_code})")

            self._save_state()
            time.sleep(poll_interval_s)

        # Collect any remaining results
        for name, job in self._jobs.items():
            if name not in results and job.is_complete():
                result = job.get_result()
                if result:
                    results[name] = result
                    self._state.jobs[name].complete = True
                    self._state.jobs[name].result = asdict(result)

        self._state.completed = True
        self._save_state()

        return results

    def get_results(self) -> dict[str, Optional[JobResult]]:
        """Get current results (may include None for incomplete jobs)."""
        results = {}
        for name, job in self._jobs.items():
            result = job.get_result()
            results[name] = result
        return results

    def print_summary(self) -> None:
        """Print summary of all jobs."""
        results = self.get_results()

        total = len(results)
        completed = sum(1 for r in results.values() if r is not None)
        passed = sum(1 for r in results.values() if r and r.success)
        failed = sum(1 for r in results.values() if r and not r.success)

        print("\n" + "=" * 80)
        print(f"Job Dispatcher Summary: {self.name}")
        print("=" * 80)
        print(f"Total:     {total}")
        print(f"Completed: {completed}")
        print(f"Passed:    {passed}")
        print(f"Failed:    {failed}")
        print()

        # Print per-job results
        for name, result in results.items():
            if result is None:
                print(f"  ⏳ {name:30} - In progress")
            elif result.success:
                duration = f"{result.duration_s:.0f}s" if result.duration_s else "N/A"
                print(f"  ✅ {name:30} - Passed ({duration})")
            else:
                print(f"  ❌ {name:30} - Failed (exit {result.exit_code})")

        print("=" * 80)

        # Print state file location
        state_file = self._get_state_path()
        print(f"\nState saved to: {state_file}")

    def _save_state(self) -> None:
        """Save dispatcher state to JSON."""
        state_file = self._get_state_path()
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        state_dict = {
            "name": self._state.name,
            "created_at": self._state.created_at,
            "completed": self._state.completed,
            "jobs": {name: asdict(status) for name, status in self._state.jobs.items()},
        }

        state_file.write_text(json.dumps(state_dict, indent=2))

    def _get_state_path(self) -> Path:
        """Get path to state file."""
        timestamp = self._state.created_at.replace(":", "-")
        return self.state_dir / f"{self.name}_{timestamp}.json"

    @classmethod
    def load_state(cls, state_file: str) -> Optional[DispatcherState]:
        """Load dispatcher state from JSON file."""
        path = Path(state_file)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            jobs = {
                name: JobStatus(
                    name=status["name"],
                    submitted=status["submitted"],
                    complete=status["complete"],
                    result=status["result"],
                )
                for name, status in data.get("jobs", {}).items()
            }

            return DispatcherState(
                name=data["name"],
                created_at=data["created_at"],
                jobs=jobs,
                completed=data.get("completed", False),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to load state from {path}: {e}")
            return None


# Convenience function for simple use cases
def run_jobs_parallel(jobs: list[Job], name: str = "batch", timeout_s: int = 3600) -> dict[str, JobResult]:
    """Run multiple jobs in parallel and wait for completion.

    Args:
        jobs: List of Job instances to run
        name: Name for this batch
        timeout_s: Timeout for all jobs

    Returns:
        Dict mapping job name to JobResult
    """
    dispatcher = JobDispatcher(name=name)

    for job in jobs:
        dispatcher.add_job(job)

    dispatcher.run_all()
    results = dispatcher.wait_all(timeout_s=timeout_s)
    dispatcher.print_summary()

    return results
