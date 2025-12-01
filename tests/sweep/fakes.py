"""Test fakes for sweep orchestration.

Provides in-memory Store and Dispatcher implementations to exercise orchestrator
logic without external dependencies (e.g., WandB or Skypilot).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from metta.sweep.models import JobDefinition, JobTypes, RunInfo
from metta.sweep.protocols import Dispatcher, Store


class FakeStore(Store):
    """In-memory store that returns mutable RunInfo objects for testing."""

    def __init__(self, now_fn: Callable[[], datetime] | None = None):
        self._runs: Dict[str, RunInfo] = {}
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.init_calls: list[dict[str, Any]] = []
        self.summary_updates: list[tuple[str, dict[str, Any]]] = []

    def init_run(
        self,
        run_id: str,
        group: str | None = None,
        tags: list[str] | None = None,
        initial_summary: dict[str, Any] | None = None,
    ) -> None:
        if run_id not in self._runs:
            run = RunInfo(
                run_id=run_id,
                group=group,
                tags=tags,
                created_at=self._now(),
                last_updated_at=self._now(),
                summary=dict(initial_summary) if initial_summary else {},
            )
            self._runs[run_id] = run
        self.init_calls.append({"run_id": run_id, "group": group, "tags": tags, "summary": initial_summary or {}})

    def fetch_runs(self, filters: dict) -> List[RunInfo]:
        if not filters:
            return list(self._runs.values())
        group = filters.get("group")
        if group is None:
            return list(self._runs.values())
        return [run for run in self._runs.values() if run.group == group]

    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        if run.summary is None:
            run.summary = {}
        run.summary.update(summary_update)
        run.last_updated_at = self._now()
        self.summary_updates.append((run_id, dict(summary_update)))
        return True

    # Convenience methods to drive state transitions in tests

    def start_training(self, run_id: str) -> None:
        run = self._runs[run_id]
        run.has_started_training = True
        run.last_updated_at = self._now()

    def complete_training(self, run_id: str) -> None:
        run = self._runs[run_id]
        run.has_started_training = True
        run.has_completed_training = True
        run.last_updated_at = self._now()

    def start_eval(self, run_id: str) -> None:
        run = self._runs[run_id]
        run.has_started_eval = True
        run.last_updated_at = self._now()

    def finish_eval(self, run_id: str, summary: Optional[dict[str, Any]] = None) -> None:
        run = self._runs[run_id]
        run.has_started_eval = True
        run.has_been_evaluated = True
        if summary:
            if run.summary is None:
                run.summary = {}
            run.summary.update(summary)
        run.last_updated_at = self._now()

    def fail_run(self, run_id: str, error: str | None = None) -> None:
        run = self._runs[run_id]
        run.has_failed = True
        if error:
            if run.summary is None:
                run.summary = {}
            run.summary["error"] = error
        run.last_updated_at = self._now()

    def mark_stale(self, run_id: str, stale_seconds: int = 3600) -> None:
        """Force a run into STALE status by aging last_updated_at."""
        run = self._runs[run_id]
        run.last_updated_at = self._now() - timedelta(seconds=stale_seconds)

    def ensure_run(self, run_id: str, group: str | None = None, summary: Optional[dict[str, Any]] = None) -> RunInfo:
        """Create a run if it does not exist."""
        if run_id not in self._runs:
            self.init_run(run_id=run_id, group=group, initial_summary=summary or {})
        return self._runs[run_id]


class FakeDispatcher(Dispatcher):
    """In-memory dispatcher that records jobs and optionally seeds the store."""

    def __init__(self, store: FakeStore | None = None):
        self.jobs: list[JobDefinition] = []
        self.counter = 0
        self.store = store
        self.dispatched_by_type: dict[JobTypes, list[JobDefinition]] = defaultdict(list)

    def dispatch(self, job: JobDefinition) -> str:
        self.counter += 1
        dispatch_id = f"dispatch-{self.counter}"
        self.jobs.append(job)
        self.dispatched_by_type[job.type].append(job)

        # Seed store with a run entry when training is dispatched
        if self.store and job.type == JobTypes.LAUNCH_TRAINING:
            self.store.ensure_run(run_id=job.run_id, group=job.metadata.get("group") if job.metadata else None)
        return dispatch_id
