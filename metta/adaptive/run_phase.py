"""Centralized run lifecycle phase management."""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from metta.adaptive.models import RunInfo

from metta.adaptive.models import JobStatus


class StoreProtocol(Protocol):
    """Minimal store interface needed by RunPhaseManager."""

    def update_run_summary(self, run_id: str, data: dict[str, Any]) -> bool: ...


class RunPhaseManager:
    """
    Centralized run lifecycle phase management.

    Responsible for:
    - Computing the current phase (JobStatus) of a run from its data
    - Marking phase transitions in the store
    - Recording sweep observations
    - Managing hook processing guards
    """

    STALE_THRESHOLD = timedelta(seconds=1200)

    def __init__(self, store: StoreProtocol):
        self.store = store

    # ─────────────────────────────────────────────────────────────────
    # Phase queries
    # ─────────────────────────────────────────────────────────────────

    def get_phase(self, run: "RunInfo") -> JobStatus:
        """Compute current phase (JobStatus) from run data."""
        if self._is_stale(run):
            return JobStatus.STALE
        if run.has_failed:
            return JobStatus.FAILED
        if not run.has_started_training:
            return JobStatus.PENDING
        if not run.has_completed_training:
            return JobStatus.IN_TRAINING
        # Training complete
        if not self._has_eval_started(run):
            return JobStatus.TRAINING_DONE_NO_EVAL
        if not self._has_eval_completed(run):
            return JobStatus.IN_EVAL
        return JobStatus.COMPLETED

    def has_observation(self, run: "RunInfo") -> bool:
        """Check if this run has a recorded sweep observation."""
        summary = run.summary or {}
        return summary.get("sweep/score") is not None

    # ─────────────────────────────────────────────────────────────────
    # Phase transitions
    # ─────────────────────────────────────────────────────────────────

    def mark_eval_started(self, run_id: str) -> None:
        """Record that evaluation has been dispatched."""
        self.store.update_run_summary(
            run_id,
            {
                "sweep/eval_started": True,
                "sweep/eval_started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def record_observation(
        self,
        run_id: str,
        score: float,
        cost: float,
        source: str,
    ) -> None:
        """Record the sweep observation for the optimizer."""
        self.store.update_run_summary(
            run_id,
            {
                "sweep/score": score,
                "sweep/cost": cost,
                "sweep/observation_source": source,
                "sweep/observation_recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def mark_hook_processed(self, run_id: str, hook_name: str) -> None:
        """Mark a lifecycle hook as processed to prevent re-triggering."""
        self.store.update_run_summary(
            run_id,
            {
                f"sweep/{hook_name}_processed": True,
                f"sweep/{hook_name}_processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def is_hook_processed(self, run: "RunInfo", hook_name: str) -> bool:
        """Check if a lifecycle hook has already been processed."""
        summary = run.summary or {}
        return bool(summary.get(f"sweep/{hook_name}_processed", False))

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _is_stale(self, run: "RunInfo") -> bool:
        """Check if run is stale (no updates for too long while training)."""
        if run.has_failed or run.has_completed_training:
            return False
        if run.last_updated_at is None:
            return False
        time_since_update = datetime.now(timezone.utc) - run.last_updated_at
        return time_since_update > self.STALE_THRESHOLD

    def _has_eval_started(self, run: "RunInfo") -> bool:
        """Check if evaluation has been started."""
        summary = run.summary or {}
        return bool(summary.get("sweep/eval_started"))

    def _has_eval_completed(self, run: "RunInfo") -> bool:
        """Check if evaluation has completed (observation recorded)."""
        summary = run.summary or {}
        return summary.get("sweep/score") is not None
