"""Shared scheduler state tracking for sweep schedulers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metta.adaptive.models import JobStatus, RunInfo


@dataclass
class SchedulerState:
    """Persistent state shared across sweep schedulers."""

    runs_in_training: set[str] = field(default_factory=set)
    runs_in_eval: set[str] = field(default_factory=set)
    runs_completed: set[str] = field(default_factory=set)
    runs_pending_force_eval: set[str] = field(default_factory=set)
    in_progress_suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.runs_in_training
            or self.runs_in_eval
            or self.runs_completed
            or self.runs_pending_force_eval
            or self.in_progress_suggestions
        ):
            self._initialized = True

    def refresh(self, runs: list[RunInfo], *, force_eval: bool = False) -> None:
        """Sync internal state from current runs, preserving cached suggestions."""
        run_by_id = {r.run_id: r for r in runs}

        if not self._initialized:
            self.runs_in_training.clear()
            self.runs_in_eval.clear()
            self.runs_completed.clear()
            self.runs_pending_force_eval.clear()
            self.in_progress_suggestions.clear()

        for run_id, run in run_by_id.items():
            status = run.status

            if status in (JobStatus.PENDING, JobStatus.IN_TRAINING):
                self.runs_in_training.add(run_id)
                self.runs_in_eval.discard(run_id)
                self.runs_completed.discard(run_id)
            elif status == JobStatus.TRAINING_DONE_NO_EVAL:
                if run_id in self.runs_in_eval:
                    self.runs_in_training.discard(run_id)
                else:
                    self.runs_in_training.add(run_id)
                    self.runs_in_eval.discard(run_id)
                self.runs_completed.discard(run_id)
            elif status == JobStatus.IN_EVAL:
                self.runs_in_training.discard(run_id)
                self.runs_completed.discard(run_id)
                if force_eval:
                    if run_id not in self.runs_pending_force_eval and run_id not in self.runs_in_eval:
                        self.runs_pending_force_eval.add(run_id)
                    self.runs_in_eval.add(run_id)
                else:
                    self.runs_in_eval.add(run_id)
                    self.runs_pending_force_eval.discard(run_id)
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                self.runs_completed.add(run_id)
                self.runs_in_training.discard(run_id)
                self.runs_in_eval.discard(run_id)
                self.runs_pending_force_eval.discard(run_id)

            if status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                suggestion = self._extract_suggestion(run)
                if suggestion is not None and run_id not in self.in_progress_suggestions:
                    self.in_progress_suggestions[run_id] = suggestion

        # Prune suggestions for completed runs
        for run_id in list(self.in_progress_suggestions.keys()):
            if run_id in self.runs_completed:
                del self.in_progress_suggestions[run_id]

        self._initialized = True

    def eval_capacity(self, max_concurrent_evals: int | None) -> int | None:
        if max_concurrent_evals is None:
            return None
        return max(0, max_concurrent_evals - len(self.runs_in_eval))

    def mark_eval_scheduled(self, run_id: str) -> None:
        self.runs_in_training.discard(run_id)
        self.runs_pending_force_eval.discard(run_id)
        self.runs_in_eval.add(run_id)

    def mark_training_scheduled(self, run_id: str, suggestion: dict[str, Any] | None = None) -> None:
        self.runs_in_training.add(run_id)
        if suggestion is not None:
            self.in_progress_suggestions[run_id] = dict(suggestion)

    def model_dump(self) -> dict[str, Any]:
        return {
            "runs_in_training": list(self.runs_in_training),
            "runs_in_eval": list(self.runs_in_eval),
            "runs_completed": list(self.runs_completed),
            "runs_pending_force_eval": list(self.runs_pending_force_eval),
            "in_progress_suggestions": dict(self.in_progress_suggestions),
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "SchedulerState":
        return cls(
            runs_in_training=set(data.get("runs_in_training", [])),
            runs_in_eval=set(data.get("runs_in_eval", [])),
            runs_completed=set(data.get("runs_completed", [])),
            runs_pending_force_eval=set(data.get("runs_pending_force_eval", [])),
            in_progress_suggestions=dict(data.get("in_progress_suggestions", {})),
        )

    @staticmethod
    def _extract_suggestion(run: RunInfo) -> dict[str, Any] | None:
        if run.summary and isinstance(run.summary, dict):
            sg = run.summary.get("sweep/suggestion")
            if isinstance(sg, dict):
                return dict(sg)
        return None
