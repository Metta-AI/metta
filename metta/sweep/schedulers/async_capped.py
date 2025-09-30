"""Asynchronous optimizing scheduler with capped eval concurrency and pending-point fantasies.

This scheduler differs from the batched, synchronized variant by:

- Scheduling training jobs as soon as slots become available (no batch barrier).
- Enforcing a strict cap on concurrent evaluations (default = 1).
- Tracking in-progress suggestions (pending runs) and applying a Constant Liar
  fantasy over them when generating new suggestions. This encourages diversity
  and reduces the risk of repeatedly proposing near-duplicates while prior
  suggestions are still running.

The fantasy method used here is the Constant Liar (CL), a standard, practical
approach in batch Bayesian optimization. For maximization, the "best" liar uses
the best observed score; for minimization, the "worst" liar uses the worst
observed score. We also support a "mean" liar as a neutral option.

Implementation notes:
- The optimizer (ProteinOptimizer) is kept unchanged. We pass fantasy
  observations into `suggest()` alongside completed observations, so the GP
  inside Protein conditions on them implicitly.
- We source the hyperparameters for in-progress runs from the run summary field
  `"sweep/suggestion"`, which is present because training jobs initialize the
  run with the suggestion in `initial_summary` (see AdaptiveController).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import Field

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.utils import (
    create_eval_job,
    create_training_job,
    generate_run_id,
)
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class AsyncCappedSchedulerConfig(Config):
    """Configuration for the asynchronous optimizing scheduler."""

    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    stats_server_uri: str | None = None
    gpus: int = 1
    nodes: int = 1
    experiment_id: str = "async"
    protein_config: Any = Field(description="ProteinConfig for optimization")
    force_eval: bool = False

    # New settings
    max_concurrent_evals: int = 1
    liar_strategy: str = "best"  # one of: "best", "mean", "worst"
    min_suggestion_distance: float = 0.0  # optional distance floor for external use


@dataclass
class AsyncSchedulerState:
    """State tracking for the asynchronous scheduler.

    - runs_in_training: run_ids currently training
    - runs_in_eval: run_ids currently evaluating
    - runs_completed: run_ids finished (COMPLETED/FAILED/STALE)
    - in_progress_suggestions: map run_id -> suggestion dict (as recorded at dispatch time)
    """

    runs_in_training: set[str] = field(default_factory=set)
    runs_in_eval: set[str] = field(default_factory=set)
    runs_completed: set[str] = field(default_factory=set)
    in_progress_suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "runs_in_training": list(self.runs_in_training),
            "runs_in_eval": list(self.runs_in_eval),
            "runs_completed": list(self.runs_completed),
            "in_progress_suggestions": self.in_progress_suggestions,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "AsyncSchedulerState":
        return cls(
            runs_in_training=set(data.get("runs_in_training", [])),
            runs_in_eval=set(data.get("runs_in_eval", [])),
            runs_completed=set(data.get("runs_completed", [])),
            in_progress_suggestions=dict(data.get("in_progress_suggestions", {})),
        )


class AsyncCappedOptimizingScheduler:
    """Asynchronous scheduler with capped eval concurrency and CL fantasies."""

    def __init__(self, config: AsyncCappedSchedulerConfig, state: AsyncSchedulerState | None = None):
        from metta.sweep.optimizer.protein import ProteinOptimizer

        self.config = config
        self.optimizer = ProteinOptimizer(config.protein_config)
        self.state = state or AsyncSchedulerState()
        self._state_initialized = False
        logger.info(
            "[AsyncCappedOptimizingScheduler] Initialized (max_trials=%s, max_concurrent_evals=%s)",
            config.max_trials,
            config.max_concurrent_evals,
        )

    # ---------- State management ----------
    def _update_state_from_runs(self, runs: list[RunInfo]) -> None:
        if not runs:
            return

        run_by_id = {r.run_id: r for r in runs}

        # Initialize from runs once (resume support)
        if not self._state_initialized:
            self.state.runs_in_training.clear()
            self.state.runs_in_eval.clear()
            self.state.runs_completed.clear()

            for run in runs:
                status = run.status
                if status == JobStatus.IN_TRAINING or status == JobStatus.PENDING or status == JobStatus.TRAINING_DONE_NO_EVAL:
                    self.state.runs_in_training.add(run.run_id)
                if status == JobStatus.IN_EVAL:
                    if self.config.force_eval:
                        logger.info(
                            "[AsyncCappedOptimizingScheduler] force_eval=True: will re-dispatch eval for %s",
                            run.run_id,
                        )
                    else:
                        self.state.runs_in_eval.add(run.run_id)
                if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                    self.state.runs_completed.add(run.run_id)

                # Recover suggestion for in-progress runs from summary if present
                if status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                    suggestion = self._extract_suggestion(run)
                    if suggestion is not None:
                        self.state.in_progress_suggestions[run.run_id] = suggestion

            self._state_initialized = True

        # Transition updates
        for run_id in list(self.state.runs_in_training):
            if run_id in run_by_id:
                st = run_by_id[run_id].status
                if st == JobStatus.IN_EVAL:
                    self.state.runs_in_training.discard(run_id)
                    self.state.runs_in_eval.add(run_id)
                elif st in (JobStatus.FAILED, JobStatus.STALE):
                    self.state.runs_in_training.discard(run_id)
                    self.state.runs_completed.add(run_id)

        for run_id in list(self.state.runs_in_eval):
            if run_id in run_by_id:
                st = run_by_id[run_id].status
                if st in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                    self.state.runs_in_eval.discard(run_id)
                    self.state.runs_completed.add(run_id)

        # Cleanup in_progress suggestions for completed runs
        for run_id in list(self.state.in_progress_suggestions.keys()):
            if run_id in self.state.runs_completed:
                del self.state.in_progress_suggestions[run_id]

    # ---------- Scheduling ----------
    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        jobs: list[JobDefinition] = []

        # Update state
        self._update_state_from_runs(runs)

        # Eval scheduling: honor max_concurrent_evals
        eval_capacity = max(0, self.config.max_concurrent_evals - len(self.state.runs_in_eval))
        # First, schedule any forced re-evaluations
        if eval_capacity > 0 and self.state.runs_pending_force_eval:
            for run_id in list(self.state.runs_pending_force_eval):
                if eval_capacity <= 0:
                    break
                run = next((r for r in runs if r.run_id == run_id), None)
                if run is None:
                    self.state.runs_pending_force_eval.discard(run_id)
                    continue
                job = create_eval_job(
                    run_id=run_id,
                    experiment_id=self.config.experiment_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                    stats_server_uri=self.config.stats_server_uri,
                    eval_overrides=self.config.eval_overrides,
                )
                jobs.append(job)
                self.state.runs_in_eval.add(run_id)
                self.state.runs_pending_force_eval.discard(run_id)
                eval_capacity -= 1
                logger.info("[AsyncCappedOptimizingScheduler] Scheduling forced re-evaluation for %s", run_id)
        # Then, schedule normal eval candidates up to remaining capacity
        if eval_capacity > 0:
            eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
            for candidate in eval_candidates:
                if eval_capacity <= 0:
                    break
                job = create_eval_job(
                    run_id=candidate.run_id,
                    experiment_id=self.config.experiment_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                    stats_server_uri=self.config.stats_server_uri,
                    eval_overrides=self.config.eval_overrides,
                )
                jobs.append(job)
                self.state.runs_in_training.discard(candidate.run_id)
                self.state.runs_in_eval.add(candidate.run_id)
                eval_capacity -= 1
                logger.info("[AsyncCappedOptimizingScheduler] Scheduling evaluation for %s", candidate.run_id)

        # If any runs still need evaluation, do not schedule new training
        if any(r.status == JobStatus.TRAINING_DONE_NO_EVAL for r in runs):
            return jobs

        # Training scheduling: fill as slots free up (only when no eval backlog)
        total_created = len(runs)
        if total_created >= self.config.max_trials:
            logger.info(
                "[AsyncCappedOptimizingScheduler] Max trials reached (%s). No further training will be scheduled",
                self.config.max_trials,
            )
            return jobs

        to_launch = min(max(0, available_training_slots), self.config.max_trials - total_created)
        if to_launch <= 0:
            return jobs

        # Gather completed observations
        observations = self._collect_observations(runs)

        # Apply Constant Liar fantasies over pending suggestions
        fantasies = self._build_constant_liar_fantasies(runs, observations)
        if fantasies:
            logger.info(
                "[AsyncCappedOptimizingScheduler] Applying %d constant-liar fantasy observation(s)",
                len(fantasies),
            )
        observations_with_fantasies = observations + fantasies

        # Request suggestions
        logger.info(
            "[AsyncCappedOptimizingScheduler] Requesting %s suggestion(s) (loaded obs=%s, fantasies=%s)",
            to_launch,
            len(observations),
            len(fantasies),
        )
        suggestions = self.optimizer.suggest(observations_with_fantasies, n_suggestions=to_launch)
        if not suggestions:
            logger.warning("[AsyncCappedOptimizingScheduler] Optimizer returned no suggestions")
            return jobs

        base_trial_num = total_created
        for i, suggestion in enumerate(suggestions):
            trial_num = base_trial_num + i + 1
            run_id = generate_run_id(self.config.experiment_id, trial_num)

            merged_overrides = dict(self.config.train_overrides)
            merged_overrides.update(suggestion)
            job = create_training_job(
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                train_entrypoint=self.config.train_entrypoint,
                gpus=self.config.gpus,
                nodes=self.config.nodes,
                stats_server_uri=self.config.stats_server_uri,
                train_overrides=merged_overrides,
            )
            job.metadata["sweep/suggestion"] = suggestion
            jobs.append(job)

            # Update state tracking
            self.state.runs_in_training.add(run_id)
            self.state.in_progress_suggestions[run_id] = suggestion
            logger.info("[AsyncCappedOptimizingScheduler] Scheduling training %s", run_id)

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        finished = [r for r in runs if r.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE)]
        if len(finished) >= self.config.max_trials:
            completed = [r for r in runs if r.status == JobStatus.COMPLETED]
            logger.info(
                "[AsyncCappedOptimizingScheduler] Experiment complete! %s/%s trials finished (%s successful)",
                len(finished),
                self.config.max_trials,
                len(completed),
            )
            return True
        return False

    # ---------- Helpers ----------
    def _collect_observations(self, runs: list[RunInfo]) -> list[dict[str, Any]]:
        obs: list[dict[str, Any]] = []
        for run in runs:
            if run.status != JobStatus.COMPLETED:
                continue
            summary = run.summary if isinstance(run.summary, dict) else {}
            if not summary:
                continue
            score = summary.get("sweep/score")
            cost = summary.get("sweep/cost")
            suggestion = summary.get("sweep/suggestion", {})
            if score is not None:
                try:
                    obs.append(
                        {
                            "score": float(score),
                            "cost": float(cost) if cost is not None else 0.0,
                            "suggestion": dict(suggestion) if isinstance(suggestion, dict) else {},
                        }
                    )
                except Exception:
                    continue
        return obs

    def _extract_suggestion(self, run: RunInfo) -> dict[str, Any] | None:
        if run.summary and isinstance(run.summary, dict):
            sg = run.summary.get("sweep/suggestion")
            if isinstance(sg, dict):
                return dict(sg)
        return None

    def _build_constant_liar_fantasies(
        self, runs: list[RunInfo], observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Construct Constant Liar fantasies for in-progress suggestions.

        We treat any run not yet completed as pending (PENDING, IN_TRAINING,
        TRAINING_DONE_NO_EVAL, or IN_EVAL). For each pending run with a recorded
        suggestion, create a pseudo-observation using a constant liar score.
        """
        # Determine liar score
        liar_score: float
        if observations:
            scores = [o.get("score", 0.0) for o in observations]
            if self.config.liar_strategy == "best":
                liar_score = max(scores)
            elif self.config.liar_strategy == "worst":
                liar_score = min(scores)
            else:  # mean
                liar_score = sum(scores) / len(scores)
        else:
            # No completed observations yet; use neutral 0.0
            liar_score = 0.0

        # Choose liar cost as mean observed cost to avoid triggering cost mask
        liar_cost: float
        if observations:
            costs = [o.get("cost", 0.0) for o in observations]
            liar_cost = float(sum(costs) / len(costs)) if costs else 0.0
        else:
            liar_cost = 0.0

        # Build set of pending run_ids
        pending_ids: set[str] = set()
        for r in runs:
            if r.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE):
                pending_ids.add(r.run_id)

        fantasies: list[dict[str, Any]] = []
        for run_id in pending_ids:
            suggestion = self.state.in_progress_suggestions.get(run_id)
            if not isinstance(suggestion, dict):
                continue
            fantasies.append({"score": liar_score, "cost": liar_cost, "suggestion": dict(suggestion)})

        return fantasies
