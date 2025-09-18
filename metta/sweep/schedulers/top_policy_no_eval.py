"""Batched synchronized scheduler for adaptive experiments.

This scheduler waits for all runs to complete (including evaluation) before
generating a new batch of suggestions. This ensures perfect synchronization
between batches and is ideal for comparing hyperparameters fairly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.protocols import Store
from metta.adaptive.utils import create_training_job, generate_run_id
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.schedulers import BatchedSyncedSchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class SchedulerState:
    """State tracking for the batched synchronized scheduler.

    Tracks which runs are in training, evaluation, and completed states
    to ensure proper synchronization and prevent duplicate job dispatches.
    """

    runs_in_training: set[str] = field(default_factory=set)
    runs_completed: set[str] = field(default_factory=set)

    # We use this dictionary as an alternative to running evals.
    top_score_per_run: dict[str, float] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "runs_in_training": list(self.runs_in_training),
            "runs_completed": list(self.runs_completed),
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "SchedulerState":
        """Deserialize state from dictionary."""
        return cls(
            runs_in_training=set(data.get("runs_in_training", [])),
            runs_completed=set(data.get("runs_completed", [])),
        )


class NoEvalSweepScheduler:
    """Scheduler that generates batches of suggestions synchronously.
    Uses in-training evals instead of separate eval runs.
    """

    def __init__(
        self,
        config: BatchedSyncedSchedulerConfig,
        store: Store,
        state: SchedulerState | None = None,
    ):
        self.config = config
        self.store = store
        self.state = state or SchedulerState()
        self.optimizer = ProteinOptimizer(config.protein_config)
        logger.info(
            "[BatchedSyncedOptimizingScheduler] Initialized with max_trials=%s, batch_size=%s",
            config.max_trials,
            config.batch_size,
        )

    def _update_state_from_runs(self, runs: list[RunInfo]) -> None:
        """Update internal state from observed run statuses.

        Only updates state based on explicitly observed changes.
        Empty runs list means no updates should be made.
        """
        if not runs:
            # Check for inconsistency
            if self.state.runs_in_training or self.state.runs_completed:
                logger.warning(
                    "[BatchedSyncedOptimizingScheduler] WARNING: Received empty runs list but internal state shows "
                    f"{len(self.state.runs_in_training)} runs in training and "
                    f"{len(self.state.runs_completed)} runs in eval. This may indicate a data fetch issue."
                )
            return

        # Build a map of run_id to status and run object for easy lookup
        run_status_map = {run.run_id: run.status for run in runs}
        run_map = {run.run_id: run for run in runs}

        # Keep track of best score for each.
        for run in runs:
            if run.summary is not None:
                score_key = self.config.protein_config.metric
                score = run.summary.get(score_key, 0)
                if run.run_id in self.state.top_score_per_run:
                    self.state.top_score_per_run[run.run_id] = max(score, self.state.top_score_per_run[run.run_id])
                else:
                    self.state.top_score_per_run[run.run_id] = score

        # Update runs that have moved from training to completed/failed
        for run_id in list(self.state.runs_in_training):
            if run_id in run_status_map:
                status = run_status_map[run_id]
                run = run_map[run_id]
                if status == JobStatus.TRAINING_DONE_NO_EVAL:
                    # Training is done, update with the best score seen
                    self.state.runs_in_training.discard(run_id)
                    self.state.runs_completed.add(run_id)
                    # Use the best score we've tracked for this run
                    best_score = self.state.top_score_per_run.get(run_id, 0)
                    self.store.update_run_summary(run_id, {"sweep/score": best_score, "sweep/cost": run.cost})
                    logger.info(
                        f"[BatchedSyncedOptimizingScheduler] Run {run_id} completed training with score {best_score}"
                    )

                elif status in (JobStatus.FAILED, JobStatus.STALE):
                    # Remove from training due to failure
                    self.state.runs_in_training.discard(run_id)
                    logger.info(
                        f"[BatchedSyncedOptimizingScheduler] Run {run_id} removed from training due to status: {status}"
                    )

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """Schedule next jobs based on current state and available resources."""
        jobs: list[JobDefinition] = []

        # Update internal state from observed runs
        self._update_state_from_runs(runs)

        # 2) Check if any training jobs are still in progress - if so, wait
        # This is the key constraint for batched scheduling
        if self.state.runs_in_training:
            logger.info(
                f"[BatchedSyncedOptimizingScheduler] Waiting for {len(self.state.runs_in_training)} "
                "training job(s) to complete before next batch"
            )
            return []

        # 4) Check completion barrier and max trials
        total_created = len(runs)
        if total_created >= self.config.max_trials:
            # Allow remaining jobs (if any) to finish naturally; no new training
            logger.info(
                "[BatchedSyncedOptimizingScheduler] Max trials reached (%s). Waiting for all to complete",
                self.config.max_trials,
            )
            return []

        # 5) Generate a new batch of training suggestions
        remaining = self.config.max_trials - total_created
        capacity = max(0, available_training_slots)
        to_launch = min(self.config.batch_size, remaining, capacity)
        if to_launch <= 0:
            return []

        # Collect observations from completed runs
        observations = self._collect_observations(runs)
        logger.info(
            "[BatchedSyncedOptimizingScheduler] Requesting %s suggestion(s) from optimizer (loaded %s obs)",
            to_launch,
            len(observations),
        )

        suggestions = self.optimizer.suggest(observations, n_suggestions=to_launch)
        if not suggestions:
            logger.warning("[BatchedSyncedOptimizingScheduler] Optimizer returned no suggestions")
            return []

        # Build training jobs merging suggestion into overrides
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
            # Record suggestion directly in metadata with sweep namespace
            # This will be written to WandB summary when run is initialized
            job.metadata["sweep/suggestion"] = suggestion
            jobs.append(job)

            # Update state: add to runs_in_training
            self.state.runs_in_training.add(run_id)
            logger.info(f"[BatchedSyncedOptimizingScheduler] Scheduling training {run_id}")

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        # Count all finished runs (COMPLETED, FAILED, STALE) toward the trial limit
        finished = [r for r in runs if r.status in (JobStatus.TRAINING_DONE_NO_EVAL, JobStatus.FAILED, JobStatus.STALE)]
        is_done = len(finished) >= self.config.max_trials
        if is_done:
            completed = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
            logger.info(
                "[BatchedSyncedOptimizingScheduler] Experiment complete! %s/%s trials finished (%s successful)",
                len(finished),
                self.config.max_trials,
                len(completed),
            )
        return is_done

    def _collect_observations(self, runs: list[RunInfo]) -> list[dict[str, Any]]:
        """Extract observations from run summaries written by sweep hooks.

        Only collects observations from COMPLETED runs to ensure we're learning
        from reliable, complete training results.
        """
        obs_list: list[dict[str, Any]] = []
        for run in runs:
            # Only collect observations from completed runs (training done, no eval needed)
            if run.status != JobStatus.TRAINING_DONE_NO_EVAL:
                continue

            summary = run.summary if isinstance(run.summary, dict) else {}
            if not summary:
                continue
            score = summary.get("sweep/score")
            cost = summary.get("sweep/cost")
            suggestion = summary.get("sweep/suggestion", {})
            if score is not None:
                try:
                    obs_dict = {
                        "score": float(score),
                        "cost": float(cost) if cost is not None else 0.0,
                        "suggestion": dict(suggestion) if isinstance(suggestion, dict) else {},
                    }
                    obs_list.append(obs_dict)
                    logger.debug(
                        f"[BatchedSyncedOptimizingScheduler] Collected observation for {run.run_id}: {obs_dict}"
                    )
                except Exception as e:
                    # Skip malformed entries
                    logger.warning(
                        f"[BatchedSyncedOptimizingScheduler] Failed to collect observation for {run.run_id}: {e}"
                    )
                    continue
        return obs_list
