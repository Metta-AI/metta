"""Batched synchronized scheduler for adaptive experiments.

This scheduler waits for all runs to complete (including evaluation) before
generating a new batch of suggestions. This ensures perfect synchronization
between batches and is ideal for comparing hyperparameters fairly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.utils import create_eval_job, create_training_job, generate_run_id
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class BatchedSyncedSchedulerConfig(Config):
    """Configuration for batched synchronized scheduler."""

    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "eval"
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    stats_server_uri: str | None = None
    gpus: int = 1
    nodes: int = 1
    batch_size: int = 4
    experiment_id: str = "batched_synced"
    protein_config: Any = Field(description="ProteinConfig for optimization")
    force_eval: bool = False  # Force re-dispatch of IN_EVAL jobs on relaunch


@dataclass
class SchedulerState:
    """State tracking for the batched synchronized scheduler.

    Tracks which runs are in training, evaluation, and completed states
    to ensure proper synchronization and prevent duplicate job dispatches.
    """

    runs_in_training: set[str] = field(default_factory=set)
    runs_in_eval: set[str] = field(default_factory=set)
    runs_completed: set[str] = field(default_factory=set)
    # Runs detected in IN_EVAL at startup that should be re-evaluated
    runs_pending_force_eval: set[str] = field(default_factory=set)

    def model_dump(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "runs_in_training": list(self.runs_in_training),
            "runs_in_eval": list(self.runs_in_eval),
            "runs_completed": list(self.runs_completed),
            "runs_pending_force_eval": list(self.runs_pending_force_eval),
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "SchedulerState":
        """Deserialize state from dictionary."""
        return cls(
            runs_in_training=set(data.get("runs_in_training", [])),
            runs_in_eval=set(data.get("runs_in_eval", [])),
            runs_completed=set(data.get("runs_completed", [])),
            runs_pending_force_eval=set(data.get("runs_pending_force_eval", [])),
        )


class BatchedSyncedOptimizingScheduler:
    """Scheduler that generates batches of suggestions synchronously.

    Key behaviors:
    - Only generates new suggestions when ALL current runs (including evals) are complete
    - Schedules evals for any runs with training complete and eval not yet started
    - Generates up to `batch_size` training jobs at a time (bounded by available slots)
    - Suggestions come from a stateless Optimizer; observations are read from run summaries
    - Maintains stateful tracking of runs to prevent duplicate dispatches
    """

    def __init__(
        self,
        config: BatchedSyncedSchedulerConfig,
        state: SchedulerState | None = None,
    ):
        from metta.sweep.optimizer.protein import ProteinOptimizer

        self.config = config
        self.optimizer = ProteinOptimizer(config.protein_config)
        self.state = state or SchedulerState()
        self._state_initialized = False  # Track if we've initialized state from runs
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
            if self.state.runs_in_training or self.state.runs_in_eval:
                logger.warning(
                    "[BatchedSyncedOptimizingScheduler] WARNING: Received empty runs list but internal state shows "
                    f"{len(self.state.runs_in_training)} runs in training and "
                    f"{len(self.state.runs_in_eval)} runs in eval. This may indicate a data fetch issue."
                )
            return

        # Build a map of run_id to status for easy lookup
        run_status_map = {run.run_id: run.status for run in runs}

        # On first call after restart with empty state, rebuild from observed run statuses
        # This ensures we don't lose track of IN_EVAL runs after a restart
        # Only do this if state is completely empty (fresh start or restart)
        if not self._state_initialized:
            is_empty_state = (
                not self.state.runs_in_training and not self.state.runs_in_eval and not self.state.runs_completed
            )
            if is_empty_state and runs:
                logger.info("[BatchedSyncedOptimizingScheduler] Empty state detected, initializing from observed runs")
                force_eval_count = 0
                for run_id, status in run_status_map.items():
                    if status == JobStatus.IN_TRAINING:
                        self.state.runs_in_training.add(run_id)
                        logger.info(f"[BatchedSyncedOptimizingScheduler] Found run {run_id} in training")
                    elif status == JobStatus.IN_EVAL:
                        if self.config.force_eval:
                            # Do not mark as in_eval so we can re-dispatch an eval job
                            logger.info(
                                f"[BatchedSyncedOptimizingScheduler] Found run {run_id} in evaluation - "
                                "FORCING RE-EVALUATION due to force_eval=True"
                            )
                            self.state.runs_pending_force_eval.add(run_id)
                            force_eval_count += 1
                        else:
                            self.state.runs_in_eval.add(run_id)
                            logger.info(f"[BatchedSyncedOptimizingScheduler] Found run {run_id} in evaluation")
                    elif status == JobStatus.COMPLETED:
                        self.state.runs_completed.add(run_id)
                        logger.info(f"[BatchedSyncedOptimizingScheduler] Found run {run_id} completed")
                if self.config.force_eval and force_eval_count > 0:
                    logger.info(
                        "[BatchedSyncedOptimizingScheduler] force_eval=True: Will re-dispatch "
                        f"{force_eval_count} evaluation job(s)"
                    )
                    # Clear the flag after first use to prevent continuous re-dispatching
                    self.config.force_eval = False
                logger.info(
                    f"[BatchedSyncedOptimizingScheduler] State initialized: "
                    f"{len(self.state.runs_in_training)} training, "
                    f"{len(self.state.runs_in_eval)} evaluating, "
                    f"{len(self.state.runs_completed)} completed"
                )
            self._state_initialized = True

        # Update runs that have moved from training to completed/failed
        for run_id in list(self.state.runs_in_training):
            if run_id in run_status_map:
                status = run_status_map[run_id]
                if status == JobStatus.TRAINING_DONE_NO_EVAL:
                    # Training is done, ready for eval (keep in runs_in_training until eval is dispatched)
                    pass
                elif status in (JobStatus.FAILED, JobStatus.STALE):
                    # Remove from training due to failure
                    self.state.runs_in_training.discard(run_id)
                    logger.info(
                        f"[BatchedSyncedOptimizingScheduler] Run {run_id} removed from training due to status: {status}"
                    )

        # Update runs that have moved from eval to completed
        for run_id in list(self.state.runs_in_eval):
            if run_id in run_status_map:
                status = run_status_map[run_id]
                if status == JobStatus.COMPLETED:
                    self.state.runs_in_eval.discard(run_id)
                    self.state.runs_completed.add(run_id)
                    logger.info(f"[BatchedSyncedOptimizingScheduler] Run {run_id} completed evaluation")
                elif status == JobStatus.FAILED:
                    # Eval failed, move to completed anyway (won't retry)
                    self.state.runs_in_eval.discard(run_id)
                    self.state.runs_completed.add(run_id)
                    logger.info(f"[BatchedSyncedOptimizingScheduler] Run {run_id} eval failed, marking as completed")

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """Schedule next jobs based on current state and available resources."""
        jobs: list[JobDefinition] = []

        # Update internal state from observed runs
        self._update_state_from_runs(runs)

        # 0) Handle any runs pending forced re-evaluation first
        if self.state.runs_pending_force_eval:
            for run_id in list(self.state.runs_pending_force_eval):
                # Only schedule if the run still exists
                run = next((r for r in runs if r.run_id == run_id), None)
                if run is None:
                    # Drop if not visible in current run list
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
                # Mark as in eval to avoid duplicate dispatch
                self.state.runs_in_eval.add(run_id)
                self.state.runs_pending_force_eval.discard(run_id)
                logger.info(f"[BatchedSyncedOptimizingScheduler] Scheduling forced re-evaluation for {run_id}")

            if jobs:
                return jobs

        # 1) Schedule evals for any runs with training done but no eval yet
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
        for run in eval_candidates:
            # Check if we've already dispatched eval for this run
            if run.run_id in self.state.runs_in_eval:
                logger.debug(f"[BatchedSyncedOptimizingScheduler] Eval already dispatched for {run.run_id}, skipping")
                continue

            job = create_eval_job(
                run_id=run.run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                eval_entrypoint=self.config.eval_entrypoint,
                stats_server_uri=self.config.stats_server_uri,
                eval_overrides=self.config.eval_overrides,
            )
            jobs.append(job)

            # Update state: move from training to eval
            self.state.runs_in_training.discard(run.run_id)
            self.state.runs_in_eval.add(run.run_id)
            logger.info(f"[BatchedSyncedOptimizingScheduler] Scheduling evaluation for {run.run_id}")

        if jobs:
            return jobs

        # 2) Check if any training jobs are still in progress - if so, wait
        # This is the key constraint for batched scheduling
        if self.state.runs_in_training:
            logger.info(
                f"[BatchedSyncedOptimizingScheduler] Waiting for {len(self.state.runs_in_training)} "
                "training job(s) to complete before next batch"
            )
            return []

        # 3) Check if any eval jobs are still in progress - if so, wait
        if self.state.runs_in_eval:
            logger.info(
                f"[BatchedSyncedOptimizingScheduler] Waiting for {len(self.state.runs_in_eval)} "
                "eval job(s) to complete before next batch"
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
        if observations:
            logger.debug(f"[BatchedSyncedOptimizingScheduler] Observations being passed to optimizer: {observations}")
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
        finished = [r for r in runs if r.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STALE)]
        is_done = len(finished) >= self.config.max_trials
        if is_done:
            completed = [r for r in runs if r.status == JobStatus.COMPLETED]
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
            # Only collect observations from completed runs
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
