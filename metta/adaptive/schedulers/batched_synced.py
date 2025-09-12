"""Batched synchronized scheduler for adaptive experiments.

This scheduler waits for all runs to complete (including evaluation) before
generating a new batch of suggestions. This ensures perfect synchronization
between batches and is ideal for comparing hyperparameters fairly.
"""

from __future__ import annotations

import logging
from typing import Any, List

from pydantic import Field

from metta.adaptive.models import JobDefinition, JobStatus, Observation, RunInfo
from metta.adaptive.protocols import Optimizer
from metta.adaptive.utils import create_eval_job, create_training_job, generate_run_id
from metta.mettagrid.config import Config

logger = logging.getLogger(__name__)


class BatchedSyncedSchedulerConfig(Config):
    """Configuration for batched synchronized scheduler."""

    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    train_overrides: dict[str, Any] = Field(default_factory=dict)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)
    stats_server_uri: str | None = None
    gpus: int = 1
    nodes: int = 1
    batch_size: int = 4
    experiment_id: str = "batched_synced"


class BatchedSyncedOptimizingScheduler:
    """Scheduler that generates batches of suggestions synchronously.

    Key behaviors:
    - Only generates new suggestions when ALL current runs (including evals) are complete
    - Schedules evals for any runs with training complete and eval not yet started
    - Generates up to `batch_size` training jobs at a time (bounded by available slots)
    - Suggestions come from a stateless Optimizer; observations are read from run summaries
    """

    def __init__(self, config: BatchedSyncedSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        logger.info(
            "[BatchedSyncedOptimizingScheduler] Initialized with max_trials=%s, batch_size=%s",
            config.max_trials,
            config.batch_size,
        )

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """Schedule next jobs based on current state and available resources."""
        jobs: list[JobDefinition] = []

        # 1) Schedule evals for any runs with training done but no eval yet
        eval_candidates = [r for r in runs if r.status == JobStatus.TRAINING_DONE_NO_EVAL]
        for run in eval_candidates:
            job = create_eval_job(
                run_id=run.run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                eval_entrypoint=self.config.eval_entrypoint,
                stats_server_uri=self.config.stats_server_uri,
                eval_overrides=self.config.eval_overrides,
            )
            jobs.append(job)
            logger.info("[BatchedSyncedOptimizingScheduler] Scheduling evaluation for %s", run.run_id)

        if jobs:
            return jobs

        # 2) Check completion barrier and max trials
        total_created = len(runs)
        if total_created >= self.config.max_trials:
            # Allow remaining jobs (if any) to finish naturally; no new training
            logger.info(
                "[BatchedSyncedOptimizingScheduler] Max trials reached (%s). Waiting for all to complete",
                self.config.max_trials,
            )
            return []

        # Barrier: wait until all runs are COMPLETED or FAILED
        incomplete = [r for r in runs if r.status not in (JobStatus.COMPLETED, JobStatus.FAILED)]
        if incomplete:
            logger.info(
                "[BatchedSyncedOptimizingScheduler] Waiting for %s run(s) to complete before next batch",
                len(incomplete),
            )
            return []

        # 3) Generate a new batch of training suggestions
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
            # Record suggestion for downstream hooks; controller can write it into summary
            job.metadata["adaptive/suggestion"] = suggestion
            jobs.append(job)
            logger.info("[BatchedSyncedOptimizingScheduler] Scheduling training %s", run_id)

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        completed = [r for r in runs if r.status == JobStatus.COMPLETED]
        is_done = len(completed) >= self.config.max_trials
        if is_done:
            logger.info(
                "[BatchedSyncedOptimizingScheduler] Experiment complete! %s/%s trials finished",
                len(completed),
                self.config.max_trials,
            )
        return is_done

    def _collect_observations(self, runs: list[RunInfo]) -> List[Observation]:
        """Extract observations from run summaries written by hooks or training."""
        obs_list: list[Observation] = []
        for run in runs:
            summary = run.summary if isinstance(run.summary, dict) else {}
            if not summary:
                continue
            score = summary.get("observation/score")
            cost = summary.get("observation/cost")
            suggestion = summary.get("observation/suggestion", {})
            if score is not None:
                try:
                    obs_list.append(
                        Observation(
                            score=float(score),
                            cost=float(cost) if cost is not None else 0.0,
                            suggestion=dict(suggestion) if isinstance(suggestion, dict) else {},
                        )
                    )
                except Exception:
                    # Skip malformed entries
                    continue
        return obs_list
