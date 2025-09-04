from platform import node
"""Optimizing Scheduler for Sweep Orchestration.

This scheduler integrates with an Optimizer (e.g., Protein) to get hyperparameter suggestions
and schedules jobs based on those suggestions."""

import logging
from dataclasses import dataclass
from typing import Any

from metta.sweep.models import JobDefinition, JobStatus, RunInfo, SweepMetadata
from metta.sweep.protocols import Optimizer
from metta.sweep.utils import (
    create_eval_job,
    create_training_job,
    generate_run_id,
    get_display_id,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizingSchedulerConfig:
    """Configuration for optimizing scheduler."""

    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"  # e.g., "experiments.recipes.arena"
    train_entrypoint: str = "train_shaped"  # Function name for training
    eval_entrypoint: str = "evaluate"  # Function name for evaluation
    train_overrides: dict[str, Any] | None = None  # Additional overrides for training jobs
    eval_args: list[str] | None = None  # Additional args for evaluation
    eval_overrides: dict[str, Any] | None = None  # Additional overrides for evaluation
    stats_server_uri: str | None = None  # Stats server for remote evaluations
    gpus: int = 1  # Number of GPUs per training job
    nodes: int = 1  # Number of nodes per training job


class OptimizingScheduler:
    """Scheduler that gets suggestions from optimizer."""

    def __init__(self, config: OptimizingSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        logger.info(f"[OptimizingScheduler] Initialized with max_trials={config.max_trials}")

    def schedule(
        self,
        sweep_metadata: SweepMetadata,
        all_runs: list[RunInfo],
        dispatched_trainings: set[str],
        dispatched_evals: set[str],
    ) -> list[JobDefinition]:
        """Schedule next jobs based on current state."""

        # Check for completed training runs that need evaluation
        runs_needing_eval = [run for run in all_runs if run.status == JobStatus.TRAINING_DONE_NO_EVAL]

        if runs_needing_eval:
            return self._schedule_evaluation(runs_needing_eval[0], sweep_metadata, all_runs, dispatched_evals)

        # Check if we've hit the trial limit
        total_runs = max(len(all_runs), len(dispatched_trainings))
        if total_runs >= self.config.max_trials:
            return self._handle_max_trials_reached(all_runs)

        # Wait for incomplete jobs to finish before scheduling new ones
        incomplete_jobs = [run for run in all_runs if run.status not in (JobStatus.COMPLETED, JobStatus.FAILED)]

        if incomplete_jobs:
            logger.info(
                f"[OptimizingScheduler] Waiting for {len(incomplete_jobs)} incomplete job(s) "
                "to finish before scheduling next"
            )
            return []

        # Schedule new training job with optimizer suggestion
        return self._schedule_training(sweep_metadata, all_runs, dispatched_trainings)

    def _schedule_evaluation(
        self, train_run: RunInfo, sweep_metadata: SweepMetadata, all_runs: list[RunInfo], dispatched_evals: set[str]
    ) -> list[JobDefinition]:
        """Schedule evaluation job for a completed training run."""
        # Check if evaluation already dispatched
        if train_run.run_id in dispatched_evals:
            logger.debug(f"[OptimizingScheduler] Evaluation already dispatched for {train_run.run_id}, skipping")
            return []

        # Create evaluation job
        eval_job = create_eval_job(
            run_id=train_run.run_id,
            sweep_id=sweep_metadata.sweep_id,
            recipe_module=self.config.recipe_module,
            eval_entrypoint=self.config.eval_entrypoint,
            stats_server_uri=self.config.stats_server_uri,
            eval_args=self.config.eval_args,
            eval_overrides=self.config.eval_overrides,
        )

        # Log scheduling with clean display ID
        display_id = get_display_id(train_run.run_id)
        logger.info(f"[OptimizingScheduler] Scheduling evaluation for run {display_id}")

        return [eval_job]

    def _schedule_training(
        self, sweep_metadata: SweepMetadata, all_runs: list[RunInfo], dispatched_trainings: set[str]
    ) -> list[JobDefinition]:
        """Schedule new training job with optimizer suggestion."""
        # Collect observations from completed runs
        observations = [run.observation for run in all_runs if run.observation]

        # Get suggestion from optimizer
        suggestions = self.optimizer.suggest(observations, n_suggestions=1)
        if not suggestions:
            logger.warning("[OptimizingScheduler] No suggestions from optimizer")
            return []

        # Create new training job
        trial_num = len(dispatched_trainings) + 1
        run_id = generate_run_id(sweep_metadata.sweep_id, trial_num)

        # Avoid duplicates
        if run_id in dispatched_trainings:
            logger.warning(f"[OptimizingScheduler] Run {run_id} already created, skipping")
            return []

        # Create training job
        job = create_training_job(
            run_id=run_id,
            sweep_id=sweep_metadata.sweep_id,
            recipe_module=self.config.recipe_module,
            train_entrypoint=self.config.train_entrypoint,
            config=suggestions[0],
            gpus=self.config.gpus,
            nodes=self.config.nodes,
            stats_server_uri=self.config.stats_server_uri,
            train_overrides=self.config.train_overrides,
        )

        logger.info(
            f"[OptimizingScheduler] 🚀 Scheduling trial {trial_num}/{self.config.max_trials}: trial_{trial_num:04d}"
        )
        return [job]

    def _handle_max_trials_reached(self, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """Handle case when maximum trials have been reached."""
        # Check if all runs are complete
        all_complete = all(run.status in (JobStatus.COMPLETED, JobStatus.FAILED) for run in all_runs)

        if all_complete:
            logger.info(f"[OptimizingScheduler] All {self.config.max_trials} trials finished!")
        else:
            incomplete_count = sum(1 for run in all_runs if run.status not in (JobStatus.COMPLETED, JobStatus.FAILED))
            logger.info(f"[OptimizingScheduler] Waiting for {incomplete_count} remaining job(s) to complete")

        return []
