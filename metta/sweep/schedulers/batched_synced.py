"""Batched synchronized scheduler for sweep orchestration.

This scheduler waits for all runs to complete (including evaluation) before
generating a new batch of suggestions. This ensures perfect synchronization
between batches and is ideal for comparing hyperparameters fairly.
"""

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
class BatchedSyncedSchedulerConfig:
    """Configuration for batched synchronized scheduler."""

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
    batch_size: int = 4


class BatchedSyncedOptimizingScheduler:
    """Scheduler that generates batches of suggestions synchronously.

    This scheduler waits for ALL runs (including evaluations) to complete
    before generating the next batch of suggestions. The batch size is
    determined by max_parallel_jobs from the controller.

    Key behaviors:
    - Only generates new suggestions when ALL runs are COMPLETED
    - Generates exactly max_parallel_jobs suggestions at once
    - Ensures fair comparison within each batch
    - Prevents any overlap between batches
    """

    def __init__(self, config: BatchedSyncedSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        self._total_scheduled = 0  # Track total number of trials scheduled
        logger.info(f"[BatchedSyncedScheduler] Initialized with max_trials={config.max_trials}")

    def schedule(
        self,
        sweep_metadata: SweepMetadata,
        all_runs: list[RunInfo],
        dispatched_trainings: set[str],
        dispatched_evals: set[str],
    ) -> list[JobDefinition]:
        """Schedule next batch of jobs when all current runs are complete.

        Args:
            sweep_metadata: Current sweep metadata
            all_runs: All runs in the sweep
            dispatched_trainings: Set of already dispatched training job IDs
            dispatched_evals: Set of already dispatched eval job IDs

        Returns:
            List of jobs to schedule (either empty or a full batch)
        """

        # First, check for any training runs that need evaluation
        runs_needing_eval = [
            run
            for run in all_runs
            if run.status == JobStatus.TRAINING_DONE_NO_EVAL and run.run_id not in dispatched_evals
        ]

        if runs_needing_eval:
            # Schedule evaluations for all completed training runs
            eval_jobs = []
            for run in runs_needing_eval:
                eval_job = create_eval_job(
                    run_id=run.run_id,
                    sweep_id=sweep_metadata.sweep_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                    stats_server_uri=self.config.stats_server_uri,
                    eval_args=self.config.eval_args,
                    eval_overrides=self.config.eval_overrides,
                )
                eval_jobs.append(eval_job)

                display_id = get_display_id(run.run_id)
                logger.info(f"[BatchedSyncedScheduler] Scheduling evaluation for {display_id}")

            return eval_jobs

        # Check if we've hit the trial limit
        total_runs = len(dispatched_trainings)
        if total_runs >= self.config.max_trials:
            return self._handle_max_trials_reached(all_runs)

        # Check if ALL runs are completed before generating next batch
        incomplete_runs = [run for run in all_runs if run.status not in (JobStatus.COMPLETED, JobStatus.FAILED)]

        if incomplete_runs:
            # Still have incomplete runs, wait for them
            logger.info(
                f"[BatchedSyncedScheduler] Waiting for {len(incomplete_runs)} run(s) "
                f"to complete before generating next batch"
            )
            return []

        # All runs are complete - generate a new batch
        return self._schedule_training_batch(sweep_metadata, all_runs, dispatched_trainings)

    def _schedule_training_batch(
        self,
        sweep_metadata: SweepMetadata,
        all_runs: list[RunInfo],
        dispatched_trainings: set[str],
    ) -> list[JobDefinition]:
        """Schedule a batch of training jobs with optimizer suggestions.

        This generates multiple suggestions at once for parallel evaluation.
        """
        # Determine batch size (controller will pass this via metadata or we infer)
        # For now, we'll calculate based on how many we can still schedule
        remaining_trials = self.config.max_trials - len(dispatched_trainings)

        if remaining_trials <= 0:
            return []

        # Use the configured batch size, but don't exceed remaining trials
        batch_size = min(remaining_trials, self.config.batch_size)

        # Collect observations from completed runs
        observations = [run.observation for run in all_runs if run.observation]

        # Get batch of suggestions from optimizer
        logger.info(f"[BatchedSyncedScheduler] Requesting {batch_size} suggestions from optimizer")
        suggestions = self.optimizer.suggest(observations, n_suggestions=batch_size)

        if not suggestions:
            logger.warning("[BatchedSyncedScheduler] No suggestions from optimizer")
            return []

        # Create training jobs for all suggestions
        jobs = []
        base_trial_num = len(dispatched_trainings)

        for i, suggestion in enumerate(suggestions):
            trial_num = base_trial_num + i + 1
            run_id = generate_run_id(sweep_metadata.sweep_id, trial_num)

            # Check for duplicates (shouldn't happen but safety check)
            if run_id in dispatched_trainings:
                logger.warning(f"[BatchedSyncedScheduler] Run {run_id} already exists, skipping")
                continue

            # Create training job
            job = create_training_job(
                run_id=run_id,
                sweep_id=sweep_metadata.sweep_id,
                recipe_module=self.config.recipe_module,
                train_entrypoint=self.config.train_entrypoint,
                config=suggestion,
                gpus=self.config.gpus,
                stats_server_uri=self.config.stats_server_uri,
                train_overrides=self.config.train_overrides,
            )
            jobs.append(job)

        if jobs:
            logger.info(
                f"[BatchedSyncedScheduler] 🚀 Scheduling batch of {len(jobs)} trials "
                f"({base_trial_num + 1}-{base_trial_num + len(jobs)}/{self.config.max_trials})"
            )

            # Log individual trials
            for job in jobs:
                display_id = get_display_id(job.run_id)
                logger.info(f"  - {display_id}")

        return jobs

    def _handle_max_trials_reached(self, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """Handle case when maximum trials have been reached."""
        # Check if all runs are complete
        all_complete = all(run.status in (JobStatus.COMPLETED, JobStatus.FAILED) for run in all_runs)

        if all_complete:
            logger.info(f"[BatchedSyncedScheduler] ✅ All {self.config.max_trials} trials finished!")
        else:
            incomplete_count = sum(1 for run in all_runs if run.status not in (JobStatus.COMPLETED, JobStatus.FAILED))
            logger.info(f"[BatchedSyncedScheduler] Waiting for {incomplete_count} remaining job(s) to complete")

        return []
