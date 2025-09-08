"""Optimizing Scheduler for Sweep Orchestration.

This scheduler integrates with an Optimizer (e.g., Protein) to get hyperparameter suggestions
and schedules jobs based on those suggestions."""

import logging
from dataclasses import dataclass
from typing import Any

from metta.sweep.models import JobDefinition, JobStatus, JobTypes, RunInfo, SweepMetadata
from metta.sweep.protocols import Optimizer

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
        self._is_complete = False  # Track if sweep is complete
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

        # Build evaluation overrides
        eval_overrides = self._build_eval_overrides(train_run.run_id, sweep_metadata.sweep_id)

        # Create evaluation job
        eval_job = JobDefinition(
            run_id=train_run.run_id,
            cmd=f"{self.config.recipe_module}.{self.config.eval_entrypoint}",
            type=JobTypes.LAUNCH_EVAL,
            args=self.config.eval_args or [],
            overrides=eval_overrides,
            metadata={"policy_uri": f"wandb://metta/{train_run.run_id}"},
        )

        # Log scheduling with clean display ID
        display_id = self._get_display_id(train_run.run_id)
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
        run_id = f"{sweep_metadata.sweep_id}_trial_{trial_num:04d}"

        # Avoid duplicates
        if run_id in dispatched_trainings:
            logger.warning(f"[OptimizingScheduler] Run {run_id} already created, skipping")
            return []

        # Build training job
        overrides = self._build_train_overrides()

        job = JobDefinition(
            run_id=run_id,
            cmd=f"{self.config.recipe_module}.{self.config.train_entrypoint}",
            type=JobTypes.LAUNCH_TRAINING,
            gpus=self.config.gpus,
            nodes=self.config.nodes,
            config=suggestions[0],
            overrides=overrides,
            metadata={"group": sweep_metadata.sweep_id},
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
            self._is_complete = True
            logger.info(f"[OptimizingScheduler] All {self.config.max_trials} trials finished!")
        else:
            incomplete_count = sum(1 for run in all_runs if run.status not in (JobStatus.COMPLETED, JobStatus.FAILED))
            logger.info(f"[OptimizingScheduler] Waiting for {incomplete_count} remaining job(s) to complete")

        return []

    def _build_eval_overrides(self, run_id: str, sweep_id: str) -> dict[str, Any]:
        """Build evaluation override parameters."""
        eval_overrides = self.config.eval_overrides.copy() if self.config.eval_overrides else {}
        eval_overrides["push_metrics_to_wandb"] = "True"
        eval_overrides["wandb.name"] = run_id
        eval_overrides["wandb.run_id"] = run_id
        eval_overrides["wandb.group"] = sweep_id

        if self.config.stats_server_uri:
            eval_overrides["stats_server_uri"] = self.config.stats_server_uri

        return eval_overrides

    def _build_train_overrides(self) -> dict[str, Any]:
        """Build training override parameters."""
        overrides = self.config.train_overrides.copy() if self.config.train_overrides else {}

        if self.config.stats_server_uri:
            overrides["stats_server_uri"] = self.config.stats_server_uri
            overrides["trainer.evaluation.evaluate_remote"] = "True"
            overrides["trainer.evaluation.evaluate_local"] = "False"
            overrides["trainer.evaluation.skip_git_check"] = "True"

        return overrides

    def _get_display_id(self, run_id: str) -> str:
        """Extract clean display ID from run ID."""
        if "_trial_" in run_id:
            display_id = run_id.split("_trial_")[-1]
            return f"trial_{display_id}" if not display_id.startswith("trial_") else display_id
        return run_id

    @property
    def is_complete(self) -> bool:
        """Check if sweep is complete."""
        return self._is_complete
