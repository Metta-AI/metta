"""
Optimizing Scheduler for Sweep Orchestration.

This scheduler integrates with an Optimizer (e.g., Protein) to get hyperparameter suggestions
and schedules jobs based on those suggestions.
"""

import logging
from dataclasses import dataclass
from typing import Any

from metta.sweep.sweep_orchestrator import (
    JobDefinition,
    JobStatus,
    JobTypes,
    Optimizer,
    RunInfo,
    SweepMetadata,
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


class OptimizingScheduler:
    """Scheduler that gets suggestions from optimizer."""

    def __init__(self, config: OptimizingSchedulerConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        self._created_runs = set()  # Track runs we've created to avoid duplicates
        self._is_complete = False  # Track if sweep is complete
        logger.info(f"[OptimizingScheduler] Initialized with max_trials={config.max_trials}")

    def schedule(self, sweep_metadata: SweepMetadata, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """Schedule next jobs based on current state."""

        # First, check for completed training runs that need evaluation
        runs_needing_eval = [run for run in all_runs if run.status == JobStatus.TRAINING_DONE_NO_EVAL]

        if runs_needing_eval:
            train_run = runs_needing_eval[0]

            # Merge eval_overrides with required push_metrics_to_wandb
            eval_overrides = self.config.eval_overrides.copy() if self.config.eval_overrides else {}
            eval_overrides["push_metrics_to_wandb"] = "True"  # Always push metrics to WandB for sweeps

            eval_job = JobDefinition(
                run_id=train_run.run_id,  # Use same run_id for eval
                cmd=f"{self.config.recipe_module}.{self.config.eval_entrypoint}",
                type=JobTypes.LAUNCH_EVAL,
                args=self.config.eval_args or [],
                overrides=eval_overrides,
                metadata={
                    "policy_uri": f"wandb://run/{train_run.run_id}",  # Pass policy URI as metadata
                },
            )
            # Extract just the trial portion for cleaner display
            display_id = train_run.run_id.split("_trial_")[-1] if "_trial_" in train_run.run_id else train_run.run_id
            display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id
            logger.info(f"[OptimizingScheduler] Scheduling evaluation for {display_id}")
            return [eval_job]

        # Check if we've hit the trial limit based on total runs created
        # Use both fetched runs and our internal tracking (in case fetch fails)
        total_runs = max(len(all_runs), len(self._created_runs))
        if total_runs >= self.config.max_trials:
            # Check if all runs are complete
            all_complete = all(run.status == JobStatus.COMPLETED for run in all_runs)
            if all_complete:
                self._is_complete = True
                logger.info(f"[OptimizingScheduler] All {self.config.max_trials} trials completed successfully!")
            else:
                logger.info(
                    f"[OptimizingScheduler] Reached max trials ({self.config.max_trials}), "
                    "waiting for remaining jobs to complete"
                )
            return []

        # For sequential scheduler, wait for ALL runs to complete before starting new ones
        incomplete_jobs = [run for run in all_runs if run.status != JobStatus.COMPLETED]

        if incomplete_jobs:
            # Build a status table for better visibility
            logger.info("[OptimizingScheduler] Run Status Table:")
            logger.info(f"[OptimizingScheduler] {'=' * 70}")
            logger.info(f"[OptimizingScheduler] {'Run ID':<30} {'Status':<25} {'Score':<15}")
            logger.info(f"[OptimizingScheduler] {'-' * 70}")

            for run in all_runs:
                score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
                # Extract just the trial portion of the run_id for cleaner display
                display_id = run.run_id.split("_trial_")[-1] if "_trial_" in run.run_id else run.run_id
                display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id
                logger.info(f"[OptimizingScheduler] {display_id:<30} {str(run.status):<25} {score_str:<15}")

            logger.info(f"[OptimizingScheduler] {'=' * 70}")
            logger.info(
                f"[OptimizingScheduler] Waiting for {len(incomplete_jobs)} incomplete job(s) "
                "to finish before scheduling next"
            )
            return []

        # Get observations for completed runs
        observations = []
        for run in all_runs:
            if run.observation:
                observations.append(run.observation)
                logger.debug(
                    f"[OptimizingScheduler] Found observation: "
                    f"score={run.observation.score:.3f}, cost={run.observation.cost:.1f}"
                )

        # Get suggestion from optimizer
        suggestions = self.optimizer.suggest(observations, n_suggestions=1)
        if not suggestions:
            logger.warning("[OptimizingScheduler] No suggestions from optimizer")
            return []

        suggestion = suggestions[0]

        # Create new training job with suggestion
        trial_num = len(self._created_runs) + 1
        run_id = f"{sweep_metadata.sweep_id}_trial_{trial_num:04d}"

        # Check if we've already created this run
        if run_id in self._created_runs:
            logger.warning(f"[OptimizingScheduler] Run {run_id} already created, skipping")
            return []

        self._created_runs.add(run_id)

        # Merge train_overrides with optimizer suggestions
        overrides = self.config.train_overrides.copy() if self.config.train_overrides else {}

        job = JobDefinition(
            run_id=run_id,
            cmd=f"{self.config.recipe_module}.{self.config.train_entrypoint}",
            type=JobTypes.LAUNCH_TRAINING,
            config=suggestion,  # Pass optimizer suggestion as config
            overrides=overrides,  # Pass any additional overrides
            metadata={
                "group": sweep_metadata.sweep_id,  # Pass group as metadata
            },
        )

        logger.info(
            f"[OptimizingScheduler] 🚀 Scheduling trial {trial_num}/{self.config.max_trials}: trial_{trial_num:04d}"
        )

        # Log some of the suggested hyperparameters for visibility
        for key in ["trainer.optimizer.learning_rate", "trainer.ppo.clip_coef", "trainer.ppo.ent_coef"]:
            if key in suggestion:
                logger.info(f"[OptimizingScheduler]    {key}: {suggestion[key]}")

        return [job]

    @property
    def is_complete(self) -> bool:
        """Check if sweep is complete."""
        return self._is_complete
