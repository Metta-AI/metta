"""Simple train-and-eval scheduler for PoC."""

import logging
from typing import Any

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.protocols import ExperimentState
from metta.adaptive.utils import create_eval_job, create_training_job, generate_run_id
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class TrainAndEvalConfig(Config):
    """Configuration for simple train-and-eval scheduler."""

    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    max_trials: int = 3
    gpus: int = 1
    experiment_id: str = "train_eval_poc"
    train_overrides: dict[str, Any] = {}
    stats_server_uri: str | None = None


class TrainAndEvalScheduler:
    """
    Simple scheduler for PoC: just train jobs followed by eval jobs.

    Behavior:
    1. Create one training job at a time
    2. When training completes, create eval job for it
    3. Repeat until max_trials reached
    """

    def __init__(self, config: TrainAndEvalConfig, state: ExperimentState | None = None):
        self.config = config
        self.state = state
        logger.info(f"[TrainAndEvalScheduler] Initialized with max_trials={config.max_trials}")

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """Schedule train or eval jobs based on current state."""

        # 1. First priority: Create eval jobs for completed training
        jobs = []
        for run in runs:
            if run.status == JobStatus.TRAINING_DONE_NO_EVAL:
                eval_job = create_eval_job(
                    run_id=run.run_id,  # Same run_id as the training job
                    experiment_id=self.config.experiment_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                    stats_server_uri=self.config.stats_server_uri,
                )
                jobs.append(eval_job)
                logger.info(f"[TrainAndEvalScheduler] Creating eval job for {run.run_id}")

        # 2. Create new training jobs if we have capacity and haven't hit max trials
        # Use the number of existing runs to remain idempotent across restarts
        current_trials = len(runs)
        while available_training_slots > 0 and current_trials < self.config.max_trials:
            trial_num = current_trials + 1
            run_id = generate_run_id(self.config.experiment_id, trial_num)

            training_job = create_training_job(
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                train_entrypoint=self.config.train_entrypoint,
                train_overrides=self.config.train_overrides,
                gpus=self.config.gpus,
                stats_server_uri=self.config.stats_server_uri,
            )
            jobs.append(training_job)
            available_training_slots -= 1
            current_trials += 1
            logger.info(
                f"[TrainAndEvalScheduler] Creating training job {run_id} ({current_trials}/{self.config.max_trials})"
            )

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        """Check if train-and-eval experiment is complete."""
        # Experiment is complete when:
        # 1. We've reached max_trials
        # 2. All runs have been fully completed (trained AND evaluated)
        # Check that all runs are completed (not just training done)
        completed_runs = [run for run in runs if run.status == JobStatus.COMPLETED]

        # We're done when we have max_trials completed runs
        is_complete = len(completed_runs) >= self.config.max_trials

        if is_complete:
            logger.info(
                "[TrainAndEvalScheduler] Experiment complete! %s/%s trials finished",
                len(completed_runs),
                self.config.max_trials,
            )

        return is_complete
