"""Simple train-and-eval scheduler for PoC."""

import logging
from dataclasses import dataclass

from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.utils import create_eval_job, create_training_job, generate_run_id

logger = logging.getLogger(__name__)


@dataclass
class TrainAndEvalConfig:
    """Configuration for simple train-and-eval scheduler."""
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"
    max_trials: int = 3
    gpus_per_job: int = 1
    experiment_id: str = "train_eval_poc"


class TrainAndEvalScheduler:
    """
    Simple scheduler for PoC: just train jobs followed by eval jobs.

    Behavior:
    1. Create one training job at a time
    2. When training completes, create eval job for it
    3. Repeat until max_trials reached
    """

    def __init__(self, config: TrainAndEvalConfig):
        self.config = config
        self._trial_count = 0
        logger.info(f"[TrainAndEvalScheduler] Initialized with max_trials={config.max_trials}")

    def schedule(self, runs: list[RunInfo], available_training_slots: int) -> list[JobDefinition]:
        """Schedule train or eval jobs based on current state."""

        # 1. First priority: Create eval jobs for completed training
        eval_jobs = []
        for run in runs:
            if run.status == JobStatus.TRAINING_DONE_NO_EVAL:
                eval_job = create_eval_job(
                    run_id=run.run_id,  # Same run_id as the training job
                    experiment_id=self.config.experiment_id,
                    recipe_module=self.config.recipe_module,
                    eval_entrypoint=self.config.eval_entrypoint,
                )
                eval_jobs.append(eval_job)
                logger.info(f"[TrainAndEvalScheduler] Creating eval job for {run.run_id}")

        if eval_jobs:
            return eval_jobs

        # 2. Create new training job if we have capacity and haven't hit max trials
        if available_training_slots > 0 and self._trial_count < self.config.max_trials:
            self._trial_count += 1
            run_id = generate_run_id(self.config.experiment_id, self._trial_count)

            training_job = create_training_job(
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                recipe_module=self.config.recipe_module,
                train_entrypoint=self.config.train_entrypoint,
                config={},  # No hyperparameter optimization for PoC
                gpus=self.config.gpus_per_job,
            )

            logger.info(f"[TrainAndEvalScheduler] Creating training job {run_id} ({self._trial_count}/{self.config.max_trials})")
            return [training_job]

        return []
