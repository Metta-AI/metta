"""
Sequential Scheduler for Sweep Orchestration.

This scheduler implements the simplest possible scheduling strategy:
- Always schedule exactly one job at a time
- No parallelism
- No early stopping
- Runs through all configurations sequentially
"""

import logging
from dataclasses import dataclass
from typing import Any

from metta.sweep.sweep_orchestrator import (
    JobDefinition,
    JobStatus,
    RunInfo,
    SweepMetadata,
    Scheduler,
)

logger = logging.getLogger(__name__)


@dataclass
class SequentialSchedulerConfig:
    """Configuration for the sequential scheduler."""
    
    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"  # e.g., "experiments.recipes.arena"
    train_entrypoint: str = "train_shaped"  # Function name for training
    eval_entrypoint: str = "evaluate"  # Function name for evaluation
    eval_args: list[str] | None = None  # Additional args for evaluation
    eval_overrides: dict[str, Any] | None = None  # Additional overrides for evaluation
    

class SequentialScheduler:
    """
    Sequential scheduler that always schedules exactly one job at a time.
    
    This is the simplest possible scheduler:
    - Maintains internal count of trials
    - Always returns one job if under trial limit
    - No early stopping
    - Schedules evaluations for all completed training jobs
    """
    
    def __init__(self, config: SequentialSchedulerConfig):
        self.config = config
        self._trial_count = 0
        self._initialized = False
        
    def initialize(self, sweep_id: str) -> list[JobDefinition]:
        """
        Generate initial job for warmup phase.
        
        For sequential scheduling, we start with exactly one job.
        """
        if self._initialized:
            logger.warning(f"Scheduler already initialized for sweep {sweep_id}")
            return []
            
        self._initialized = True
        self._trial_count = 1
        
        # Create first job
        job = JobDefinition(
            run_id=f"{sweep_id}_trial_{self._trial_count:04d}",
            cmd=f"{self.config.recipe_module}.{self.config.train_entrypoint}",
            type="train",
        )
        
        logger.info(f"Initialized sequential scheduler with first job: {job.run_id}")
        return [job]
    
    def schedule(self, sweep_metadata: SweepMetadata, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """
        Decide which jobs to schedule based on current state of all runs.
        Handles both training and evaluation jobs.
        
        Sequential logic:
        - Only schedule if no jobs are currently running
        - Schedule evaluations for completed training jobs
        - Schedule next training job if under trial limit
        - Always return at most one job at a time
        """
        jobs = []
        
        # First, check if any training runs need evaluation
        runs_needing_eval = [
            run for run in all_runs 
            if run.status == JobStatus.IN_TRAINING 
            and run.has_completed_training 
            and not run.has_been_evaluated
        ]
        
        # Schedule evaluation for the first completed training job
        if runs_needing_eval:
            train_run = runs_needing_eval[0]
            eval_job = JobDefinition(
                run_id=f"{train_run.run_id}_eval",
                cmd=f"{self.config.recipe_module}.{self.config.eval_entrypoint}",
                type="eval",
                parent_job_id=train_run.run_id,
                args=self.config.eval_args or [],
                overrides=self.config.eval_overrides or {},
            )
            
            # Add policy URI pointing to parent training job
            eval_job.overrides["policy_uri"] = f"wandb://run/{train_run.run_id}"
            
            logger.info(f"Scheduling evaluation for {train_run.run_id}: {eval_job.run_id}")
            return [eval_job]
        
        # Check if we've hit the trial limit
        if self._trial_count >= self.config.max_trials:
            logger.info(f"Reached max trials ({self.config.max_trials}), not scheduling more")
            return []
        
        # Count running jobs - only schedule new training if nothing is running
        running_jobs = [
            run for run in all_runs 
            if run.status in [JobStatus.PENDING, JobStatus.IN_TRAINING, JobStatus.IN_EVAL]
        ]
        
        if running_jobs:
            logger.debug(f"{len(running_jobs)} jobs still running, waiting...")
            return []
        
        # Schedule next training job
        self._trial_count += 1
        job = JobDefinition(
            run_id=f"{sweep_metadata.sweep_id}_trial_{self._trial_count:04d}",
            cmd=f"{self.config.recipe_module}.{self.config.train_entrypoint}",
            type="train",
        )
        
        logger.info(f"Scheduling job {self._trial_count}/{self.config.max_trials}: {job.run_id}")
        return [job]
    
    def schedule_evaluations(self, jobs_needing_eval: list[JobDefinition]) -> list[JobDefinition]:
        """
        DEPRECATED: Evaluation scheduling is now handled in the main schedule() method.
        This method is kept for backward compatibility but should not be called.
        """
        logger.warning("schedule_evaluations() is deprecated. Evaluation scheduling is handled in schedule()")
        return []
    
    def should_stop_job(self, job: JobDefinition, current_metrics: dict[str, float]) -> bool:
        """
        Decide if a running job should be stopped early.
        
        Sequential scheduler never stops jobs early.
        """
        return False


def create_sequential_scheduler(
    max_trials: int = 10,
    recipe_module: str = "experiments.recipes.arena",
    train_entrypoint: str = "train_shaped",
    eval_entrypoint: str = "evaluate",
    **kwargs
) -> SequentialScheduler:
    """
    Factory function to create a sequential scheduler.
    
    Args:
        max_trials: Maximum number of trials to run
        recipe_module: Python module path to recipe (e.g., "experiments.recipes.arena")
        train_entrypoint: Function name in recipe module for training
        eval_entrypoint: Function name in recipe module for evaluation
        **kwargs: Additional config options
    
    Returns:
        Configured SequentialScheduler instance
    """
    config = SequentialSchedulerConfig(
        max_trials=max_trials,
        recipe_module=recipe_module,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        eval_args=kwargs.get("eval_args"),
        eval_overrides=kwargs.get("eval_overrides"),
    )
    
    return SequentialScheduler(config)