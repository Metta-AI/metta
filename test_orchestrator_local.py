#!/usr/bin/env python3
"""Test script for the new sweep orchestrator with local dispatch."""

import logging
import time
from dataclasses import dataclass
from typing import Any

# Remove unused import - RunStatus not needed

from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig
from metta.sweep.scheduler.sequential import SequentialScheduler, SequentialSchedulerConfig
from metta.sweep.store.wandb import WandbStore
from metta.sweep.sweep_orchestrator import (
    JobDefinition,
    JobStatus,
    JobTypes,
    LocalDispatcher,
    Observation,
    RunInfo,
    SweepController,
    SweepMetadata,
)
from metta.common.wandb.wandb_context import WandbConfig

# Configure logging to be concise
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Suppress verbose loggers
logging.getLogger("metta.sweep.store.wandb").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


@dataclass
class TestConfig:
    """Configuration for the test."""

    sweep_name: str = None  # Will be generated with timestamp
    entity: str = "metta-research"
    project: str = "metta"  # Use existing project
    num_trials: int = 5
    max_parallel_jobs: int = 1
    monitoring_interval: int = 5

    def __post_init__(self):
        if self.sweep_name is None:
            # Generate unique sweep name with timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.sweep_name = f"test_sweep_{timestamp}"


class OptimizingScheduler:
    """
    Enhanced scheduler that integrates with the Optimizer to get suggestions.
    """

    def __init__(self, config: SequentialSchedulerConfig, optimizer: ProteinOptimizer):
        self.config = config
        self.optimizer = optimizer
        self._created_runs = set()  # Track runs we've created to avoid duplicates
        logger.info(f"Initialized OptimizingScheduler with max_trials={config.max_trials}")

    def schedule(self, sweep_metadata: SweepMetadata, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """Schedule jobs with optimizer suggestions."""

        # First, check for completed training runs that need evaluation
        runs_needing_eval = [run for run in all_runs if run.status == JobStatus.TRAINING_DONE_NO_EVAL]

        if runs_needing_eval:
            train_run = runs_needing_eval[0]
            eval_job = JobDefinition(
                run_id=train_run.run_id,  # Use same run_id for eval
                cmd=f"{self.config.recipe_module}.{self.config.eval_entrypoint}",
                type=JobTypes.LAUNCH_EVAL,
                args=[],  # No positional args
                overrides={
                    "push_metrics_to_wandb": "True",  # This is an override
                },
                metadata={
                    "policy_uri": f"wandb://run/{train_run.run_id}",  # This is an arg
                },
            )
            logger.info(f"📊 Scheduling evaluation for {train_run.run_id}")
            return [eval_job]

        # Check if we've hit the trial limit based on total runs created
        # Use both fetched runs and our internal tracking (in case fetch fails)
        total_runs = max(len(all_runs), len(self._created_runs))
        if total_runs >= self.config.max_trials:
            logger.info(f"✅ Reached max trials ({self.config.max_trials})")
            return []

        # For sequential scheduler, wait for ALL runs to complete before starting new ones
        incomplete_jobs = [run for run in all_runs if run.status != JobStatus.COMPLETED]

        if incomplete_jobs:
            logger.info(f"Waiting for {len(incomplete_jobs)} incomplete jobs to finish (including PENDING)")
            for job in incomplete_jobs[:3]:  # Show first 3 for debugging
                logger.debug(f"  - {job.run_id}: status={job.status}")
            return []

        # Get observations for completed runs
        observations = []
        for run in all_runs:
            if run.observation:
                observations.append(run.observation)
                logger.debug(f"Found observation: score={run.observation.score:.3f}, cost={run.observation.cost:.1f}")

        # Get suggestion from optimizer
        suggestions = self.optimizer.suggest(observations, n_suggestions=1)
        if not suggestions:
            logger.warning("No suggestions from optimizer")
            return []

        suggestion = suggestions[0]

        # Create new training job with suggestion
        trial_num = len(self._created_runs) + 1
        run_id = f"{sweep_metadata.sweep_id}_trial_{trial_num:04d}"

        # Check if we've already created this run
        if run_id in self._created_runs:
            logger.warning(f"Run {run_id} already created, skipping")
            return []

        self._created_runs.add(run_id)

        job = JobDefinition(
            run_id=run_id,
            cmd=f"{self.config.recipe_module}.{self.config.train_entrypoint}",
            type=JobTypes.LAUNCH_TRAINING,
            config=suggestion,  # Pass optimizer suggestion as config
            overrides={
                "trainer.total_timesteps": "10000",  # Override for quick testing
            },
            metadata={
                "group": sweep_metadata.sweep_id,  # Pass group as an arg
            },
        )

        logger.info(f"🚀 Scheduling trial {trial_num}/{self.config.max_trials}: {job.run_id}")
        logger.info(f"   Suggestion: lr={suggestion.get('trainer.optimizer.learning_rate', 'default')}")
        return [job]


def run_test():
    """Run the sweep orchestrator test."""

    config = TestConfig()

    logger.info("=" * 60)
    logger.info(f"Starting sweep test: {config.sweep_name}")
    logger.info(f"Config: {config.num_trials} trials, max {config.max_parallel_jobs} parallel")
    logger.info("=" * 60)

    # 1. Create minimal sweep config
    protein_config = ProteinConfig(
        metric="evaluator/arena_shaped/score",
        goal="maximize",
        method="random",  # Use random for simplicity in testing
        parameters={
            "trainer.optimizer.learning_rate": ParameterConfig(
                min=1e-5, max=1e-3, distribution="log_normal", mean=1e-4, scale="auto"
            ),
        },
    )

    # 2. Create components
    # For now, use WandB store but with better error handling
    store = WandbStore(entity=config.entity, project=config.project)
    dispatcher = LocalDispatcher()
    optimizer = ProteinOptimizer(protein_config)

    logger.info(f"Using sweep name: {config.sweep_name}")

    # Create scheduler with optimizer integration
    scheduler_config = SequentialSchedulerConfig(
        max_trials=config.num_trials,
        recipe_module="experiments.recipes.arena",
        train_entrypoint="train_shaped",
        eval_entrypoint="evaluate",
    )
    scheduler = OptimizingScheduler(scheduler_config, optimizer)

    # 3. Create and run controller
    controller = SweepController(
        sweep_id=config.sweep_name,
        scheduler=scheduler,
        optimizer=optimizer,
        dispatcher=dispatcher,
        store=store,
        protein_config=protein_config,
        max_parallel_jobs=config.max_parallel_jobs,
        monitoring_interval=config.monitoring_interval,
    )

    # Track metrics
    start_time = time.time()

    logger.info("\n🏃 Starting control loop...")

    # Run controller in a separate thread with timeout
    import threading
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.daemon = True  # Allow main thread to exit
    controller_thread.start()

    try:
        # Monitor progress while controller runs
        max_wait_time = 600  # 10 minutes max
        check_interval = 5  # Check every 5 seconds
        elapsed = 0
        
        while elapsed < max_wait_time:
            time.sleep(check_interval)
            elapsed += check_interval
            
            # Fetch current status
            all_runs = store.fetch_runs(filters={"group": config.sweep_name})
            
            # Log progress
            completed_count = sum(1 for run in all_runs if run.status == JobStatus.COMPLETED)
            logger.info(f"Progress: {completed_count}/{config.num_trials} trials completed (elapsed: {elapsed}s)")
            
            # Check if all trials are done
            if completed_count >= config.num_trials:
                logger.info(f"\n✨ All {config.num_trials} trials completed!")
                break
                
            # Show active processes
            active_processes = dispatcher.check_processes()
            if active_processes > 0:
                logger.info(f"  Active processes: {active_processes}")
                
        else:
            logger.warning(f"Timeout reached after {max_wait_time} seconds")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")

    # Final summary
    elapsed = time.time() - start_time
    final_runs = store.fetch_runs(filters={"group": config.sweep_name})

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"Total runs: {len(final_runs)}")

    # Verify all expected phases
    phase_checks = {
        "Started training": lambda r: r.has_started_training,
        "Completed training": lambda r: r.has_completed_training,
        "Started eval": lambda r: r.has_started_eval,
        "Completed eval": lambda r: r.has_been_evaluated,
        "Has observation": lambda r: r.observation is not None,
    }

    for phase_name, check_fn in phase_checks.items():
        count = sum(1 for run in final_runs if check_fn(run))
        status = "✅" if count > 0 else "❌"
        logger.info(f"{status} {phase_name}: {count}/{len(final_runs)}")

    # Show observations
    observations = [run.observation for run in final_runs if run.observation]
    if observations:
        logger.info(f"\n📊 Observations collected: {len(observations)}")
        for i, obs in enumerate(observations, 1):
            logger.info(f"   {i}. Score: {obs.score:.3f}, Cost: {obs.cost:.1f}h")

    # Success criteria
    success = (
        len(final_runs) >= config.num_trials and len(observations) >= config.num_trials - 1  # Allow one pending
    )

    if success:
        logger.info("\n🎉 TEST PASSED!")
    else:
        logger.info("\n❌ TEST FAILED - Not all trials completed")
        return 1

    return 0


if __name__ == "__main__":
    exit(run_test())
