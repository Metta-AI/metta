#!/usr/bin/env python3
"""Simple hardware scaling experiment using standard sweeps.

This script launches independent sweeps for each hardware configuration,
then analyzes the results to understand scaling laws and trade-offs.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.dispatcher import LocalDispatcher, SkypilotDispatcher
from metta.adaptive.stores import WandbStore
from metta.common.util.log_config import init_logging
from metta.sweep import BatchedSyncedOptimizingScheduler, BatchedSyncedSchedulerConfig
from metta.sweep.protein_config import ParameterConfig, ProteinConfig

# Import the optimized config function
from experiments.adaptive.hardware_scaling.optimized_sweep_config import (
    create_optimized_protein_config,
)

logger = logging.getLogger(__name__)


# This function is now replaced by the optimized version from optimized_sweep_config.py
# Keeping it here commented for reference
# def create_protein_config_for_hardware(...)


def launch_sweep_for_hardware(
    gpus: int,
    nodes: int,
    experiment_base_name: str,
    recipe_module: str,
    wandb_entity: str,
    wandb_project: str,
    dispatcher_type: str = "local",
    max_trials: int = 50,
    batch_size: int = 4,
    total_timesteps: int = 300_000_000,
) -> str:
    """Launch a sweep for a specific hardware configuration.

    Args:
        gpus: Number of GPUs
        nodes: Number of nodes
        experiment_base_name: Base name for the experiment
        recipe_module: Recipe module to use
        wandb_entity: WandB entity
        wandb_project: WandB project
        dispatcher_type: Type of dispatcher to use
        max_trials: Maximum trials for this sweep
        batch_size: Batch size for sweep (parallel trials)
        total_timesteps: Total timesteps per trial

    Returns:
        Experiment ID for tracking
    """
    # Create unique experiment ID for this hardware config
    hw_id = f"g{gpus}_n{nodes}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_base_name}_{hw_id}_{timestamp}"

    logger.info(f"Launching sweep for hardware {hw_id}: {experiment_id}")

    # Create optimized Protein config for this hardware with correct defaults
    protein_config = create_optimized_protein_config(
        gpus=gpus,
        nodes=nodes,
        num_agents=24,
        bptt_horizon=64,  # Using correct default from TrainerConfig
    )

    # Create scheduler config
    scheduler_config = BatchedSyncedSchedulerConfig(
        max_trials=max_trials,
        batch_size=batch_size,
        recipe_module=recipe_module,
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        experiment_id=experiment_id,
        protein_config=protein_config,
        gpus=gpus,
        nodes=nodes,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )

    # Initialize components
    store = WandbStore(entity=wandb_entity, project=wandb_project)

    if dispatcher_type == "skypilot":
        dispatcher = SkypilotDispatcher()
    else:
        dispatcher = LocalDispatcher(capture_output=False)

    scheduler = BatchedSyncedOptimizingScheduler(
        config=scheduler_config,
        state=None,
    )

    # Create adaptive controller
    adaptive_config = AdaptiveConfig(
        max_parallel=batch_size,
        monitoring_interval=60,
        resume=False,
        experiment_tags=[
            "hardware-scaling",
            hw_id,
            recipe_module.split(".")[-1],
        ],
    )

    controller = AdaptiveController(
        experiment_id=experiment_id,
        scheduler=scheduler,
        dispatcher=dispatcher,
        store=store,
        config=adaptive_config,
    )

    # Run the sweep
    logger.info(f"Starting sweep for {hw_id} with {max_trials} trials...")

    # Run in background or synchronously depending on setup
    # For now, return immediately and let it run
    # In practice, might want to use threading or multiprocessing

    from threading import Thread

    def run_sweep():
        try:
            # Import the sweep tool's hooks
            from metta.tools.sweep import create_on_eval_completed_hook

            controller.run(
                on_eval_completed=create_on_eval_completed_hook(protein_config.metric),
            )
        except Exception as e:
            logger.error(f"Sweep failed for {hw_id}: {e}")

    thread = Thread(target=run_sweep, daemon=True)
    thread.start()

    return experiment_id


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run hardware scaling experiment using independent sweeps"
    )

    # WandB settings
    parser.add_argument("--wandb-entity", type=str, required=True, help="WandB entity")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hardware-scaling",
        help="WandB project",
    )

    # Hardware configurations
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        help="GPU counts to test",
    )
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Node counts to test",
    )

    # Sweep settings
    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="Maximum trials per hardware configuration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for sweeps (parallel trials)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000_000,
        help="Total timesteps per trial",
    )

    # Experiment settings
    parser.add_argument(
        "--recipe",
        type=str,
        default="experiments.recipes.arena_basic_easy_shaped",
        help="Recipe module to use",
    )
    parser.add_argument(
        "--dispatcher",
        type=str,
        choices=["local", "skypilot"],
        default="local",
        help="Dispatcher type",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hardware_scaling",
        help="Base experiment name",
    )

    # Execution mode
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run hardware configs sequentially instead of in parallel",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Delay between launching sweeps (seconds)",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging(level=logging.INFO)

    # Generate hardware configurations
    hardware_configs = []
    for gpus in args.gpu_counts:
        for nodes in args.node_counts:
            # Only include valid configs (max 8 GPUs per node)
            if gpus <= 8 * nodes:
                hardware_configs.append((gpus, nodes))

    logger.info(f"Testing {len(hardware_configs)} hardware configurations")
    logger.info(f"Configurations: {hardware_configs}")

    # Launch sweeps
    experiment_ids = []

    for gpus, nodes in hardware_configs:
        logger.info(f"Launching sweep for {gpus} GPUs on {nodes} nodes")

        exp_id = launch_sweep_for_hardware(
            gpus=gpus,
            nodes=nodes,
            experiment_base_name=args.experiment_name,
            recipe_module=args.recipe,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            dispatcher_type=args.dispatcher,
            max_trials=args.max_trials,
            batch_size=args.batch_size,
            total_timesteps=args.total_timesteps,
        )

        experiment_ids.append(exp_id)
        logger.info(f"Launched experiment {exp_id}")

        if args.sequential:
            # Wait for this sweep to complete before launching next
            logger.info("Running sequentially - waiting for completion...")
            # In practice, would monitor for completion
            # For now, just wait
            input("Press Enter when sweep is complete...")
        elif args.delay > 0:
            # Add delay between launches
            logger.info(f"Waiting {args.delay} seconds before next launch...")
            time.sleep(args.delay)

    logger.info("All sweeps launched!")
    logger.info(f"Experiment IDs: {experiment_ids}")
    logger.info("")
    logger.info("Monitor progress in WandB:")
    logger.info(f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    logger.info("")
    logger.info("Once complete, run analysis:")
    logger.info("python experiments/adaptive/hardware_scaling/analysis.py")

    # Keep main thread alive if running parallel sweeps
    if not args.sequential:
        try:
            logger.info("Sweeps running in background. Press Ctrl+C to exit.")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Exiting...")


if __name__ == "__main__":
    main()
