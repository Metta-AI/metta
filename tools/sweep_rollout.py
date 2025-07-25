#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import gc
import logging
import os
import signal
import subprocess
import sys
import time

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from metta.common.util.lock import run_once
from metta.sweep.sweep_lifecycle import evaluate_rollout, prepare_sweep_run, setup_sweep

logger = logging.getLogger(__name__)

# Global variable to store original command-line arguments
ORIGINAL_ARGS = []

# Global shutdown flag
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Store original command-line arguments for later use
    global ORIGINAL_ARGS
    ORIGINAL_ARGS = sys.argv[1:]  # Skip the script name

    logger.info(f"Starting sweep rollout with config: {list(cfg.keys())}")
    logger.debug(f"Full config: {OmegaConf.to_yaml(cfg)}")
    logger.debug(f"Original command-line args: {ORIGINAL_ARGS}")

    # Set up environment for sweep orchestration to use a different port
    # Training will use the default 8008, sweep orchestration will use 8007
    sweep_env_setup()

    # Setup the sweep - only rank 0 does this, others wait
    try:
        wandb_sweep_id = run_once(
            lambda: setup_sweep(cfg, logger),
            destroy_on_finish=False,
        )
    except Exception as e:
        logger.error(f"Sweep setup failed: {e}", exc_info=True)
        return 1

    # Set the sweep ID in the config
    cfg.sweep_id = wandb_sweep_id
    num_consecutive_failures = 0
    exit_code = 0  # Track exit status

    while not SHUTDOWN_REQUESTED:
        err_occurred = False
        # Run the rollout
        try:
            run_single_rollout(cfg)

            # Cleanup after successful run
            logger.info("Performing cleanup after successful rollout...")
            _cleanup_resources()

        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            err_occurred = True

            # Cleanup after failed run too
            logger.info("Performing cleanup after failed rollout...")
            _cleanup_resources()

            logger.info(f"Waiting {cfg.rollout_retry_delay} seconds before retry...")
            time.sleep(cfg.rollout_retry_delay)

        if err_occurred:
            num_consecutive_failures += 1
            if num_consecutive_failures > cfg.max_consecutive_failures:
                logger.error(f"Max consecutive failures reached: {cfg.max_consecutive_failures}")
                exit_code = 1  # Set error exit code
                break
        else:
            num_consecutive_failures = 0

    # Coordinated shutdown
    exit_code = coordinated_shutdown(exit_code)

    return exit_code


def coordinated_shutdown(exit_code: int) -> int:
    """Perform coordinated shutdown across all nodes."""
    logger.info("Initiating coordinated shutdown...")

    try:
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Create a tensor to share exit codes
            exit_codes = torch.tensor([exit_code], dtype=torch.int32)
            if torch.cuda.is_available():
                exit_codes = exit_codes.cuda()

            # Gather all exit codes to rank 0
            if world_size > 1:
                all_exit_codes = [torch.zeros_like(exit_codes) for _ in range(world_size)]
                dist.all_gather(all_exit_codes, exit_codes)

                # Use the worst exit code from any node
                if rank == 0:
                    final_exit_code = int(max(code.item() for code in all_exit_codes))
                else:
                    final_exit_code = exit_code

                # Broadcast final exit code to all nodes
                final_exit_tensor = torch.tensor([final_exit_code], dtype=torch.int32)
                if torch.cuda.is_available():
                    final_exit_tensor = final_exit_tensor.cuda()
                dist.broadcast(final_exit_tensor, src=0)
                exit_code = int(final_exit_tensor.item())

                # Synchronize all nodes before cleanup
                logger.info(f"Rank {rank}: Synchronizing before shutdown...")
                dist.barrier()

            # Clean up process group
            logger.info(f"Rank {rank}: Destroying process group...")
            dist.destroy_process_group()

        else:
            logger.info("No distributed process group to clean up")

    except Exception as e:
        logger.error(f"Error during coordinated shutdown: {e}", exc_info=True)
        exit_code = 1

    # Final resource cleanup
    _final_cleanup()

    logger.info(f"Shutdown complete with exit code: {exit_code}")
    return exit_code


def _final_cleanup():
    """Perform final cleanup operations."""
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

        # Small delay to ensure all cleanup completes
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error during final cleanup: {e}", exc_info=True)


def sweep_env_setup():
    """Set up environment for sweep orchestration to use a different port."""
    # If we're in a distributed sweep setup, use a different port for orchestration
    if "MASTER_PORT" in os.environ:
        current_port = os.environ.get("MASTER_PORT", "8008")
        if current_port == "8008":
            # Use 8007 for sweep orchestration, leaving 8008 for training
            os.environ["MASTER_PORT"] = "8007"
            logger.info(f"Sweep orchestration using port 8007 (training will use {current_port})")


def _cleanup_resources():
    """Clean up resources between sweep runs to prevent accumulation."""
    # Clean up distributed process group if initialized
    if dist.is_initialized():
        logger.debug("Destroying process group...")
        dist.destroy_process_group()

    # Clear GPU memory cache if available
    if torch.cuda.is_available():
        logger.debug("Clearing GPU memory cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    logger.debug("Running garbage collection...")
    gc.collect()

    # Small delay to ensure cleanup completes
    time.sleep(0.5)


def run_single_rollout(cfg: DictConfig) -> int:
    """Run a single rollout."""
    logger.info(f"Starting single rollout for sweep: {cfg.sweep_name}")

    # Master node only
    preparation_result = run_once(lambda: prepare_sweep_run(cfg, logger))

    if preparation_result is None:
        logger.error("Failed to prepare sweep rollout")
        raise RuntimeError("Sweep preparation failed")

    run_name, downstream_cfg, protein_suggestion = preparation_result

    # All ranks participate in training
    # The train.sh script handles distributed coordination
    train_for_run(
        run_name=run_name,
        dist_cfg_path=downstream_cfg.dist_cfg_path,
        data_dir=downstream_cfg.data_dir,
        original_args=ORIGINAL_ARGS,
    )
    logger.info("Training completed...")

    # Master node only
    eval_results = run_once(lambda: evaluate_rollout(downstream_cfg, protein_suggestion, logger))

    if eval_results is None:
        logger.error("Evaluation failed")
        raise RuntimeError("Evaluation failed")

    logger.info(f"Rollout completed successfully for run: {run_name}")
    return 0


def train_for_run(
    run_name: str,
    dist_cfg_path: str,
    data_dir: str,
    original_args: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """Launch training as a subprocess and wait for completion."""

    # Reset environment for training subprocess to use the standard port
    training_env = os.environ.copy()
    training_env["MASTER_PORT"] = "8008"  # Ensure training uses the standard port

    # Build the command exactly like the bash script
    cmd = [
        "./devops/train.sh",
        f"run={run_name}",
        f"dist_cfg_path={dist_cfg_path}",
        f"data_dir={data_dir}",
    ]

    # Pass through relevant arguments from the original command line
    # Filter out arguments that we're already setting explicitly
    if original_args:
        skip_prefixes = ["run=", "sweep_name=", "dist_cfg_path=", "data_dir="]
        for arg in original_args:
            # Skip arguments we're already setting
            if any(arg.startswith(prefix) for prefix in skip_prefixes):
                continue
            # Pass through everything else (like hardware configs, wandb settings, etc.)
            cmd.append(arg)

    if logger:
        logger.info(f"[SWEEP:{run_name}] Running: {' '.join(cmd)}")
    else:
        print(f"[SWEEP:{run_name}] Running: {' '.join(cmd)}")

    try:
        # Launch and wait (no capture_output to maintain real-time logging)
        result = subprocess.run(cmd, check=True, env=training_env)
        return result

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[ERROR] Training failed for run: {run_name}")
        else:
            print(f"[ERROR] Training failed for run: {run_name}")
        raise Exception(f"Training failed for {run_name} with exit code {e.returncode}") from e


if __name__ == "__main__":
    sys.exit(main())
