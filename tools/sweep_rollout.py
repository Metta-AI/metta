#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import logging
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.common.util.lock import run_once
from metta.sweep.sweep_lifecycle import evaluate_rollout, prepare_sweep_run, setup_sweep

logger = logging.getLogger(__name__)

# Global variable to store original command-line arguments
ORIGINAL_ARGS = []


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    # Store original command-line arguments for later use
    global ORIGINAL_ARGS
    ORIGINAL_ARGS = sys.argv[1:]  # Skip the script name

    logger.info(f"Starting sweep rollout with config: {list(cfg.keys())}")
    logger.debug(f"Full config: {OmegaConf.to_yaml(cfg)}")
    logger.debug(f"Original command-line args: {ORIGINAL_ARGS}")

    # Setup the sweep - only rank 0 does this, others wait
    try:
        wandb_sweep_id = run_once(
            lambda: setup_sweep(cfg, logger),
        )
    except Exception as e:
        logger.error(f"Sweep setup failed: {e}", exc_info=True)
        return 1

    # Set the sweep ID in the config
    cfg.sweep_id = wandb_sweep_id
    num_consecutive_failures = 0

    while True:
        err_occurred = False
        # Run the rollout
        try:
            run_single_rollout(cfg)
        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            err_occurred = True
            logger.info(f"Waiting {cfg.rollout_retry_delay} seconds before retry...")
            time.sleep(cfg.rollout_retry_delay)
        if err_occurred:
            num_consecutive_failures += 1
            if num_consecutive_failures > cfg.max_consecutive_failures:
                logger.error(f"Max consecutive failures reached: {cfg.max_consecutive_failures}")
                break
        else:
            num_consecutive_failures = 0

    return 0


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
        result = subprocess.run(cmd, check=True)
        return result

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[ERROR] Training failed for run: {run_name}")
        else:
            print(f"[ERROR] Training failed for run: {run_name}")
        raise Exception(f"Training failed for {run_name} with exit code {e.returncode}") from e


if __name__ == "__main__":
    sys.exit(main())
