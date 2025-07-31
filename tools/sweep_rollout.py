#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import logging
import os
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

    # Setup the sweep - only rank 0 does this, others wait
    try:
        run_once(
            lambda: setup_sweep(cfg, logger),
        )
    except Exception as e:
        logger.error(f"Sweep setup failed: {e}", exc_info=True)
        return 1

    num_consecutive_failures = 0
    exit_code = 0

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
                exit_code = 1
                break
        else:
            num_consecutive_failures = 0

    return exit_code


def run_single_rollout(cfg: DictConfig) -> int:
    """Run a single rollout."""
    logger.info(f"Starting single rollout for sweep: {cfg.sweep_name}")

    # Master node only
    run_name, train_job_cfg, protein_suggestion, wandb_run_id = run_once(
        lambda: prepare_sweep_run(cfg, logger),
    )

    # All ranks participate in training
    # The train.sh script handles distributed coordination
    train_for_run(
        run_name=run_name,
        train_job_cfg=train_job_cfg,
        wandb_run_id=wandb_run_id or "",  # Handle None case
        original_args=ORIGINAL_ARGS,
        logger=logger,
    )
    logger.info("Training completed...")

    config_path = os.path.join(train_job_cfg.run_dir, "sweep_eval_config.yaml")
    full_train_job_cfg = OmegaConf.load(config_path)
    assert isinstance(full_train_job_cfg, DictConfig)
    # Master node only
    eval_results = run_once(
        lambda: evaluate_rollout(
            full_train_job_cfg,
            protein_suggestion,
            metric=cfg.sweep.metric,
            sweep_name=cfg.sweep_name,
            logger=logger,
        ),
    )

    if eval_results is None:
        logger.error("Evaluation failed")
        raise RuntimeError("Evaluation failed")

    logger.info(f"Rollout completed successfully for run: {run_name}")
    return 0


def train_for_run(
    run_name: str,
    train_job_cfg: DictConfig,
    wandb_run_id: str,
    original_args: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess:
    """Launch training as a subprocess and wait for completion."""

    # Build the command exactly like the bash script
    cmd = [
        "./devops/train.sh",
        f"run={run_name}",
        f"data_dir={train_job_cfg.data_dir}",
        f"wandb.run_id={wandb_run_id}",
        f"wandb.group={train_job_cfg.sweep_name}",
        f"wandb.name={run_name}",
    ]

    # Pass through relevant arguments from the original command line
    # Filter out arguments that we're already setting explicitly
    if original_args:
        # TODO: Skim those keys
        skip_prefixes = [
            "run=",
            "sweep_name=",
            "data_dir=",
            "sweep_dir=",
            "wandb.run_id=",
            "wandb.group=",
            "wandb.name=",
        ]
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

    # Prepare environment variables for subprocess
    # This ensures distributed training variables are passed through
    env = os.environ.copy()

    # Log distributed training environment variables if present
    dist_vars = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NUM_NODES",
        "NODE_INDEX",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NUM_GPUS",
        "CUDA_VISIBLE_DEVICES",
    ]
    dist_env_info = {var: env.get(var) for var in dist_vars if env.get(var)}
    if dist_env_info:
        if logger:
            logger.info(f"[SWEEP:{run_name}] Distributed env vars: {dist_env_info}")
        else:
            print(f"[SWEEP:{run_name}] Distributed env vars: {dist_env_info}")

    # Check if we should enable batch scaling for multi-GPU
    # Note: We don't auto-enable this as it changes the effective batch size
    # Users should explicitly set this if they want batch scaling

    try:
        # Launch and wait (no capture_output to maintain real-time logging)
        # Pass the environment to ensure distributed training variables are available
        result = subprocess.run(cmd, check=True, env=env)
        return result

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[ERROR] Training failed for run: {run_name}")
        else:
            print(f"[ERROR] Training failed for run: {run_name}")
        raise Exception(f"Training failed for {run_name} with exit code {e.returncode}") from e


if __name__ == "__main__":
    sys.exit(main())
