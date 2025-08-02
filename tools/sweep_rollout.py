#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.lock import run_once
from metta.common.wandb.wandb_context import WandbContext
from metta.sweep.sweep_lifecycle import evaluate_rollout, prepare_sweep_run, setup_sweep
from tools.train import setup_device_and_distributed, train

logger = logging.getLogger(__name__)

# Global variable to store original command-line arguments
ORIGINAL_ARGS = []


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    record_heartbeat()
    logger.info(
        "Sweep rollout starting on "
        + f"{os.environ.get('NODE_INDEX', '0')}: "
        + f"{os.environ.get('LOCAL_RANK', '0')} ({cfg.device})"
    )

    # Use shared distributed setup function
    device, is_master, world_size, rank = setup_device_and_distributed(cfg.device)

    # Update cfg.device to include the local rank if distributed
    cfg.device = str(device)

    # Setup the sweep - run_once ensures only master does this
    try:
        run_once(lambda: setup_sweep(cfg, logger))
        logger.info("Sweep setup completed")
    except Exception as e:
        logger.error(f"Sweep setup failed: {e}", exc_info=True)
        return 1

    # Configuration for rollout loop
    max_consecutive_failures = int(os.environ.get("MAX_CONSECUTIVE_FAILURES", "3"))
    rollout_retry_delay = int(os.environ.get("ROLLOUT_RETRY_DELAY", "60"))
    num_consecutive_failures = 0

    # Main rollout loop
    while True:
        try:
            run_single_rollout(cfg, str(device), is_master, world_size, rank)
            logger.info("Rollout completed successfully")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            num_consecutive_failures = 0
        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            num_consecutive_failures += 1

            if num_consecutive_failures >= max_consecutive_failures:
                logger.error(f"Max consecutive failures reached: {num_consecutive_failures}")
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                return 1

            logger.info(f"Consecutive failures: {num_consecutive_failures}/{max_consecutive_failures}")
            logger.info(f"Waiting {rollout_retry_delay} seconds before retry...")
            time.sleep(rollout_retry_delay)

    # This should never be reached
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


def run_single_rollout(cfg: DictConfig, device: str, is_master: bool, world_size: int, rank: int) -> int:
    """Run a single rollout."""
    logger.info(f"Starting single rollout for sweep: {cfg.sweep_name}")
    record_heartbeat()

    # Prepare the sweep run - run_once ensures only master does this
    prepare_result = run_once(lambda: prepare_sweep_run(cfg, logger))
    if prepare_result is not None:
        run_name, train_job_cfg, protein_suggestion, _ = prepare_result
    else:
        run_name = train_job_cfg = protein_suggestion = None

    # Ensure we have valid config after broadcast
    if train_job_cfg is None:
        raise RuntimeError("Failed to get train_job_cfg from master")

    train_job_cfg_final = _build_train_cfg(train_job_cfg)
    logger.info(
        f"Training with config: {json.dumps(OmegaConf.to_container(train_job_cfg_final, resolve=True), indent=2)}"
    )

    # All ranks participate in training
    # Only master gets the wandb_run, others get None
    logger.info(f"Rank {rank} starting training")

    # Master handles WandB context
    if is_master:
        logger.info(f"Rank {rank} creating WandB context")
        with WandbContext(train_job_cfg_final.wandb, train_job_cfg_final) as wandb_run:
            logger.info(f"Rank {rank} starting training with WandB")
            train(train_job_cfg_final, wandb_run, logger)
            logger.info(f"Rank {rank} completed training with wandb context")
        logger.info(f"Rank {rank} exited WandB context")
    else:
        # Non-master ranks train without WandB
        train(train_job_cfg_final, None, logger)
        logger.info(f"Rank {rank} completed training")

    # Flush logs to ensure we see where we are
    sys.stdout.flush()
    sys.stderr.flush()

    # Evaluate the rollout - run_once ensures only master does this
    # protein_suggestion is guaranteed to be valid after broadcast
    assert protein_suggestion is not None
    logger.info(f"Rank {rank} starting evaluation phase")
    eval_results = run_once(
        lambda: evaluate_rollout(
            train_job_cfg_final,
            protein_suggestion,
            metric=cfg.sweep.metric,
            sweep_name=cfg.sweep_name,
            logger=logger,
        )
    )
    logger.info(f"Rank {rank} completed evaluation phase")

    if eval_results is None:
        logger.error("Evaluation failed")
        raise RuntimeError("Evaluation failed")

    logger.info(f"Rollout completed successfully for run: {run_name}")

    return 0


def _build_train_cfg(train_job_cfg: DictConfig) -> DictConfig:
    """
    Build the training configuration by loading common.yaml and merging with sweep_train_job.
    """
    # Get the path to common.yaml
    current_file_path = Path(__file__).resolve()
    config_dir = current_file_path.parent.parent / "configs"

    # Load common.yaml directly
    common_cfg = OmegaConf.load(config_dir / "common.yaml")

    # Merge common with sweep_train_job (sweep_train_job takes precedence)
    train_cfg = OmegaConf.merge(common_cfg, train_job_cfg)

    # Set cmd to 'train' since we're training, not sweeping
    train_cfg.cmd = "train"

    # Ensure we return a DictConfig
    assert isinstance(train_cfg, DictConfig)
    return train_cfg


if __name__ == "__main__":
    sys.exit(main())
