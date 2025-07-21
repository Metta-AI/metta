#!/usr/bin/env -S uv run
"""
sweep_rollout.py - Execute a single sweep rollout in Python

This module replaces sweep_rollout.sh with a Python implementation that:
- Uses direct function imports for preparation and evaluation phases
- Only launches training as a subprocess via devops/train.sh
- Maintains compatibility with existing sweep infrastructure
"""

import os
import subprocess
import sys
from logging import Logger

import hydra
from omegaconf import DictConfig

from metta.common.util.lock import run_once
from metta.common.util.logging_helpers import setup_mettagrid_logger
from tools.sweep_eval import main as sweep_eval_main
from tools.sweep_prepare_run import setup_next_run

logger = setup_mettagrid_logger("sweep_rollout")


@hydra.main(config_path="../../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    # Don't use @metta_script decorator since we need to generate run ID first
    # Use module-level logger created at import time

    try:
        # Validate required arguments
        if not cfg.get("sweep_name"):
            raise ValueError("'sweep_name' argument is required (e.g., sweep_name=my_sweep_name)")

        # Extract hardware argument from command line arguments (like bash script does)
        hardware_arg = None
        for arg in sys.argv[1:]:
            if arg.startswith("+hardware="):
                hardware_arg = arg
                break

        # Get data directory from environment (set by sweep.sh)
        data_dir = os.environ.get("DATA_DIR", "./train_dir")

        logger.info(f"[INFO] Starting sweep rollout: {cfg.sweep_name}")

        # 1. Preparation phase (ONLY rank 0 - through run_once)
        run_id, dist_cfg_path = run_once(lambda: prepare_sweep_run(cfg, logger))

        # 2. Training phase (ALL ranks participate)
        logger.info(f"[SWEEP:{cfg.sweep_name}] Starting training phase...")
        run_training(
            run_id=run_id,
            dist_cfg_path=dist_cfg_path,
            data_dir=data_dir,
            sweep_name=cfg.sweep_name,
            hardware_arg=hardware_arg,
            logger=logger,
        )

        # 3. Evaluation phase (ONLY rank 0 - handled by sweep_eval internally)
        logger.info(f"[SWEEP:{cfg.sweep_name}] Starting evaluation phase...")

        # Create evaluation config
        eval_cfg = cfg.copy()
        eval_cfg.dist_cfg_path = dist_cfg_path
        eval_cfg.data_dir = f"{data_dir}/sweep/{cfg.sweep_name}/runs"

        # Log the command for consistency with bash script
        logger.info(
            f"[SWEEP:{cfg.sweep_name}] Running: ./tools/sweep_eval.py dist_cfg_path={dist_cfg_path} "
            f"data_dir={data_dir}/sweep/{cfg.sweep_name}/runs"
        )

        # Direct function call to sweep_eval main
        result = sweep_eval_main(eval_cfg)
        if result != 0:
            raise Exception(f"Evaluation failed for {cfg.sweep_name}")

        logger.info(f"[SUCCESS] Sweep rollout completed: {cfg.sweep_name}")
        return 0

    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        return 1


def prepare_sweep_run(cfg: DictConfig, logger: Logger) -> tuple[str, str]:
    """
    Prepare a sweep run by calling setup_next_run.
    Returns the generated run ID and the dist_cfg_path.
    This function is called by run_once, so it is only executed by rank 0.
    """
    try:
        # Direct function call instead of subprocess
        run_id, dist_cfg_path = setup_next_run(cfg, logger)

        if not run_id:
            raise ValueError("setup_next_run returned None/empty run_id")

        # Construct final dist_cfg_path in run directory
        logger.info(f"Generated run ID: {run_id}")
        logger.info(f"Dist config written to: {dist_cfg_path}")

        return run_id, dist_cfg_path

    except Exception as e:
        logger.error(f"[ERROR] Sweep run preparation failed: {cfg.sweep_name}")
        logger.error(f"Preparation error details: {e}")
        raise Exception(f"Preparation failed for {cfg.sweep_name}") from e


def run_training(
    run_id: str,
    dist_cfg_path: str,
    data_dir: str,
    sweep_name: str,
    hardware_arg: str | None = None,
    logger: Logger | None = None,
) -> subprocess.CompletedProcess:
    """Launch training as a subprocess and wait for completion."""

    # Build the command exactly like the bash script
    cmd = [
        "./devops/train.sh",
        f"run={run_id}",
        f"dist_cfg_path={dist_cfg_path}",
        f"data_dir={data_dir}/sweep/{sweep_name}/runs",
    ]

    if hardware_arg:
        cmd.append(hardware_arg)

    if logger:
        logger.info(f"[SWEEP:{sweep_name}] Running: {' '.join(cmd)}")
    else:
        print(f"[SWEEP:{sweep_name}] Running: {' '.join(cmd)}")

    try:
        # Launch and wait (no capture_output to maintain real-time logging)
        result = subprocess.run(cmd, check=True)
        return result

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[ERROR] Training failed for sweep: {sweep_name}")
        else:
            print(f"[ERROR] Training failed for sweep: {sweep_name}")
        raise Exception(f"Training failed for {sweep_name} with exit code {e.returncode}") from e


if __name__ == "__main__":
    sys.exit(main())
