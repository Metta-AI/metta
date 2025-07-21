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
import uuid
from logging import Logger

import hydra
from omegaconf import DictConfig

from metta.common.util.logging_helpers import setup_mettagrid_logger

logger = setup_mettagrid_logger("sweep_rollout")


class SweepRolloutError(Exception):
    """Custom exception for sweep rollout failures."""

    pass


def generate_dist_cfg_path(data_dir: str, sweep_name: str) -> str:
    """Generate distributed config path with process-unique hash."""
    dist_id = os.environ.get("DIST_ID", "localhost")
    process_hash = uuid.uuid4().hex[:8]
    return f"{data_dir}/sweep/{sweep_name}/dist_{dist_id}_{process_hash}.yaml"


def extract_run_id_from_config(dist_cfg_path: str) -> str:
    """Extract run ID from the generated distributed config file."""
    if not os.path.exists(dist_cfg_path):
        raise FileNotFoundError(f"Distributed config file not found: {dist_cfg_path}")

    with open(dist_cfg_path, "r") as f:
        content = f.read()

    # Parse line by line to match bash logic: grep '^run:' | awk '{print $2}'
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("run:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()

    raise ValueError(f"Failed to read run ID from {dist_cfg_path}")


def prepare_sweep_run(cfg: DictConfig, dist_cfg_path: str, logger: Logger) -> str:
    """Prepare sweep run using direct Python import."""
    from tools.sweep_prepare_run import setup_next_run

    logger.info(f"[SWEEP:{cfg.sweep_name}] Preparing next run...")

    # Ensure the dist config file is clean (remove if exists)
    if os.path.exists(dist_cfg_path):
        os.remove(dist_cfg_path)

    # Create a modified config for sweep_prepare_run
    prep_cfg = cfg.copy()
    prep_cfg.dist_cfg_path = dist_cfg_path

    try:
        # Direct function call instead of subprocess
        run_id = setup_next_run(prep_cfg, logger)

        # setup_next_run returns the run_id, but let's ensure it's not None
        if not run_id:
            # Fallback: extract from config file if setup_next_run somehow returns None
            run_id = extract_run_id_from_config(dist_cfg_path)

        logger.info(f"Generated run ID: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"[ERROR] Sweep run preparation failed: {cfg.sweep_name}")
        logger.error(f"Preparation error details: {e}")
        raise SweepRolloutError(f"Preparation failed for {cfg.sweep_name}") from e


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
        raise SweepRolloutError(f"Training failed for {sweep_name} with exit code {e.returncode}") from e


def run_evaluation(cfg: DictConfig, dist_cfg_path: str, data_dir: str, sweep_name: str, logger: Logger) -> bool:
    """Run evaluation using direct Python import."""
    from tools.sweep_eval import main as sweep_eval_main

    logger.info(f"[SWEEP:{cfg.sweep_name}] Starting evaluation phase...")

    # Create evaluation config
    eval_cfg = cfg.copy()
    eval_cfg.dist_cfg_path = dist_cfg_path
    eval_cfg.data_dir = f"{data_dir}/sweep/{sweep_name}/runs"

    # Build command string for logging (matching bash script output)
    args_str = f"dist_cfg_path={dist_cfg_path} data_dir={data_dir}/sweep/{sweep_name}/runs"
    logger.info(f"[SWEEP:{sweep_name}] Running: ./tools/sweep_eval.py {args_str}")

    try:
        # Direct function call instead of subprocess
        result = sweep_eval_main(eval_cfg)
        success = result == 0

        if not success:
            logger.error(f"[ERROR] Evaluation failed for sweep: {sweep_name}")
            raise SweepRolloutError(f"Evaluation failed for {sweep_name}")

        return success

    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed for sweep: {sweep_name}")
        logger.error(f"Evaluation error details: {e}")
        raise SweepRolloutError(f"Evaluation failed for {sweep_name}") from e


def run_single_rollout(cfg: DictConfig, logger: Logger) -> dict:
    """Complete rollout: prepare, train, evaluate."""

    # Validate required arguments
    if not cfg.get("sweep_name"):
        raise ValueError("'sweep_name' argument is required (e.g., sweep_name=my_sweep_name)")

    # Extract hardware argument from command line arguments (like bash script does)
    import sys

    hardware_arg = None
    for arg in sys.argv[1:]:
        if arg.startswith("+hardware="):
            hardware_arg = arg
            break

    # Get data directory from environment (set by sweep.sh)
    data_dir = os.environ.get("DATA_DIR", "./train_dir")

    # Generate distributed config path
    dist_cfg_path = generate_dist_cfg_path(data_dir, cfg.sweep_name)

    logger.info(f"[INFO] Starting sweep rollout: {cfg.sweep_name}")
    logger.info(f"Distributed config path: {dist_cfg_path}")

    try:
        # 1. Prepare run (Pure Python)
        run_id = prepare_sweep_run(cfg, dist_cfg_path, logger)

        # Extract run_id from generated config file (matching bash script logic)
        run_id = extract_run_id_from_config(dist_cfg_path)

        if not run_id:
            raise ValueError(f"Failed to read run ID from {dist_cfg_path}")

        # 2. Training phase (Subprocess only)
        logger.info(f"[SWEEP:{cfg.sweep_name}] Starting training phase...")
        train_result = run_training(
            run_id=run_id,
            dist_cfg_path=dist_cfg_path,
            data_dir=data_dir,
            sweep_name=cfg.sweep_name,
            hardware_arg=hardware_arg,
            logger=logger,
        )

        # 3. Evaluation phase (Pure Python)
        eval_success = run_evaluation(cfg, dist_cfg_path, data_dir, cfg.sweep_name, logger)

        logger.info(f"[SUCCESS] Sweep rollout completed: {cfg.sweep_name}")

        return {
            "run_id": run_id,
            "train_returncode": train_result.returncode,
            "eval_success": eval_success,
            "dist_cfg_path": dist_cfg_path,
        }

    except Exception as e:
        logger.error(f"Sweep rollout failed: {e}")
        raise


@hydra.main(config_path="../../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""
    # Don't use @metta_script decorator since we need to generate run ID first
    # Use module-level logger created at import time

    try:
        # Set a temporary run ID to satisfy config validation
        # This will be overridden by the actual run ID from setup_next_run
        if not cfg.get("run"):
            cfg.run = "temp_run_id"

        # Run complete rollout
        result = run_single_rollout(cfg, logger)
        logger.info(f"Rollout completed successfully: {result}")
        return 0

    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
