#!/usr/bin/env -S uv run
"""
sweep_execute.py - Execute a sweep in Python

This module implements a simplified sweep pipeline that:
- Generates protein suggestions and passes them as command-line arguments
- Launches training as a subprocess via devops/train.sh with direct parameter passing
- Evaluates results after training completes
- No intermediate config files or special directory structures needed
"""

import logging
import os
import subprocess
import sys
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from metta.common.util.datastruct import convert_dict_to_cli_args
from metta.common.util.lock import run_once
from metta.sweep.sweep_lifecycle import (
    evaluate_sweep_rollout,
    initialize_sweep,
    prepare_sweep_run,
)
from metta.util.metta_script import metta_script

logger = logging.getLogger(__name__)


def main(cfg: DictConfig) -> int:
    """Main entry point for sweep rollout."""

    logger.info(f"Starting sweep rollout with config: {list(cfg.keys())}")

    # Capture the original CLI args to optionally pass through to training
    original_args: list[str] = sys.argv[1:]
    logger.debug(f"Captured {len(original_args)} original CLI arguments")

    # Initialize the sweep - only rank 0 does this, others wait
    try:
        run_once(
            lambda: initialize_sweep(cfg, logger),
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
            run_single_rollout(cfg, original_args=original_args)
        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            err_occurred = True
            logger.info(f"Waiting {cfg.settings.rollout_retry_delay} seconds before retry...")
            time.sleep(cfg.settings.rollout_retry_delay)

        if err_occurred:
            num_consecutive_failures += 1
            if num_consecutive_failures > cfg.settings.max_consecutive_failures:
                logger.error(f"Max consecutive failures reached: {cfg.settings.max_consecutive_failures}")
                exit_code = 1
                break
        else:
            num_consecutive_failures = 0

    return exit_code


def run_single_rollout(cfg: DictConfig, original_args: list[str] | None = None) -> int:
    """Run a single rollout using the simplified pipeline."""
    logger.info(f"Starting single rollout for sweep: {cfg.sweep_name}")

    # Generate configuration for the sweep run - get run name and protein suggestion
    # Only rank 0 does this, others wait
<<<<<<< HEAD:tools/sweep_execute.py
    results = run_once(
        lambda: prepare_sweep_run(cfg, logger),
    )

    if results is None:
=======
    run_name, protein_suggestion, phase_index = run_once(
        lambda: prepare_sweep_run(cfg, logger),
    )
    cfg.sweep = cfg.settings.phase_schedule[phase_index].sweep
    if run_name is None:
>>>>>>> 2bf4ad753 (feat(sweep): add phase support):tools/sweep_rollout.py
        logger.error("Failed to prepare sweep run")
        return 1

    run_name, protein_suggestion = results

    logger.info(f"Prepared sweep run: {run_name}")

    # Launch training subprocess with all parameters as command-line args
    launch_training_subprocess(
        run_name=run_name,
        protein_suggestion=protein_suggestion,
        sweep_name=cfg.sweep_name,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        cfg=cfg,
        original_args=original_args,
    )
    logger.info("Training completed...")
    run_dir = os.path.join(cfg.data_dir, run_name)
    config_path = os.path.join(run_dir, "sweep_eval_config.yaml")
    full_train_job_cfg = OmegaConf.load(config_path)
    assert isinstance(full_train_job_cfg, DictConfig)
    # Master node only
    eval_results = run_once(
        lambda: evaluate_sweep_rollout(
            full_train_job_cfg,
            protein_suggestion,
            metric=cfg.sweep.metric,
            sweep_name=cfg.sweep_name,
        ),
    )
    if eval_results is None:
        logger.error("Evaluation failed")
        raise RuntimeError("Evaluation failed")
    logger.info(f"Rollout completed successfully for run: {run_name}")

    return 0


def launch_training_subprocess(
    run_name: str,
    protein_suggestion: dict[str, Any],
    sweep_name: str,
    wandb_entity: str,
    wandb_project: str,
    cfg: DictConfig,
    original_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Launch training subprocess with hyperparameter suggestions as CLI arguments."""

    # Build the base command
    cmd = [
        "./devops/train.sh",
        f"run={run_name}",
    ]

    # Add WandB configuration
    cmd.extend(
        [
            f"wandb.entity={wandb_entity}",
            f"wandb.project={wandb_project}",
            f"wandb.group={sweep_name}",
            f"wandb.name={run_name}",
        ]
    )

    # Add sweep-specific trainer and sim configuration overrides from cfg
    # Extract relevant config sections and convert to command-line arguments
    config_sections = {}
    if "trainer" in cfg.sweep_job_overrides:
        config_sections["trainer"] = OmegaConf.to_container(cfg.sweep_job_overrides.trainer, resolve=True)

    # Convert config sections to command-line arguments using convert_suggestion_to_cli_args
    if config_sections:
        config_args = convert_dict_to_cli_args(config_sections)
        cmd.extend(config_args)
        logger.info(f"Config args: {config_args}")

    # Convert protein suggestion to command-line arguments
    suggestion_args = convert_dict_to_cli_args(protein_suggestion)
    cmd.extend(suggestion_args)

    # Pass through relevant arguments from the original command line
    # Filter out arguments that we're already setting explicitly
    if original_args:
        skip_prefixes = [
            "run=",
            "sweep_name=",
        ]
        for arg in original_args:
            # Skip arguments we're already setting or that start with trainer.
            if any(arg.startswith(prefix) for prefix in skip_prefixes):
                continue
            # Pass through everything else
            cmd.append(arg)

    # We can either do this or pass it through CLI args
    # I would prefer the latter.
    cmd.append(f"sim={cfg.sim_name}")

    logger.info(f"[SWEEP:{run_name}] Running training with {len(suggestion_args)} parameter overrides")
    logger.debug(f"[SWEEP:{run_name}] Full command: {' '.join(cmd)}")

    try:
        # Launch and wait (no capture_output to maintain real-time logging)
        result = subprocess.run(cmd, check=True)
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Training failed for run: {run_name}")
        raise Exception(f"Training failed for {run_name} with exit code {e.returncode}") from e


# Use metta_script to handle initialization and configuration
metta_script(main, config_name="sweep_job")
