#!/usr/bin/env -S uv run

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import logging

import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import os
from logging import Logger

from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.lock import run_once
from metta.sweep.wandb_utils import create_wandb_sweep, sweep_id_from_name
from metta.util.metta_script import metta_script

logger = logging.getLogger(__name__)


def main(cfg: DictConfig) -> int:
    # Extract sweep base name from CLI sweep_name parameter (e.g., "simple_sweep")
    # Individual training runs will be "simple_sweep.r.0", etc.

    # TODO: Config should handle this
    # Looking ahead, I don't even think we really need it.
    cfg.wandb.name = cfg.sweep_name

    # TODO: Check run_once -- I think it's messing with the CUDA context.
    run_once(lambda: create_sweep(cfg, logger))

    return 0


def create_sweep(cfg: DictConfig | ListConfig, logger: Logger) -> None:
    """
    Create a new sweep with the given name. If the sweep already exists, skip creation.
    Save the sweep configuration to sweep_dir/config.yaml.
    """

    # Check if sweep already exists
    wandb_sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.wandb.entity, cfg.sweep_name)

    if wandb_sweep_id is not None:
        logger.info(f"Sweep already exists in WandB, skipping creation for: {cfg.sweep_name}")

    else:
        # TODO: Check that sweep_dir is passed as a CLI arg and not into the config.
        logger.info(f"Creating new WandB sweep: {cfg.sweep_name}: {cfg.sweep_dir}")
        # Create dummy sweep using static methods from protein_wandb (Protein will control all parameters)
        wandb_sweep_id = create_wandb_sweep(cfg.sweep_name, cfg.wandb.entity, cfg.wandb.project)

    # Save sweep metadata locally
    # in join(cfg.sweep_dir, "metadata.yaml"
    os.makedirs(cfg.runs_dir, exist_ok=True)
    OmegaConf.save(
        {
            "sweep": cfg.sweep,  # The sweep parameters/settings
            "sweep_name": cfg.sweep_name,
            "wandb_sweep_id": wandb_sweep_id,
            "wandb_path": f"{cfg.wandb.entity}/{cfg.wandb.project}/{wandb_sweep_id}",
        },
        os.path.join(cfg.sweep_dir, "metadata.yaml"),
    )


metta_script(main, "sweep_job")
