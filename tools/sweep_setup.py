#!/usr/bin/env -S uv run

"""Setup and initialize a hyperparameter optimization sweep.

This tool sets up the initial state for a distributed hyperparameter sweep:
1. Creates or verifies WandB sweep exists
2. Registers sweep in centralized coordination database
3. Creates local directory structure and metadata files

Backend URL Configuration:
- By default, uses stats_server_uri from your Hydra config (e.g., from common.yaml)
- Production: stats_server_uri: https://api.observatory.softmax-research.net
- Local development: stats_server_uri: http://localhost:8000
- Authentication uses machine tokens from ~/.metta/observatory_tokens.yaml

Example usage:
  # Use production backend (default from common.yaml)
  uv run python tools/sweep_setup.py sweep_name=my_sweep

  # Override to use local backend
  uv run python tools/sweep_setup.py sweep_name=my_sweep stats_server_uri=http://localhost:8000
"""

# NumPy 2.0 compatibility for WandB - must be imported before wandb
import logging
import sys

import hydra
import numpy as np  # noqa: E402

if not hasattr(np, "byte"):
    np.byte = np.int8

import os
from logging import Logger

from omegaconf import DictConfig, ListConfig, OmegaConf

from cogweb.cogweb_client import CogwebClient
from metta.common.util.lock import run_once
from metta.sweep.wandb_utils import create_wandb_sweep

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="sweep_job", version_base=None)
def main(cfg: DictConfig) -> int:
    # Extract sweep base name from CLI sweep_name parameter (e.g., "simple_sweep")
    # Individual training runs will be "simple_sweep.r.0", etc.

    # TODO: Check run_once -- I think it's messing with the CUDA context.
    run_once(lambda: create_sweep(cfg, logger))

    return 0


def create_sweep(cfg: DictConfig | ListConfig, logger: Logger) -> None:
    """
    Create a new sweep with the given name. If the sweep already exists, skip creation.
    Save the sweep configuration to sweep_dir/metadata.yaml.
    """
    # Check if sweep already exists
    backend_url = cfg.sweep_server_uri
    client = CogwebClient(base_url=backend_url)
    wandb_sweep_id = client.sweep_id(cfg.sweep_name)

    # The sweep hasn't been registered with the centralized DB
    if wandb_sweep_id is None:
        # Create the sweep in WandB
        wandb_sweep_id = create_wandb_sweep(cfg.sweep_name, cfg.wandb.entity, cfg.wandb.project)
        # Register the sweep in the centralized DB
        client.create_sweep(cfg.sweep_name, cfg.wandb.project, cfg.wandb.entity, wandb_sweep_id)

    # Save sweep metadata locally
    # in join(cfg.sweep_dir, "metadata.yaml"
    # Creating runs_dir creates the sweep_dir
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


if __name__ == "__main__":
    sys.exit(main())
