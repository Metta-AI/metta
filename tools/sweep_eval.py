#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
from typing import List, Optional
import hydra
from util.runtime_configuration import setup_omega_conf
import wandb
from omegaconf import OmegaConf, DictConfig
from rich.logging import RichHandler
import yaml

from rl.wandb.sweep import generate_run_id_for_sweep, sweep_id_from_name
from rl.carbs.metta_carbs import carbs_params_from_cfg
import wandb_carbs
import json
from rl.carbs.metta_carbs import MettaCarbs

from rl.wandb.wandb_context import WandbContext

# Configure rich colored logging to stderr instead of stdout
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("sweep_eval")

@hydra.main(config_path="../configs", config_name="sweep", version_base=None)
def main(cfg: OmegaConf) -> int:
    setup_omega_conf()


if __name__ == "__main__":
    sys.exit(main())
