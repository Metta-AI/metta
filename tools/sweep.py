import logging
import os
import signal  # Aggressively exit on ctrl+c

import hydra
import wandb
from mettagrid.config.config import setup_metta_environment
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rl.carbs.metta_carbs import carbs_params_from_cfg
from rl.carbs.rollout import CarbsSweepRollout
from rl.wandb.sweep import sweep_id_from_name
from rl.wandb.wandb_context import WandbContext

from wandb_carbs import create_sweep

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Set up colored logging for the current file
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

global _cfg
global _consecutive_failures
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    global _cfg
    _cfg = cfg
    OmegaConf.set_readonly(_cfg, True)

    setup_metta_environment(cfg)

    sweep_id = sweep_id_from_name(cfg.wandb.project, cfg.run)
    if not sweep_id:
        logger.debug(f"Sweep {cfg.run} not found, creating new sweep")
        os.makedirs(os.path.join(cfg.run_dir, "runs"))

        sweep_id = create_sweep(
            cfg.run,
            cfg.wandb.entity,
            cfg.wandb.project,
            carbs_params_from_cfg(cfg)[0]
        )

        logger.debug(f"WanDb Sweep created with ID: {sweep_id}")

    global _consecutive_failures
    _consecutive_failures = 0

    wandb.agent(sweep_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                function=run_carb_sweep_rollout,
                count=999999)

def run_carb_sweep_rollout():
    global _consecutive_failures
    global _cfg

    if _consecutive_failures > 10:
        logger.debug("Too many consecutive failures, exiting")
        os._exit(0)

    success = False
    try:
        with WandbContext(_cfg) as wandb_run:
            rollout = CarbsSweepRollout(_cfg, wandb_run)
            success = rollout.run()
            if success:
                _consecutive_failures = 0
            else:
                _consecutive_failures += 1
    except Exception as e:
        _consecutive_failures += 1
        raise e


if __name__ == "__main__":
    main()
