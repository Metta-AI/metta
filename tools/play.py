import os
import signal  # Aggressively exit on ctrl+c
import sys

import hydra
from omegaconf import OmegaConf

import metta.rl.pufferlib.play
from metta.agent.policy_store import PolicyStore
from metta.util.config import config_from_path
from metta.util.logging import rich_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def play(cfg):
    logger = rich_logger(__name__)

    # Validate configuration details
    configuration_is_valid = True
    try:
        logger.debug(f"Configuration type: {type(cfg)}")

        if hasattr(cfg, "run"):
            # Handle different types of run attribute
            if isinstance(cfg.run, str):
                logger.info(f"cfg.run: {cfg.run}")
            elif hasattr(cfg.run, "__dict__"):
                logger.info(f"cfg.run: {cfg.run.__dict__}")
            else:
                logger.error(f"cfg.run type: {type(cfg.run)}")
                configuration_is_valid = False
        else:
            logger.error("No 'run' attribute in cfg")
            configuration_is_valid = False

        if hasattr(cfg, "eval"):
            if hasattr(cfg.eval, "env"):
                logger.info(f"cfg.eval.env: {cfg.eval.env}")
            else:
                logger.error("No 'env' attribute in cfg.eval")
                configuration_is_valid = False
        else:
            logger.error("No 'eval' attribute in cfg")
            configuration_is_valid = False

        if hasattr(cfg, "policy_uri"):
            logger.info(f"cfg.policy_uri: {cfg.eval.env}")
        else:
            logger.error("No 'policy_uri' attribute in cfg")
            configuration_is_valid = False

    except Exception as e:
        logger.error(f"Error during configuration logging: {e}")

    if not configuration_is_valid:
        logger.info("Configuration details:")
        yaml_str = OmegaConf.to_yaml(cfg)
        for line in yaml_str.split("\n"):
            logger.info(line)
        exit()

    setup_mettagrid_environment(cfg)

    cfg.eval.env = config_from_path(cfg.eval.env, cfg.eval.env_overrides)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        metta.rl.pufferlib.play.play(cfg, policy_store)


if __name__ == "__main__":
    sys.exit(play())
