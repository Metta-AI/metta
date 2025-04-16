import logging
import os
import signal  # Aggressively exit on ctrl+c

import hydra
from omegaconf import OmegaConf
from rich.logging import RichHandler

from metta.agent.policy_store import PolicyStore
from metta.rl.pufferlib.play import play
from metta.util.config import config_from_path
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Configure rich colored logging
logging.basicConfig(
    level="DEBUG", format="%(processName)s %(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg):
    # Safely log configuration details
    try:
        logger.debug(f"Configuration type: {type(cfg)}")

        if hasattr(cfg, "run"):
            # Handle different types of run attribute
            if isinstance(cfg.run, str):
                logger.info(f"Run configuration: {cfg.run}")
            elif hasattr(cfg.run, "__dict__"):
                logger.info(f"Run configuration: {cfg.run.__dict__}")
            else:
                logger.info(f"Run configuration type: {type(cfg.run)}")
        else:
            logger.info("No 'run' attribute in configuration")

        if hasattr(cfg, "eval"):
            if hasattr(cfg.eval, "env"):
                logger.info(f"Environment configuration: {cfg.eval.env}")
            else:
                logger.info("No 'env' attribute in eval configuration")
        else:
            logger.info("No 'eval' attribute in configuration")
    except Exception as e:
        logger.error(f"Error during configuration logging: {e}")

    setup_mettagrid_environment(cfg)

    # Add fallback if environment is not configured
    if not hasattr(cfg.eval, "env") or cfg.eval.env is None:
        default_env = "env/mettagrid-base/simple"
        logger.warning(f"No eval.env specified in config, using default {default_env}")
        cfg.eval.env = default_env

    # check for a policy_uri
    if not hasattr(cfg, "policy_uri") or cfg.policy_uri is None:
        default_policy_uri = "wandb://run/b.daveey.t.8.rdr9.3"
        logger.warning(f"No policy_uri specified in config, using default {default_policy_uri}")
        # Create new config
        raw_cfg_data = OmegaConf.to_container(cfg, resolve=True)
        raw_cfg_data["policy_uri"] = default_policy_uri
        cfg = OmegaConf.create(raw_cfg_data)

    cfg.eval.env = config_from_path(cfg.eval.env, cfg.eval.env_overrides)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        play(cfg, policy_store)


if __name__ == "__main__":
    main()
