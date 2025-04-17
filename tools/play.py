import logging
import os
import signal  # Aggressively exit on ctrl+c
import sys
from datetime import datetime

import hydra
from omegaconf import OmegaConf
from rich.logging import RichHandler

from metta.agent.policy_store import PolicyStore
from metta.rl import pufferlib
from metta.util.config import config_from_path
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


# Create a custom formatter that supports milliseconds
class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        created = datetime.fromtimestamp(record.created)
        # Convert microseconds to milliseconds (keep only 3 digits)
        msec = created.microsecond // 1000

        if datefmt:
            # Replace %f with just 3 digits for milliseconds
            datefmt = datefmt.replace("%f", f"{msec:03d}")
        else:
            datefmt = "[%H:%M:%S.%03d]"
            return created.strftime(datefmt) % msec

        return created.strftime(datefmt)


# Create a custom handler that always shows the timestamp
class AlwaysShowTimeRichHandler(RichHandler):
    def emit(self, record):
        # Force a unique timestamp for each record
        record.created = record.created + (record.relativeCreated % 1000) / 1000000
        super().emit(record)


# Configure rich colored logging
handler = AlwaysShowTimeRichHandler(rich_tracebacks=True)
formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
handler.setFormatter(formatter)

logging.basicConfig(level="DEBUG", handlers=[handler])
logger = logging.getLogger(__name__)


# Create a function to reset logging to Rich after Hydra takes over
def setup_rich_logging():
    # Remove all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add back our custom Rich handler
    rich_handler = AlwaysShowTimeRichHandler(rich_tracebacks=True)
    formatter = MillisecondFormatter("%(message)s", datefmt="[%H:%M:%S.%f]")
    rich_handler.setFormatter(formatter)
    root_logger.addHandler(rich_handler)

    # Set the level
    root_logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def play(cfg):
    # Reset logging to use Rich
    setup_rich_logging()

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
        pufferlib.play(cfg, policy_store)


if __name__ == "__main__":
    sys.exit(play())
