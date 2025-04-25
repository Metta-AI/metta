import logging
import os
import signal  # Aggressively exit on ctrl+c
import sys
from datetime import datetime

import hydra
from rich.logging import RichHandler

import metta.sim.simulator
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config
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


class PlayJob(Config):
    sim: SimulationConfig
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="play_job")
def play(cfg):
    # Reset logging to use Rich
    setup_rich_logging()
    setup_mettagrid_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        play_job = PlayJob(cfg.play_job)
        metta.sim.simulator.play(play_job.sim, policy_store, play_job.policy_uri)


if __name__ == "__main__":
    sys.exit(play())
