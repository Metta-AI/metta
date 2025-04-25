import logging
import os
import signal  # Aggressively exit on ctrl+c
import sys

import hydra
from rich.logging import RichHandler

import metta.sim.simulator
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

# Configure rich colored logging
logging.basicConfig(
    level="INFO",
    format="%(processName)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("eval")

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


class PlayJob(Config):
    sim: SimulationConfig
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="play_job")
def play(cfg):
    setup_mettagrid_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        play_job = PlayJob(cfg.play_job)
        metta.sim.simulator.play(play_job.sim, policy_store, play_job.policy_uri)


if __name__ == "__main__":
    sys.exit(play())
