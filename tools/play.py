import os
import signal  # Aggressively exit on ctrl+c
import sys

import hydra

import metta.sim.simulator
from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


class PlayJob(Config):
    sim: SimulationSuiteConfig
    policy_uri: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="play_job")
def main(cfg) -> int:
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("play")
    logger.info(f"Playing {cfg.run}")

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        play_job = PlayJob(cfg.play_job)
        policy_record = policy_store.policy(play_job.policy_uri)
        metta.sim.simulator.play(list(play_job.sim.simulations.values())[0], policy_record)

    return 0


if __name__ == "__main__":
    sys.exit(main())
