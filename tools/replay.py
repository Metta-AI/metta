#!/usr/bin/env -S uv run
# Generate a replay file that can be used in MettaScope to visualize a single run.

import os
import platform
import webbrowser

import hydra
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext
from mettagrid.util.file import http_url


# TODO: This job can be replaced with sim now that Simulations create replays
class ReplayJob(Config):
    __init__ = Config.__init__
    sim: SingleEnvSimulationConfig
    policy_uri: str
    selector_type: str
    replay_dir: str
    stats_dir: str


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("metta.tools.replay")
    logger.info(f"Replay job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        replay_job = ReplayJob(cfg.replay_job)
        ma = policy_store.policy(cfg.replay_job.policy_uri)
        logger.info(f"Generating replays for policy: {ma.name}")

        if not os.path.exists(cfg.replay_job.replay_dir):
            os.makedirs(cfg.replay_job.replay_dir, exist_ok=True)
        if not os.path.exists(cfg.replay_job.stats_dir):
            os.makedirs(cfg.replay_job.stats_dir, exist_ok=True)

        sim_config = SingleEnvSimulationConfig(cfg.replay_job.sim)

        sim = Simulation(
            name=f"replay_{ma.name.replace('/', '_')}",
            config=sim_config,
            policy_ma=ma,
            policy_store=policy_store,
            device=cfg.device,
            vectorization=cfg.vectorization,
            replay_dir=cfg.replay_job.replay_dir,
            stats_dir=cfg.replay_job.stats_dir,
        )
        results = sim.simulate()

        replay_urls = results.stats_db.get_replay_urls(policy_key=ma.key(), policy_version=ma.version())
        if len(replay_urls) > 0:
            logger.info("Replay URLs:")
            for url in replay_urls:
                logger.info(f"  {url}")

        # Only on macos open a browser to the replay
        if platform.system() == "Darwin":
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={http_url(replay_urls[0])}")


if __name__ == "__main__":
    main()
