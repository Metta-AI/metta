#!/usr/bin/env -S uv run
# Generate a replay file that can be used in MettaScope to visualize a single run.

import platform
import webbrowser

import hydra
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.mettagrid.util.file import http_url
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


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
        policy_record = policy_store.policy(replay_job.policy_uri)
        sim_config = SingleEnvSimulationConfig(cfg.replay_job.sim)

        sim_name = sim_config.env.split("/")[-1]
        replay_dir = f"{replay_job.replay_dir}/{sim_name}"

        sim = Simulation(
            sim_name,
            sim_config,
            policy_record,
            policy_store,
            device=cfg.device,
            vectorization=cfg.vectorization,
            stats_dir=replay_job.stats_dir,
            replay_dir=replay_dir,
        )
        result = sim.simulate()
        replay_url = result.stats_db.get_replay_urls(
            policy_key=policy_record.key(), policy_version=policy_record.version()
        )[0]

        # Only on macos open a browser to the replay
        if platform.system() == "Darwin":
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={http_url(replay_url)}")


if __name__ == "__main__":
    main()
