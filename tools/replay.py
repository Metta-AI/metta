#!/usr/bin/env -S uv run
# Generate a replay file that can be used in MettaScope to visualize a single run.

import platform
from urllib.parse import quote

import hydra
from omegaconf import OmegaConf

import mettascope.server as server
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.common.util.wandb.wandb_context import WandbContext
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig


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
            if not replay_url.startswith("http"):
                # Remove ./ prefix if it exists
                clean_path = replay_url.removeprefix("./")
                local_url = f"http://localhost:8000/local/{clean_path}"
                full_url = f"/?replayUrl={quote(local_url)}"

                # Run a metascope server that serves the replay
                server.run(cfg, open_url=full_url)


if __name__ == "__main__":
    main()
