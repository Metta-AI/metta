#!/usr/bin/env -S uv run
# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import platform
from urllib.parse import quote

from omegaconf import DictConfig, OmegaConf

import mettascope.server as server
from metta.agent.mocks import MockPolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.wandb.wandb_context import WandbContext
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.metta_script import metta_script


# TODO: This job can be replaced with sim now that Simulations create replays
class ReplayJob(Config):
    __init__ = Config.__init__
    sim: SingleEnvSimulationConfig
    policy_uri: str | None
    selector_type: str
    replay_dir: str
    stats_dir: str
    open_browser_on_start: bool


def main(cfg: DictConfig):
    logger = logging.getLogger("tools.replay")

    logger.info(f"tools.replay job config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        replay_job = ReplayJob(cfg.replay_job)
        if replay_job.policy_uri is not None:
            policy_record = policy_store.policy_record(replay_job.policy_uri)
        else:
            policy_record = MockPolicyRecord(run_name="replay_run", uri=None)

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
        key, version = result.stats_db.key_and_version(policy_record)
        replay_url = result.stats_db.get_replay_urls(key, version)[0]

        # Only on macos open a browser to the replay
        if platform.system() == "Darwin":
            if not replay_url.startswith("http"):
                # Remove ./ prefix if it exists
                clean_path = replay_url.removeprefix("./")
                local_url = f"http://localhost:8000/local/{clean_path}"
                full_url = f"/?replayUrl={quote(local_url)}"

                # Run a metascope server that serves the replay
                open_browser = OmegaConf.select(cfg, "replay_job.open_browser_on_start", default=True)

                if open_browser:
                    server.run(cfg, open_url=full_url)
                else:
                    logger.info(f"Enter MettaGrid @ {full_url}")
                    server.run(cfg)


metta_script(main, "replay_job")
