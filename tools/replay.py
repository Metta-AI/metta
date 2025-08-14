#!/usr/bin/env -S uv run
# Generate a replay file that can be used in MettaScope to visualize a single run.

import argparse
import importlib
import logging
import platform
from urllib.parse import quote

import mettascope.server as server
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.logging_helpers import init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff
from metta.rl.system_config import SystemConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig

logger = logging.getLogger("tools.replay")


# TODO: This job can be replaced with sim now that Simulations create replays
class ReplayToolConfig(Config):
    system: SystemConfig = SystemConfig()
    wandb: WandbConfig = WandbConfigOff()
    sim: SimulationConfig
    policy_uri: str | None = None
    selector_type: str = "latest"
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True


def replay(cfg: ReplayToolConfig) -> None:
    logger.info(f"tools.replay job config:\n{cfg.model_dump_json(indent=2)}")

    # Create policy store directly without WandbContext
    policy_store = PolicyStore.create(
        device=cfg.system.device,
        wandb_config=cfg.wandb,
        replay_dir=cfg.replay_dir,
    )

    # Create simulation using the helper method with explicit parameters
    sim = Simulation.create(
        sim_config=cfg.sim,
        policy_store=policy_store,
        device=cfg.system.device,
        vectorization=cfg.system.vectorization,
        stats_dir=cfg.stats_dir,
        replay_dir=cfg.replay_dir,
        policy_uri=cfg.policy_uri,
        run_name="replay_run",
    )

    result = sim.simulate()
    key, version = result.stats_db.key_and_version(sim.policy_record)
    replay_url = result.stats_db.get_replay_urls(key, version)[0]

    open_browser(replay_url, cfg)


def open_browser(replay_url: str, cfg: ReplayToolConfig) -> None:
    # Only on macos open a browser to the replay
    if platform.system() == "Darwin":
        if not replay_url.startswith("http"):
            # Remove ./ prefix if it exists
            clean_path = replay_url.removeprefix("./")
            local_url = f"{DEV_METTASCOPE_FRONTEND_URL}/local/{clean_path}"
            full_url = f"/?replayUrl={quote(local_url)}"

            # Run a metascope server that serves the replay
            # Import PlayToolConfig to use with the server
            from tools.play import PlayToolConfig

            # Create a PlayToolConfig from ReplayToolConfig (they have the same fields)
            play_cfg = PlayToolConfig(
                system=cfg.system,
                wandb=cfg.wandb,
                sim=cfg.sim,
                policy_uri=cfg.policy_uri,
                selector_type=cfg.selector_type,
                replay_dir=cfg.replay_dir,
                stats_dir=cfg.stats_dir,
                open_browser_on_start=cfg.open_browser_on_start,
            )

            if cfg.open_browser_on_start:
                server.run(play_cfg, open_url=full_url)
            else:
                logger.info(f"Enter MettaGrid @ {full_url}")
                server.run(play_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=False)
    args = parser.parse_args()

    init_logging()

    if args.cfg:
        with open(args.cfg, "r") as f:
            cfg = ReplayToolConfig.model_validate_json(f.read())
    else:
        module_name, func_name = args.func.rsplit(".", 1)
        cfg = importlib.import_module(module_name).__getattribute__(func_name)()

    replay(cfg)
