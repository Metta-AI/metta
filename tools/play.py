#!/usr/bin/env -S uv run
# Starts a websocket server that allows you to play as a metta agent.

import argparse
import importlib
import logging

import mettascope.server as server
from metta.agent.policy_store import PolicyStore
from metta.common.util.config import Config
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.logging_helpers import init_logging
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff
from metta.rl.system_config import SystemConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig

logger = logging.getLogger("tools.play")


class PlayToolConfig(Config):
    system: SystemConfig = SystemConfig()
    wandb: WandbConfig = WandbConfigOff()
    sim: SingleEnvSimulationConfig
    policy_uri: str | None = None
    selector_type: str = "latest"
    # TODO #dehydration - these should be relative to system.data_dir
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True


def create_simulation(cfg: PlayToolConfig) -> Simulation:
    """Create a simulation for playing/replaying."""
    logger.info(f"Creating simulation: {cfg.sim.name}")

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
        run_name="play_run",
    )
    return sim


def play(cfg: PlayToolConfig) -> None:
    logger.info(f"tools.play job config:\n{cfg.model_dump_json(indent=2)}")

    ws_url = "%2Fws"

    if cfg.open_browser_on_start:
        server.run(cfg, open_url=f"?wsUrl={ws_url}")
    else:
        logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
        server.run(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=False)
    args = parser.parse_args()

    init_logging()

    if args.cfg:
        with open(args.cfg, "r") as f:
            cfg = PlayToolConfig.model_validate_json(f.read())
    else:
        module_name, func_name = args.func.rsplit(".", 1)
        cfg = importlib.import_module(module_name).__getattribute__(func_name)()

    play(cfg)
