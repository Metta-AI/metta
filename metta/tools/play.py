#!/usr/bin/env -S uv run
# Starts a websocket server that allows you to play as a metta agent.

import logging

import mettascope.server as server
from metta.agent.policy_store import PolicyStore
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.util.tool_config import Tool
from metta.common.wandb.wandb_context import WandbConfig, WandbConfigOff
from metta.rl.system_config import SystemConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)


class PlayTool(Tool):
    system: SystemConfig = SystemConfig.Auto()
    wandb: WandbConfig = WandbConfigOff()
    sim: SimulationConfig
    policy_uri: str | None = None
    selector_type: str = "latest"
    # TODO #dehydration - these should be relative to system.data_dir
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True

    def invoke(self) -> None:
        ws_url = "%2Fws"

        if self.open_browser_on_start:
            server.run(self, open_url=f"?wsUrl={ws_url}")
        else:
            logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
            server.run(self)


def create_simulation(cfg: PlayTool) -> Simulation:
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
