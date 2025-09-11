# Starts a websocket server that allows you to play as a metta agent.

import logging

import mettascope.server as server
from metta.common.tool import Tool
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.wandb.wandb_context import WandbConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class PlayTool(Tool):
    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str | None = None
    stats_dir: str | None = None
    open_browser_on_start: bool = True

    @property
    def effective_replay_dir(self) -> str:
        """Get the replay directory, defaulting to system.data_dir/replays if not specified."""
        return self.replay_dir if self.replay_dir is not None else f"{self.system.data_dir}/replays"

    @property
    def effective_stats_dir(self) -> str:
        """Get the stats directory, defaulting to system.data_dir/stats if not specified."""
        return self.stats_dir if self.stats_dir is not None else f"{self.system.data_dir}/stats"

    def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
        ws_url = "%2Fws"

        if self.open_browser_on_start:
            server.run(self, open_url=f"?wsUrl={ws_url}")
        else:
            logger.info(f"Enter MettaGrid @ {DEV_METTASCOPE_FRONTEND_URL}?wsUrl={ws_url}")
            server.run(self)


def create_simulation(cfg: PlayTool) -> Simulation:
    """Create a simulation for playing/replaying."""
    logger.info(f"Creating simulation: {cfg.sim.name}")

    # Create simulation using CheckpointManager integration
    sim = Simulation.create(
        sim_config=cfg.sim,
        device=cfg.system.device,
        vectorization=cfg.system.vectorization,
        stats_dir=cfg.effective_stats_dir,
        replay_dir=cfg.effective_replay_dir,
        policy_uri=cfg.policy_uri,
    )
    return sim
