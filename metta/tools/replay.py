# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import os
import platform
from typing import ClassVar
from urllib.parse import quote

import mettascope.server as server
from metta.common.tool import Tool
from metta.common.util.constants import DEV_METTASCOPE_FRONTEND_URL
from metta.common.wandb.context import WandbConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class ReplayTool(Tool):
    tool_name: ClassVar[str] = "replay"
    """Tool for generating and viewing replay files in MettaScope.
    Creates a simulation specifically to generate replay files and automatically
    opens them in a browser for visualization. This tool focuses on replay viewing
    and browser integration, unlike EvalTool which focuses on policy evaluation."""

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True

    def invoke(self, args: dict[str, str]) -> int | None:
        # Create simulation using CheckpointManager integration
        sim = Simulation.create(
            sim_config=self.sim,
            device=self.system.device,
            vectorization=self.system.vectorization,
            stats_dir=self.stats_dir,
            replay_dir=self.replay_dir,
            policy_uri=self.policy_uri,
        )

        result = sim.simulate()
        # Get all replay URLs (no filtering needed since we just ran this simulation)
        replay_urls = result.stats_db.get_replay_urls()
        if not replay_urls:
            logger.error("No replay URLs found in simulation results")
            return 1
        replay_url = replay_urls[0]

        open_browser(replay_url, self)
        return 0


def get_clean_path(replay_url: str) -> str:
    path = replay_url
    if replay_url.startswith("file://"):
        path = replay_url.removeprefix("file://")

    if path.startswith("./"):
        return path.removeprefix("./")
    else:
        # If the path url is fully qualified, we want to remove the cwd prefix
        current_dir = os.getcwd()
        return path.removeprefix(current_dir)


def open_browser(replay_url: str, cfg: ReplayTool) -> None:
    # Only on macos open a browser to the replay
    if platform.system() == "Darwin":
        if not replay_url.startswith("http"):
            # Remove ./ prefix if it exists
            clean_path = get_clean_path(replay_url)
            local_url = f"{DEV_METTASCOPE_FRONTEND_URL}/local/{clean_path}"
            full_url = f"/?replayUrl={quote(local_url)}"

            # Run a metascope server that serves the replay
            # Create a PlayTool from ReplayTool (they have the same fields)
            play_cfg = PlayTool(
                system=cfg.system,
                wandb=cfg.wandb,
                sim=cfg.sim,
                policy_uri=cfg.policy_uri,
                replay_dir=cfg.replay_dir,
                stats_dir=cfg.stats_dir,
                open_browser_on_start=cfg.open_browser_on_start,
            )

            if cfg.open_browser_on_start:
                server.run(play_cfg, open_url=full_url)
            else:
                logger.info(f"Enter MettaGrid @ {full_url}")
                server.run(play_cfg)
