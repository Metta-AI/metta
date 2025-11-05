# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import os
import subprocess
from pathlib import Path

from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config

logger = logging.getLogger(__name__)


class ReplayTool(Tool):
    """Tool for generating and viewing replay files in MettaScope.
    Creates a simulation specifically to generate replay files and automatically
    opens them in the Nim MettaScope application for visualization. This tool focuses
    on replay viewing, unlike EvaluateTool which focuses on policy evaluation."""

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str = "./train_dir/replays"
    stats_dir: str = "./train_dir/stats"
    open_browser_on_start: bool = True
    launch_viewer: bool = True

    def invoke(self, args: dict[str, str]) -> int | None:
        # Create simulation using CheckpointManager integration
        sim = Simulation.create(
            sim_config=self.sim,
            stats_dir=self.stats_dir,
            replay_dir=self.replay_dir,
            policy_uri=self.policy_uri,
        )

        result = sim.simulate()
        # Get all replay URLs (no filtering needed since we just ran this simulation)
        replay_urls = result.stats_db.get_replay_urls()
        if not replay_urls:
            logger.error("No replay URLs found in simulation results", exc_info=True)
            return 1
        replay_url = replay_urls[0]

        if self.launch_viewer:
            launch_mettascope(replay_url)
        else:
            logger.info(
                "Generated replay at %s (MettaScope viewer launch skipped because launch_viewer=False)",
                get_clean_path(replay_url),
            )
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


def launch_mettascope(replay_url: str) -> None:
    """Launch the Nim MettaScope application with the given replay file."""
    if replay_url.startswith("http"):
        logger.error("HTTP replay URLs are not supported with the Nim MettaScope. Use file:// URLs instead.")
        return

    # Get the clean file path
    replay_path = get_clean_path(replay_url)

    # Find the mettascope source directory
    project_root = Path(__file__).resolve().parent.parent.parent
    mettascope_src = project_root / "packages" / "mettagrid" / "nim" / "mettascope" / "src" / "mettascope.nim"

    if not mettascope_src.exists():
        logger.error(f"MettaScope source not found at {mettascope_src}")
        return

    # Launch mettascope with the replay file
    try:
        logger.info(f"Launching MettaScope with replay: {replay_path}")
        cmd = ["nim", "r", str(mettascope_src), "--replay=./" + replay_path]
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch MettaScope: {e}")
    except FileNotFoundError:
        logger.error("Nim compiler not found. Please ensure Nim is installed and in PATH.")
