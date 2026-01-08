# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import Field

from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.sim.runner import run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_wandb_config
from mettagrid.policy.policy import PolicySpec
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

logger = logging.getLogger(__name__)


class ReplayTool(Tool):
    """Tool for generating and viewing replay files in MettaScope.
    Creates a simulation specifically to generate replay files and automatically
    opens them in the Nim MettaScope application for visualization. This tool focuses
    on replay viewing, unlike EvaluateTool which focuses on policy evaluation."""

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: Optional[str] = None
    replay_dir: str = Field(default_factory=auto_replay_dir)
    open_browser_on_start: bool = True
    launch_viewer: bool = True

    def _build_policy_spec(self, normalized_uri: Optional[str]) -> PolicySpec:
        if normalized_uri is None:
            return PolicySpec(class_path="metta.agent.mocks.mock_agent.MockAgent", data_path=None)
        return policy_spec_from_uri(normalized_uri, device="cpu")

    def invoke(self, args: dict[str, str]) -> Optional[int]:
        policy_spec = self._build_policy_spec(self.policy_uri)

        simulation_run = self.sim.to_simulation_run_config()

        simulation_results = run_simulations(
            policy_specs=[policy_spec],
            simulations=[simulation_run],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
        )

        result = simulation_results[0]
        replay_url = result.results.episodes[0].replay_path
        if not replay_url:
            logger.error("No replay path found in simulation results", exc_info=True)
            return 1

        if self.launch_viewer:
            launch_mettascope(replay_url)
        else:
            # For CI/non-visualization, just confirm replays were generated
            logger.info("Replay generation completed (viewer launch skipped)")

        return 0


def get_clean_path(replay_url: str) -> str:
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
        cmd = ["nim", "r", "-d:fidgetUseFigma", str(mettascope_src), "--replay=./" + replay_path]
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch MettaScope: {e}")
    except FileNotFoundError:
        logger.error("Nim compiler not found. Please ensure Nim is installed and in PATH.")
