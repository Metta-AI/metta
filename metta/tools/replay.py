# Generate a replay file that can be used in MettaScope to visualize a single run.

import logging
import os
import subprocess
from pathlib import Path

import torch
from pydantic import Field

from metta.agent.mocks import MockAgent
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.runner import MultiAgentPolicyInitializer, run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir, auto_wandb_config

logger = logging.getLogger(__name__)


class ReplayTool(Tool):
    """Tool for generating and viewing replay files in MettaScope.
    Creates a simulation specifically to generate replay files and automatically
    opens them in the Nim MettaScope application for visualization. This tool focuses
    on replay viewing, unlike EvaluateTool which focuses on policy evaluation."""

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig
    policy_uri: str | None = None
    replay_dir: str = Field(default_factory=auto_replay_dir)
    open_browser_on_start: bool = True
    launch_viewer: bool = True

    def _build_policy_initializer(self, normalized_uri: str | None) -> MultiAgentPolicyInitializer:
        device = torch.device("cpu")

        def _initializer(policy_env_info):
            if normalized_uri is None:
                policy = MockAgent(policy_env_info)
            else:
                artifact = CheckpointManager.load_artifact_from_uri(normalized_uri)
                policy = artifact.instantiate(policy_env_info, device=device)

            policy = policy.to(device)
            policy.eval()
            return policy

        return _initializer

    def invoke(self, args: dict[str, str]) -> int | None:
        normalized_uri = CheckpointManager.normalize_uri(self.policy_uri) if self.policy_uri else None
        policy_initializers = [self._build_policy_initializer(normalized_uri)]

        simulation_run = self.sim.to_simulation_run_config()

        simulation_results = run_simulations(
            policy_initializers=policy_initializers,
            simulations=[simulation_run],
            replay_dir=self.replay_dir,
            seed=self.system.seed,
            enable_replays=True,
        )

        result = simulation_results[0]
        replay_urls = result.replay_urls
        if not replay_urls:
            logger.error("No replay URLs found in simulation results", exc_info=True)
            return 1

        replay_url = next(iter(replay_urls.values()))

        if self.launch_viewer:
            launch_mettascope(replay_url)
        else:
            # For CI/non-visualization, just confirm replays were generated
            logger.info("Replay generation completed (viewer launch skipped)")

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
        cmd = ["nim", "r", "-d:fidgetUseFigma", str(mettascope_src), "--replay=./" + replay_path]
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch MettaScope: {e}")
    except FileNotFoundError:
        logger.error("Nim compiler not found. Please ensure Nim is installed and in PATH.")
