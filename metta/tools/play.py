"""Interactive play tool for Metta simulations."""

import logging
import uuid
from contextlib import ExitStack
from typing import Optional

import torch
from pydantic import model_validator
from rich.console import Console

from metta.agent.policy import Policy as MettaPolicy
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.s3_policy_spec_loader import policy_spec_from_s3_submission
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.rollout import Rollout

logger = logging.getLogger(__name__)


class PlayTool(Tool):
    """Interactive play tool for Metta simulations using Rollout.

    This tool creates an interactive play session where agents act using either
    a specified policy or random actions. The simulation is rendered according
    to the specified render mode (gui, unicode, log, or none).
    """

    wandb: WandbConfig = auto_wandb_config()
    sim: SimulationConfig

    policy_uri: str | None = None  # Deprecated, use policy_version_id or s3_path instead
    s3_path: str | None = None
    policy_version_id: str | None = None

    open_browser_on_start: bool = True
    max_steps: Optional[int] = None
    seed: int = 42
    render: RenderMode = "gui"
    stats_server_uri: str | None = auto_stats_server_uri()

    def _load_policy_from_uri(
        self, policy_uri: str, policy_env_info: PolicyEnvInterface, device: torch.device
    ) -> MultiAgentPolicy:
        """Load a policy from a URI using CheckpointManager."""
        logger.info(f"Loading policy from URI: {policy_uri}")

        policy_spec = CheckpointManager.policy_spec_from_uri(policy_uri, device=str(device))
        policy: MettaPolicy = initialize_or_load_policy(policy_env_info, policy_spec)
        if hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, device)
        policy.eval()
        return policy

    @model_validator(mode="after")
    def validate(self) -> "PlayTool":
        if len([x for x in [self.policy_uri, self.policy_version_id, self.s3_path] if x is not None]) > 1:
            raise ValueError("Only one of policy_uri, policy_version_id, or s3_path can be specified")
        return self

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run an interactive play session with the configured simulation."""

        console = Console()
        device = torch.device("cpu")

        # Get environment config
        env_cfg = self.sim.env
        policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

        # Set max_steps in config if specified
        if self.max_steps is not None:
            env_cfg.game.max_steps = self.max_steps

        s3_path: str | None = self.s3_path
        if self.policy_version_id:
            if not self.stats_server_uri:
                raise ValueError("stats_server_uri is required")
            if s3_path:
                raise ValueError("s3_path and policy_version_id cannot be specified together")
            stats_client = StatsClient.create(self.stats_server_uri)
            policy_version = stats_client.get_policy_version(uuid.UUID(self.policy_version_id))
            s3_path = policy_version.s3_path
            if not s3_path:
                raise ValueError(f"Policy version {self.policy_version_id} has no s3 path")

        with ExitStack() as stack:
            controllers: list[tuple[MultiAgentPolicy, int]] = []

            def _register_policy(policy: MultiAgentPolicy) -> None:
                controllers.extend((policy, agent_id) for agent_id in range(policy_env_info.num_agents))

            if s3_path:
                assert s3_path is not None
                policy_spec = stack.enter_context(policy_spec_from_s3_submission(s3_path))
                multi_agent_policy = initialize_or_load_policy(policy_env_info, policy_spec)
                _register_policy(multi_agent_policy)
                logger.info("Loaded policy from s3 path")
            elif self.policy_uri:
                policy = self._load_policy_from_uri(self.policy_uri, policy_env_info, device)
                _register_policy(policy)
                logger.info("Loaded policy from deprecated-format policy uri")
            else:
                # Fall back to random policies only when no policy was configured explicitly.
                random_policy = RandomMultiAgentPolicy(policy_env_info)
                _register_policy(random_policy)

            # Create rollout with renderer
            rollout = Rollout(
                env_cfg,
                controllers,
                max_action_time_ms=10000,
                render_mode=self.render,
                seed=self.seed,
            )

        # Run the rollout
        logger.info("Starting interactive play session")
        console.print(f"[cyan]Running simulation with {env_cfg.game.num_agents} agents[/cyan]")
        console.print(f"[cyan]Render mode: {self.render}[/cyan]")
        console.print(f"[cyan]Max steps: {env_cfg.game.max_steps}[/cyan]")

        rollout.run_until_done()

        # Print summary
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {rollout._sim.current_step}")
        console.print(f"Total Rewards: {rollout._sim.episode_rewards}")
        console.print(f"Final Reward Sum: {float(sum(rollout._sim.episode_rewards)):.2f}")

        return None
