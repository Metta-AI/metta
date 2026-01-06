"""Interactive play tool for Metta simulations."""

import logging
import uuid
from typing import Optional

import torch
from pydantic import model_validator
from rich.console import Console

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

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

        agent_policies: list[MultiAgentPolicy] = []
        if s3_path:
            policy = initialize_or_load_policy(
                policy_env_info,
                policy_spec_from_uri(s3_path, remove_downloaded_copy_on_exit=True),
            )
            agent_policies.append(policy)
            logger.info("Loaded policy from s3 path")
        elif self.policy_uri:
            logger.info(f"Loading policy from URI: {self.policy_uri}")
            policy = initialize_or_load_policy(
                policy_env_info,
                policy_spec_from_uri(self.policy_uri, device=str(device)),
                device_override=str(device),
            )
            if hasattr(policy, "initialize_to_environment"):
                policy.initialize_to_environment(policy_env_info, device)
            if hasattr(policy, "eval"):
                policy.eval()
            agent_policies.append(policy)
            logger.info("Loaded policy from deprecated-format policy uri")
        else:
            # Fall back to random policies only when no policy was configured explicitly.
            agent_policies.append(RandomMultiAgentPolicy(policy_env_info))

        rollout_result = multi_episode_rollout(
            env_cfg=env_cfg,
            policies=agent_policies,
            episodes=1,
            seed=self.seed,
            render_mode=self.render,
            max_action_time_ms=10000,
        )
        episode = rollout_result.episodes[0]

        # Run the rollout
        logger.info("Starting interactive play session")
        console.print(f"[cyan]Running simulation with {env_cfg.game.num_agents} agents[/cyan]")
        console.print(f"[cyan]Render mode: {self.render}[/cyan]")
        console.print(f"[cyan]Max steps: {env_cfg.game.max_steps}[/cyan]")

        # Print summary
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {episode.steps}")
        console.print(f"Total Rewards: {episode.rewards}")
        console.print(f"Final Reward Sum: {float(episode.rewards.sum()):.2f}")

        return None
