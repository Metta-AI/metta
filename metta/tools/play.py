"""Interactive play tool for Metta simulations."""

import logging
from typing import Optional

import torch
from rich.console import Console

from metta.agent.policy import Policy as MettaPolicy
from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config
from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.random import RandomAgentPolicy
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
    policy_uri: str | None = None
    replay_dir: str | None = None
    stats_dir: str | None = None
    open_browser_on_start: bool = True
    max_steps: Optional[int] = None
    seed: int = 42
    render: RenderMode = "gui"

    @property
    def effective_replay_dir(self) -> str:
        """Return configured replay directory or default under system data_dir."""
        if self.replay_dir is not None:
            return self.replay_dir
        return str(self.system.data_dir / "replays")

    @property
    def effective_stats_dir(self) -> str:
        """Return configured stats directory or default under system data_dir."""
        if self.stats_dir is not None:
            return self.stats_dir
        return str(self.system.data_dir / "stats")

    def _load_policy_from_uri(
        self, policy_uri: str, policy_env_info: PolicyEnvInterface, device: torch.device
    ) -> list[AgentPolicy]:
        """Load a policy from a URI using CheckpointManager and return AgentPolicy instances."""
        logger.info(f"Loading policy from URI: {policy_uri}")

        # Load policy from URI
        policy: MettaPolicy = CheckpointManager.load_from_uri(policy_uri, policy_env_info, device)
        policy.eval()
        policy.initialize_to_environment(policy_env_info, device)

        # Create AgentPolicy instances for each agent
        return [policy.agent_policy(agent_id) for agent_id in range(policy_env_info.num_agents)]

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run an interactive play session with the configured simulation."""

        console = Console()
        device = torch.device("cpu")

        # Get environment config
        env_cfg = self.sim.env

        # Set max_steps in config if specified
        if self.max_steps is not None:
            env_cfg.game.max_steps = self.max_steps

        # Load or create policies
        if self.policy_uri:
            # Create policy environment interface from config
            policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

            agent_policies = self._load_policy_from_uri(self.policy_uri, policy_env_info, device)
        else:
            # Use random policies if no policy specified
            logger.info("No policy specified, using random actions")
            agent_policies = [RandomAgentPolicy(env_cfg.game.actions) for _ in range(env_cfg.game.num_agents)]

        # Create rollout with renderer
        rollout = Rollout(
            env_cfg,
            agent_policies,
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
