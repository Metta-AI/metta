"""Interactive play tool for Metta simulations."""

import logging
from typing import Optional

import numpy as np
from rich.console import Console

from metta.common.tool import Tool
from metta.common.wandb.context import WandbConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_wandb_config
from mettagrid.policy.policy import AgentPolicy
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator import AgentObservation
from mettagrid.simulator.rollout import Rollout

logger = logging.getLogger(__name__)


class RandomPolicyAdapter(AgentPolicy):
    """Random policy adapter for use with Rollout."""

    def __init__(self, num_actions: int, actions_config):
        super().__init__(actions_config)
        self._num_actions = num_actions

    def step(self, obs: AgentObservation):
        """Return random action."""
        return np.random.randint(0, self._num_actions)

    def reset(self) -> None:
        """No state to reset."""
        pass


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

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run an interactive play session with the configured simulation."""

        console = Console()

        # Get environment config
        env_cfg = self.sim.env

        # Set max_steps in config if specified
        if self.max_steps is not None:
            env_cfg.game.max_steps = self.max_steps

        # Count enabled actions
        actions_dict = env_cfg.game.actions.model_dump()
        num_actions = sum(1 for action_cfg in actions_dict.values() if action_cfg.get("enabled", True))

        # For now, only support random policies
        # TODO: Add support for loading trained policies via policy_uri
        if self.policy_uri:
            logger.warning(f"Policy loading from {self.policy_uri} is not yet implemented. Using random actions.")

        logger.info("Using random actions for all agents")

        # Create random policies for all agents
        agent_policies: list[AgentPolicy] = [
            RandomPolicyAdapter(num_actions, env_cfg.game.actions) for _ in range(env_cfg.game.num_agents)
        ]

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
