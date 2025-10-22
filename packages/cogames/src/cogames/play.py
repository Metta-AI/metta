"""Game playing functionality for CoGames."""

import logging
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from cogames.policy.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig
from mettagrid.policy import PolicySpec
from mettagrid.renderer import RenderMode
from mettagrid.rollout import Rollout

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig


logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_spec: PolicySpec,
    game_name: str,
    max_steps: Optional[int] = None,
    seed: int = 42,
    render_mode: RenderMode = "gui",
) -> None:
    """Play a single game episode with a policy.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_class_path: Path to policy class
        policy_data_path: Optional path to policy weights/checkpoint
        game_name: Human-readable name of the game (used for logging/metadata)
        max_steps: Maximum steps for the episode (None for no limit)
        seed: Random seed
        render: Render mode - "gui", "unicode", or "none"
    """

    logger.debug("Starting play session", extra={"game_name": game_name})

    policy = initialize_or_load_policy(policy_spec.policy_class_path, policy_spec.policy_data_path)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)]
    renderer = make_renderer(render_mode)
    # xcxc max_steps, seed, render
    rollout = Rollout(env_cfg, agent_policies, renderer)
    rollout.run(max_steps)

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {rollout.current_step()}")
    console.print(f"Total Rewards: {rollout.total_rewards()}")
    console.print(f"Final Reward Sum: {float(sum(rollout.total_rewards())):.2f}")
