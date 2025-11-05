"""Game playing functionality for CoGames."""

import logging
from typing import TYPE_CHECKING

from rich.console import Console

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.utils import initialize_or_load_policy
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator.rollout import Rollout

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig


logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_spec: PolicySpec,
    game_name: str,
    seed: int = 42,
    render_mode: RenderMode = "gui",
) -> None:
    """Play a single game episode with a policy.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_spec: Policy specification (class path and optional data path)
        game_name: Human-readable name of the game (used for logging/metadata)
        max_steps: Maximum steps for the episode (None for no limit)
        seed: Random seed
        render_mode: Render mode - "gui", "unicode", or "none"
    """

    logger.debug("Starting play session", extra={"game_name": game_name})

    # Check if this is a scripted agent policy that needs simulation
    scripted_agents = [
        "cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
        "cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
    ]
    pass_sim = policy_spec.policy_class_path in scripted_agents

    policy = initialize_or_load_policy(
        policy_spec.policy_class_path, policy_spec.policy_data_path, env_cfg.game.actions
    )

    # Create agent policies (after reset for scripted agents, before for others)
    if pass_sim:
        # For scripted agents, we'll create the agent policies in Rollout after reset
        # Pass the policy object itself, wrapped in a list for each agent
        agent_policies = [policy] * env_cfg.game.num_agents  # Placeholder, will be reset in Rollout
    else:
        agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)]

    # Create simulator and renderer
    rollout = Rollout(env_cfg, agent_policies, render_mode=render_mode, seed=seed, pass_sim_to_policies=pass_sim)
    rollout.run_until_done()

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {rollout._sim.current_step}")
    console.print(f"Total Rewards: {rollout._sim.episode_rewards}")
    console.print(f"Final Reward Sum: {float(sum(rollout._sim.episode_rewards)):.2f}")
