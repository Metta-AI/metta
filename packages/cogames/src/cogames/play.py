"""Game playing functionality for CoGames."""

import logging
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from rich.console import Console

from cogames.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig


logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
    render: Literal["gui", "text", "none"] = "gui",
    verbose: bool = False,
) -> None:
    """Play a single game episode with a policy.

    Args:
        console: Rich console for output
        env_cfg: Game configuration
        policy_class_path: Path to policy class
        policy_data_path: Optional path to policy weights/checkpoint
        max_steps: Maximum steps for the episode (None for no limit)
        seed: Random seed
        render: Render mode - "gui" (default), "text", or "none" (no rendering)
        verbose: Whether to print detailed progress
    """
    # Create environment with render_mode
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render)

    # Initialize policy
    policy = initialize_or_load_policy(policy_class_path, policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]

    # Reset environment to get initial observations
    obs, _ = env.reset(seed=seed)

    # Standard game loop
    step_count = 0
    total_rewards = np.zeros(env.num_agents)

    while max_steps is None or step_count < max_steps:
        # Check if renderer wants to continue (e.g., user quit or interactive loop finished)
        if not env._renderer.should_continue():
            break

        # Render the environment (handles display and user input)
        env.render()

        # Get actions from policies
        actions = np.zeros((env.num_agents, 2), dtype=np.int32)
        for agent_id in range(env.num_agents):
            actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])

        # Step the environment
        obs, rewards, dones, truncated, _ = env.step(actions)

        # Update total rewards
        total_rewards += rewards
        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
    console.print(f"Total Rewards: {total_rewards}")
    if render == "none":
        console.print(f"Final Reward Sum: {float(sum(total_rewards)):.2f}")
