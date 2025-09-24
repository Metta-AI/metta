"""Game playing functionality for CoGames."""

import logging
from typing import Optional

import numpy as np
import torch
from rich.console import Console

from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol

logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: MettaGridConfig,
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
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
        render: Whether to render the game
        save_video: Optional path to save video
        verbose: Whether to print detailed progress
    """
    # Create environment
    env = MettaGridEnv(env_cfg=env_cfg)
    obs, _ = env.reset(seed=seed)

    # Load and create policy
    policy_class = load_symbol(policy_class_path)
    policy = policy_class(env, torch.device("cpu"))

    if policy_data_path and hasattr(policy, "load_checkpoint"):
        policy.load_checkpoint(policy_data_path)

    policy.reset()

    # Run episode
    step_count = 0
    num_agents = env_cfg.game.num_agents

    while max_steps is None or step_count < max_steps:
        # Call policy once per agent to get actions
        actions = []
        for agent_id in range(num_agents):
            # Get action for this specific agent
            agent_obs = obs[agent_id]
            agent_action = policy.step(agent_id, agent_obs)
            actions.append(agent_action)

        # Convert list of actions to numpy array for environment
        actions = np.array(actions)

        obs, rewards, dones, truncated, info = env.step(actions)
        env.render()

        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    # total_reward = sum(rewards)
    # avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
