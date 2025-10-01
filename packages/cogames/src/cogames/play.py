"""Game playing functionality for CoGames."""

import json
import logging
from typing import Literal, Optional

import numpy as np
import torch
from rich.console import Console

import mettagrid.mettascope as mettascope
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.grid_object_formatter import format_grid_object
from mettagrid.util.module import load_symbol

logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: MettaGridConfig,
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
    render: Literal["gui", "text"] = "gui",
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
        render: Render mode - "gui" (default) or "text"
        verbose: Whether to print detailed progress
    """
    # Create environment with appropriate render mode
    render_mode = None if render == "gui" else "miniscope" if render == "text" else None
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render_mode)

    # Load and create policy
    policy_class = load_symbol(policy_class_path)
    device = torch.device("cpu")

    # Instantiate the policy
    policy_instance = policy_class(env, device)

    # Load checkpoint if provided
    if policy_data_path:
        policy_instance.load_policy_data(policy_data_path)

    # Create per-agent policies
    agent_policies = []
    for agent_id in range(env.num_agents):
        agent_policies.append(policy_instance.agent_policy(agent_id))

    # For text mode, use the interactive loop in miniscope
    if render == "text" and hasattr(env, "_renderer") and env._renderer:

        def get_actions_fn(
            obs: np.ndarray, selected_agent: Optional[int], manual_action: Optional[int | tuple]
        ) -> np.ndarray:
            """Get actions for all agents, with optional manual override."""
            actions = np.zeros((env.num_agents, 2), dtype=np.int32)
            for agent_id in range(env.num_agents):
                if agent_id == selected_agent and manual_action is not None:
                    # manual_action can be a tuple of (action_id, arg) or just a direction for move
                    if isinstance(manual_action, tuple):
                        actions[agent_id] = list(manual_action)
                    else:
                        # Get move action ID from environment
                        move_action_id = env.action_names.index("move") if "move" in env.action_names else 0
                        actions[agent_id] = [move_action_id, manual_action]
                else:
                    actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])
            return actions

        result = env._renderer.interactive_loop(env, get_actions_fn, max_steps=max_steps)
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {result['steps']}")
        console.print(f"Total Rewards: {result['total_rewards']}")
        return

    # GUI mode: use mettascope
    obs, _ = env.reset(seed=seed)
    step_count = 0
    num_agents = env_cfg.game.num_agents
    actions = np.zeros((env.num_agents, 2), dtype=np.int32)
    total_rewards = np.zeros(env.num_agents)

    # Initialize GUI replay
    initial_replay = {
        "version": 2,
        "action_names": env.action_names,
        "item_names": env.resource_names,
        "type_names": env.object_type_names,
        "map_size": [env.map_width, env.map_height],
        "num_agents": env.num_agents,
        "max_steps": 0,
        "mg_config": env.mg_config.model_dump(mode="json"),
        "objects": [],
    }

    response = mettascope.init(replay=json.dumps(initial_replay))
    if response.should_close:
        return

    def generate_replay_step():
        grid_objects = []
        for grid_object in env.grid_objects().values():
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                total_rewards[agent_id] += env.rewards[agent_id]
            grid_objects.append(
                format_grid_object(grid_object, actions, env.action_success, env.rewards, total_rewards)
            )
        step_replay = {"step": step_count, "objects": grid_objects}
        return json.dumps(step_replay)

    # GUI rendering loop
    while max_steps is None or step_count < max_steps:
        # Get actions from policies
        for agent_id in range(num_agents):
            actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])

        # Render and get user input
        replay_step = generate_replay_step()
        response = mettascope.render(step_count, replay_step)
        if response.should_close:
            break
        if response.action:
            actions[response.action_agent_id, 0] = response.action_action_id
            actions[response.action_agent_id, 1] = response.action_argument

        obs, rewards, dones, truncated, info = env.step(actions)

        # Update total rewards
        for agent_id in range(num_agents):
            total_rewards[agent_id] += rewards[agent_id]

        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
