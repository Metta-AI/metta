"""Game playing functionality for CoGames."""

import json
import logging
from typing import Optional

import numpy as np
import torch
from rich.console import Console

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
    try:
        import mettagrid.mettascope as mettascope_module
    except ImportError as err:  # pragma: no cover - renderer optional
        console.print("[red]Renderer dependencies are missing (mettascope2 bindings not found).")
        console.print(
            "[dim]Generate the Nim bindings or install the optional renderer package to use interactive play.[/dim]"
        )
        logger.debug("mettascope import failed", exc_info=err)
        return

    # Create environment
    env = MettaGridEnv(env_cfg=env_cfg)
    obs, _ = env.reset(seed=seed)

    # Load and create policy
    policy_class = load_symbol(policy_class_path)
    device = torch.device("cpu")

    # Check if this is a TrainablePolicy or a simple Policy
    from cogames.policy import Policy, TrainablePolicy

    # Instantiate the policy
    policy_instance = policy_class(env, device)

    # Create per-agent policies
    agent_policies = []
    if isinstance(policy_instance, TrainablePolicy):
        # Load checkpoint if provided
        if policy_data_path:
            policy_instance.load_policy_data(policy_data_path)
        # Create per-agent policies from the trainable policy
        for agent_id in range(env.num_agents):
            agent_policies.append(policy_instance.agent_policy(agent_id))
    elif isinstance(policy_instance, Policy):
        for agent_id in range(env.num_agents):
            agent_policies.append(policy_instance.agent_policy(agent_id))
    else:
        raise ValueError("Policy class must implement either Policy or TrainablePolicy interface")

    # Reset all agent policies
    for policy in agent_policies:
        policy.reset()

    # Run episode
    step_count = 0
    num_agents = env_cfg.game.num_agents
    actions = np.zeros((env.num_agents, 2), dtype=np.int32)
    total_rewards = np.zeros(env.num_agents)

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

    response = mettascope_module.init(replay=json.dumps(initial_replay))
    if response.should_close:
        return

    def generate_replay_step():
        grid_objects = []
        for grid_object in env.grid_objects.values():
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                total_rewards[agent_id] += env.rewards[agent_id]
            grid_objects.append(
                format_grid_object(grid_object, actions, env.action_success, env.rewards, total_rewards)
            )
        step_replay = {"step": step_count, "objects": grid_objects}
        return json.dumps(step_replay)

    while max_steps is None or step_count < max_steps:
        # Generate replay step
        replay_step = generate_replay_step()

        # Call each agent's policy to get actions
        for agent_id in range(num_agents):
            actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])

        response = mettascope_module.render(step_count, replay_step)
        if response.should_close:
            break
        if response.action:
            actions[response.action_agent_id, 0] = response.action_action_id
            actions[response.action_agent_id, 1] = response.action_argument

        obs, rewards, dones, truncated, info = env.step(actions)

        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
