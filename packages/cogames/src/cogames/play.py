"""Game playing functionality for CoGames."""

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from rich.console import Console

from cogames.policy.interfaces import PolicySpec
from cogames.policy.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv, RenderMode, dtype_actions

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig


logger = logging.getLogger("cogames.play")

DIRECTION_ACTION_NAMES: dict[int, str] = {
    0: "move_north",
    1: "move_south",
    2: "move_west",
    3: "move_east",
    4: "move_northwest",
    5: "move_northeast",
    6: "move_southwest",
    7: "move_southeast",
}


def _flatten_action_request(
    action_request: Any,
    *,
    total_actions: int,
    noop_action_id: int,
    move_action_lookup: dict[int, int],
) -> int:
    """Translate a MettaScope ActionRequest into a flattened action index."""

    raw_action_id = int(getattr(action_request, "action_id", -1))
    if 0 <= raw_action_id < total_actions:
        return raw_action_id

    argument_value = getattr(action_request, "argument", None)
    if argument_value is not None:
        orientation_idx = int(argument_value)
        flattened_move = move_action_lookup.get(orientation_idx)
        if flattened_move is not None:
            return flattened_move

    logger.debug(
        "Received unrecognized manual action; defaulting to noop",
        extra={
            "action_id": raw_action_id,
            "argument": getattr(action_request, "argument", None),
        },
    )
    return noop_action_id


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_spec: PolicySpec,
    game_name: str,
    max_steps: Optional[int] = None,
    seed: int = 42,
    render: RenderMode = "gui",
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

    if game_name:
        logger.debug("Starting play session", extra={"game_name": game_name})
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render)

    policy = initialize_or_load_policy(policy_spec.policy_class_path, policy_spec.policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]
    action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
    noop_action_id = action_lookup.get("noop", 0)
    move_action_lookup = {
        orientation: action_lookup[name]
        for orientation, name in DIRECTION_ACTION_NAMES.items()
        if name in action_lookup
    }

    obs, _ = env.reset(seed=seed)

    # Standard game loop
    step_count = 0
    num_agents = env_cfg.game.num_agents
    actions = np.zeros(env.num_agents, dtype=dtype_actions)
    total_rewards = np.zeros(env.num_agents)

    while max_steps is None or step_count < max_steps:
        # Check if renderer wants to continue (e.g., user quit or interactive loop finished)
        if not env._renderer.should_continue():
            break

        # Render the environment (handles display and user input)
        env.render()

        # Get user actions from renderer (if any)
        user_actions = env._renderer.get_user_actions()

        # Get actions - use user input if available, otherwise use policy
        for agent_id in range(num_agents):
            if agent_id in user_actions:
                # User provided action for this agent
                action_id, action_param = user_actions[agent_id]
                # Flatten the action using the helper function
                actions[agent_id] = _flatten_action_request(
                    SimpleNamespace(action_id=action_id, argument=action_param),
                    total_actions=len(env.action_names),
                    noop_action_id=noop_action_id,
                    move_action_lookup=move_action_lookup,
                )
            else:
                # Use policy action
                actions[agent_id] = int(agent_policies[agent_id].step(obs[agent_id]))

        # Step the environment
        obs, rewards, dones, truncated, _ = env.step(actions)

        # Update total rewards
        total_rewards += rewards
        step_count += 1

        if all(dones) or all(truncated):
            break

    # Print summary
    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
    console.print(f"Total Rewards: {total_rewards}")
    console.print(f"Final Reward Sum: {float(sum(total_rewards)):.2f}")
