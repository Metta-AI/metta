"""Game playing functionality for CoGames."""

import json
import logging
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np
from rich.console import Console
from typing_extensions import TYPE_CHECKING

from cogames.cogs_vs_clips.glyphs import GLYPHS
from cogames.policy.interfaces import PolicySpec
from cogames.policy.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv, dtype_actions
from mettagrid.util.grid_object_formatter import format_grid_object

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
    render: Literal["gui", "text", "none"] = "gui",
    verbose: bool = False,
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
        render: Render mode - "gui" (default), "text", or "none" (no rendering)
        verbose: Whether to print detailed progress
    """
    # Create environment with appropriate render mode
    render_mode = None if render == "gui" else "miniscope" if render == "text" else None
    if game_name:
        logger.debug("Starting play session", extra={"game_name": game_name})
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render_mode)

    policy = initialize_or_load_policy(policy_spec.policy_class_path, policy_spec.policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]
    action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
    noop_action_id = action_lookup.get("noop", 0)
    move_action_lookup = {
        orientation: action_lookup[name]
        for orientation, name in DIRECTION_ACTION_NAMES.items()
        if name in action_lookup
    }

    # For text mode, use the interactive loop in miniscope
    if render == "text" and hasattr(env, "_renderer") and env._renderer:

        def get_actions_fn(
            obs: np.ndarray,
            selected_agent: Optional[int],
            manual_action: Optional[int | tuple],
            manual_agents: set[int],
        ) -> np.ndarray:
            """Get actions for all agents, with optional manual override.

            Args:
                obs: Observations for all agents
                selected_agent: Currently selected agent (for manual control)
                manual_action: Manual action to apply to selected agent
                manual_agents: Set of agent IDs in manual mode (no policy actions)

            Returns:
                Actions array for all agents
            """
            actions = np.full(env.num_agents, noop_action_id, dtype=dtype_actions)

            for agent_id in range(env.num_agents):
                if agent_id == selected_agent and manual_action is not None:
                    manual_action_value = manual_action
                    if isinstance(manual_action_value, str):
                        if manual_action_value not in action_lookup:
                            raise ValueError(
                                f"Manual action '{manual_action_value}' is not available in the action space."
                            )
                        actions[agent_id] = action_lookup[manual_action_value]
                    elif isinstance(manual_action_value, tuple):
                        tuple_action_id = manual_action_value[0] if manual_action_value else noop_action_id
                        tuple_argument = manual_action_value[1] if len(manual_action_value) > 1 else 0
                        flattened_tuple = _flatten_action_request(
                            SimpleNamespace(action_id=tuple_action_id, argument=tuple_argument),
                            total_actions=len(env.action_names),
                            noop_action_id=noop_action_id,
                            move_action_lookup=move_action_lookup,
                        )
                        actions[agent_id] = flattened_tuple
                    else:
                        manual_idx = int(manual_action_value)
                        move_name = DIRECTION_ACTION_NAMES.get(manual_idx)
                        if move_name and move_name in action_lookup:
                            actions[agent_id] = action_lookup[move_name]
                        else:
                            actions[agent_id] = manual_idx
                elif agent_id in manual_agents:
                    # Agent is in manual mode but no action this step - use noop
                    actions[agent_id] = noop_action_id
                else:
                    # Use policy for this agent
                    policy_action = agent_policies[agent_id].step(obs[agent_id])
                    actions[agent_id] = int(policy_action)
            return actions

        # Get glyphs from environment config if available
        glyphs = None
        if env_cfg.game.actions.change_glyph.enabled:
            glyphs = GLYPHS

        result = env._renderer.interactive_loop(env, get_actions_fn, max_steps=max_steps, glyphs=glyphs)
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {result['steps']}")
        console.print(f"Total Rewards: {result['total_rewards']}")
        return

    # No rendering mode: just run the game
    if render == "none":
        obs, _ = env.reset(seed=seed)
        step_count = 0
        total_rewards = np.zeros(env.num_agents)
        actions = np.zeros(env.num_agents, dtype=dtype_actions)

        while max_steps is None or step_count < max_steps:
            # Get actions from policies
            for agent_id in range(env.num_agents):
                actions[agent_id] = int(agent_policies[agent_id].step(obs[agent_id]))

            # Step the environment
            obs, rewards, dones, truncated, _ = env.step(actions)

            # Update total rewards
            for agent_id in range(env.num_agents):
                total_rewards[agent_id] += rewards[agent_id]

            step_count += 1

            if verbose:
                console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

            if all(dones) or all(truncated):
                break

        # Print summary
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {step_count}")
        console.print(f"Total Rewards: {total_rewards}")
        console.print(f"Final Reward Sum: {float(sum(total_rewards)):.2f}")
        return

    # GUI mode: use mettascope
    obs, _ = env.reset(seed=seed)
    step_count = 0
    num_agents = env_cfg.game.num_agents
    actions = np.zeros(env.num_agents, dtype=dtype_actions)
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
    # Lazy import to avoid needing x11 dependencies during training
    import mettagrid.mettascope as mettascope

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
            actions[agent_id] = int(agent_policies[agent_id].step(obs[agent_id]))

        # Render and get user input
        replay_step = generate_replay_step()
        response = mettascope.render(step_count, replay_step)
        if response.should_close:
            break
        for action in response.actions:
            flattened = _flatten_action_request(
                action,
                total_actions=len(env.action_names),
                noop_action_id=noop_action_id,
                move_action_lookup=move_action_lookup,
            )
            actions[action.agent_id] = flattened

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
