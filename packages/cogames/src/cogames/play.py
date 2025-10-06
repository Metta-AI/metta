"""Game playing functionality for CoGames."""

import json
import logging
from typing import Literal, Optional

import numpy as np
from rich.console import Console
from typing_extensions import TYPE_CHECKING

from cogames.cogs_vs_clips.glyphs import GLYPHS
from cogames.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.grid_object_formatter import format_grid_object

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
    # Create environment with appropriate render mode
    render_mode = None if render == "gui" else "miniscope" if render == "text" else None
    env = MettaGridEnv(env_cfg=env_cfg, render_mode=render_mode)

    policy = initialize_or_load_policy(policy_class_path, policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]

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
            action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
            noop_action_id = action_lookup.get("noop", 0)
            direction_names = {
                0: "move_north",
                1: "move_south",
                2: "move_west",
                3: "move_east",
                4: "move_northwest",
                5: "move_northeast",
                6: "move_southwest",
                7: "move_southeast",
            }

            actions = np.full(env.num_agents, noop_action_id, dtype=np.int32)

            for agent_id in range(env.num_agents):
                if agent_id == selected_agent and manual_action is not None:
                    # Apply manual action to selected agent
                    if isinstance(manual_action, tuple):
                        actions[agent_id] = int(manual_action[0])
                    else:
                        move_name = direction_names.get(int(manual_action))
                        if move_name and move_name in action_lookup:
                            actions[agent_id] = action_lookup[move_name]
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
        actions = np.zeros(env.num_agents, dtype=np.int32)

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
    actions = np.zeros(env.num_agents, dtype=np.int32)
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
            actions[action.agent_id] = action.action_id

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
