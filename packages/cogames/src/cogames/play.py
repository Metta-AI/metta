"""Game playing functionality for CoGames."""

import json
import logging
from typing import Literal, Optional

import numpy as np
from rich.console import Console

import mettagrid.mettascope as mettascope
from cogames.cogs_vs_clips.glyphs import GLYPHS
from cogames.env import make_hierarchical_env
from cogames.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig
from mettagrid.util.grid_object_formatter import format_grid_object

logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: "MettaGridConfig",
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
    game_name: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
    render: Literal["gui", "text", "none"] = "gui",
    verbose: bool = False,
) -> None:
    """Play a single game episode with a policy."""

    render_mode: Optional[str]
    if render == "gui":
        render_mode = None
    elif render == "text":
        render_mode = "miniscope"
    elif render == "none":
        render_mode = None
    else:
        raise ValueError(f"Unknown render mode '{render}'.")

    env = make_hierarchical_env(env_cfg, render_mode=render_mode)

    policy = initialize_or_load_policy(policy_class_path, policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]
    action_dim = int(env.single_action_space.nvec.size)

    if render == "text" and getattr(env, "_renderer", None):
        move_action_id = env.action_names.index("move") if "move" in env.action_names else 0
        noop_action_id = env.action_names.index("noop") if "noop" in env.action_names else 0

        def get_actions_fn(
            obs: np.ndarray,
            selected_agent: Optional[int],
            manual_action: Optional[int | tuple[int, int]],
            manual_agents: set[int],
        ) -> np.ndarray:
            actions = np.zeros((env.num_agents, action_dim), dtype=np.int32)
            for agent in range(env.num_agents):
                actions[agent] = agent_policies[agent].step(obs[agent])

            for agent in manual_agents:
                if agent != selected_agent and 0 <= agent < env.num_agents:
                    override = actions[agent]
                    override[1:] = 0
                    override[0] = noop_action_id
                    actions[agent] = override

            if selected_agent is not None and manual_action is not None:
                override = actions[selected_agent]
                override[1:] = 0
                if isinstance(manual_action, tuple):
                    verb_idx, arg = manual_action
                else:
                    verb_idx = move_action_id
                    arg = manual_action
                override[0] = int(verb_idx)
                arg_slot = 1 + verb_idx
                if 0 <= arg_slot < override.size:
                    override[arg_slot] = int(arg)
                actions[selected_agent] = override

            return actions

        glyphs = None
        change_glyph_cfg = getattr(env_cfg.game.actions, "change_glyph", None)
        if change_glyph_cfg and getattr(change_glyph_cfg, "enabled", False):
            glyphs = GLYPHS

        env.reset(seed=seed)
        result = env._renderer.interactive_loop(env, get_actions_fn, max_steps=max_steps, glyphs=glyphs)
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {result['steps']}")
        console.print(f"Total Rewards: {result['total_rewards']}")
        return

    if render == "none":
        obs, _ = env.reset(seed=seed)
        total_rewards = np.zeros(env.num_agents)
        step_count = 0
        while max_steps is None or step_count < max_steps:
            actions = np.stack([agent_policies[agent_id].step(obs[agent_id]) for agent_id in range(env.num_agents)])
            obs, rewards, dones, truncated, _ = env.step(actions)
            total_rewards += rewards
            step_count += 1
            if verbose:
                console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")
            if all(dones) or all(truncated):
                break
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {step_count}")
        console.print(f"Total Rewards: {total_rewards.sum():.2f}")
        return

    obs, _ = env.reset(seed=seed)
    step_count = 0
    num_agents = env_cfg.game.num_agents
    hierarchical_actions = np.zeros((env.num_agents, action_dim), dtype=np.int32)
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

    response = mettascope.init(replay=json.dumps(initial_replay))
    if response.should_close:
        return

    def generate_replay_step() -> str:
        base_actions = env.project_actions(hierarchical_actions)
        grid_objects = []
        for grid_object in env.grid_objects().values():
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                total_rewards[agent_id] += env.rewards[agent_id]
            grid_objects.append(
                format_grid_object(grid_object, base_actions, env.action_success, env.rewards, total_rewards)
            )
        return json.dumps({"step": step_count, "objects": grid_objects})

    while max_steps is None or step_count < max_steps:
        for agent_id in range(num_agents):
            hierarchical_actions[agent_id] = agent_policies[agent_id].step(obs[agent_id])

        replay_step = generate_replay_step()
        response = mettascope.render(step_count, replay_step)
        if response.should_close:
            break

        manual_actions = getattr(response, "actions", None)
        if manual_actions:
            for manual in manual_actions:
                agent_idx = int(manual.agent_id)
                if 0 <= agent_idx < env.num_agents:
                    verb = int(manual.action_id)
                    arg = int(manual.argument)
                    hierarchical_actions[agent_idx, 1:] = 0
                    hierarchical_actions[agent_idx, 0] = verb
                    arg_slot = 1 + verb
                    if arg_slot < hierarchical_actions.shape[1]:
                        hierarchical_actions[agent_idx, arg_slot] = arg
        elif getattr(response, "action", None):  # Back-compat for single-action responses
            agent_idx = int(response.action_agent_id)
            verb = int(response.action_action_id)
            arg = int(response.action_argument)
            hierarchical_actions[agent_idx, 1:] = 0
            hierarchical_actions[agent_idx, 0] = verb
            arg_slot = 1 + verb
            if arg_slot < hierarchical_actions.shape[1]:
                hierarchical_actions[agent_idx, arg_slot] = arg

        obs, rewards, dones, truncated, _ = env.step(hierarchical_actions)
        total_rewards += rewards
        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
    console.print(f"Total Rewards: {total_rewards.sum():.2f}")
