"""Game playing functionality for CoGames."""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from rich.console import Console

import mettagrid.mettascope as mettascope
from cogames import serialization
from cogames.aws_storage import maybe_download_checkpoint
from cogames.env import make_hierarchical_env
from mettagrid import MettaGridConfig
from mettagrid.util.grid_object_formatter import format_grid_object

logger = logging.getLogger("cogames.play")


def play(
    console: Console,
    env_cfg: MettaGridConfig,
    policy_class_path: str,
    policy_data_path: Optional[str] = None,
    game_name: Optional[str] = None,
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

    # Create environment with appropriate render mode while preserving hierarchical actions
    render_mode = None if render == "gui" else "miniscope" if render == "text" else None
    env = make_hierarchical_env(env_cfg, render_mode=render_mode)

    # Load and create policy via shared serialization helpers
    device = torch.device("cpu")
    policy_path = Path(policy_data_path) if policy_data_path else None
    if policy_path and not policy_path.exists() and not policy_path.is_dir():
        downloaded = maybe_download_checkpoint(
            policy_path=policy_path,
            game_name=game_name,
            policy_class_path=policy_class_path,
            console=console,
        )
        if not downloaded and not policy_path.exists():
            console.print(f"[red]Policy checkpoint not found at {policy_path} and no remote copy was located.[/red]")
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
    if policy_path and policy_path.is_dir():
        policy_instance = serialization.load_policy_from_bundle(policy_path, env, device)
    else:
        artifact = serialization.PolicyArtifact(policy_class=policy_class_path, weights_path=policy_path)
        policy_instance = serialization.load_policy(artifact, env, device)

    from cogames.policy import AgentPolicy, Policy, TrainablePolicy

    agent_policies: list[AgentPolicy] = []
    if isinstance(policy_instance, (TrainablePolicy, Policy)):
        for agent_id in range(env.num_agents):
            agent_policies.append(policy_instance.agent_policy(agent_id))
    else:
        raise ValueError("Policy class must implement either Policy or TrainablePolicy interface")

    action_dim = int(env.single_action_space.nvec.size)

    # Text mode: drive miniscope interactive loop
    if render == "text" and getattr(env, "_renderer", None):
        move_action_id = env.action_names.index("move") if "move" in env.action_names else 0

        def get_actions_fn(
            obs: np.ndarray, selected_agent: Optional[int], manual_action: Optional[int | tuple]
        ) -> np.ndarray:
            """Return hierarchical actions for all agents with optional manual override."""

            actions = np.zeros((env.num_agents, action_dim), dtype=np.int32)
            for agent in range(env.num_agents):
                actions[agent] = agent_policies[agent].step(obs[agent])

            if selected_agent is not None and manual_action is not None:
                override = actions[selected_agent]
                override[1:] = 0

                if isinstance(manual_action, tuple):
                    verb, arg = manual_action
                    verb_idx = int(verb)
                    override[0] = verb_idx
                    arg_slot = 1 + verb_idx
                    if arg_slot < override.size:
                        override[arg_slot] = int(arg)
                else:
                    override[0] = move_action_id
                    arg_slot = 1 + move_action_id
                    if arg_slot < override.size:
                        override[arg_slot] = int(manual_action)

                actions[selected_agent] = override

            return actions

        env.reset(seed=seed)
        result = env._renderer.interactive_loop(env, get_actions_fn, max_steps=max_steps)
        console.print("\n[bold green]Episode Complete![/bold green]")
        console.print(f"Steps: {result['steps']}")
        console.print(f"Total Rewards: {result['total_rewards']}")
        return

    # GUI mode: use mettascope replay visualizer
    obs, _ = env.reset(seed=seed)
    step_count = 0
    num_agents = env_cfg.game.num_agents
    hierarchical_actions = np.zeros((env.num_agents, action_dim), dtype=np.int32)
    total_rewards = np.zeros(env.num_agents)

    # Initialize GUI replay payload
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
        step_replay = {"step": step_count, "objects": grid_objects}
        return json.dumps(step_replay)

    # GUI rendering loop
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
                if agent_idx < 0 or agent_idx >= env.num_agents:
                    continue
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

        for agent_id in range(num_agents):
            total_rewards[agent_id] += rewards[agent_id]

        step_count += 1

        if verbose:
            console.print(f"Step {step_count}: Reward = {float(sum(rewards)):.2f}")

        if all(dones) or all(truncated):
            break

    console.print("\n[bold green]Episode Complete![/bold green]")
    console.print(f"Steps: {step_count}")
    console.print(f"Total Rewards: {total_rewards.sum():.2f}")
