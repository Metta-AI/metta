"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from cogames import utils
from cogames.policy import AgentPolicy, Policy, TrainablePolicy
from cogames.policy.loader import instantiate_policy, load_policy_checkpoint
from cogames.policy.registry import resolve_policy_class_path
from mettagrid import MettaGridEnv

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats


def evaluate(
    console: Console,
    *,
    game_name: str,
    policy_class_path: str,
    checkpoint_path: Optional[Path],
    episodes: int,
    seed: int = 42,
) -> None:
    """Evaluate a policy on the requested game and render rich tables."""
    if episodes <= 0:
        raise ValueError("Number of episodes must be greater than zero")

    resolved_game, env_cfg = utils.get_game_config(game_name)
    env = MettaGridEnv(env_cfg=env_cfg)

    resolved_policy_path = resolve_policy_class_path(policy_class_path)
    device = torch.device("cpu")

    try:
        policy_instance: Policy = instantiate_policy(resolved_policy_path, env, device)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to instantiate policy class '{resolved_policy_path}' for evaluation: {exc}"
        raise RuntimeError(msg) from exc

    resolved_checkpoint: Optional[Path] = None
    if checkpoint_path is not None:
        try:
            resolved_checkpoint = load_policy_checkpoint(policy_instance, checkpoint_path)
        except (FileNotFoundError, TypeError) as exc:
            raise exc
        except Exception as exc:  # pragma: no cover - loading errors
            raise RuntimeError(f"Failed to load policy data from {checkpoint_path}: {exc}") from exc

    agent_policies: List[AgentPolicy] = []
    if isinstance(policy_instance, TrainablePolicy):
        for agent_id in range(env.num_agents):
            agent_policies.append(policy_instance.agent_policy(agent_id))
    elif isinstance(policy_instance, Policy):
        agent_policies = [policy_instance for _ in range(env.num_agents)]
    else:
        raise TypeError("Policy must implement the Policy or TrainablePolicy interface")

    per_episode_rewards: List[np.ndarray] = []
    per_episode_stats: List["EpisodeStats"] = []

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        for agent_policy in agent_policies:
            agent_policy.reset()

        done = np.zeros(env.num_agents, dtype=bool)
        truncated = np.zeros(env.num_agents, dtype=bool)

        while not done.all() and not truncated.all():
            action_list: List[np.ndarray] = []
            for agent_id in range(env.num_agents):
                action = agent_policies[agent_id].step(obs[agent_id])
                action_list.append(np.array(action))

            actions = np.stack(action_list)
            obs, rewards, done, truncated, _ = env.step(actions)

        per_episode_rewards.append(np.array(env.get_episode_rewards(), dtype=float))
        per_episode_stats.append(deepcopy(env.get_episode_stats()))

    stacked_rewards = np.stack(per_episode_rewards)
    avg_rewards = stacked_rewards.mean(axis=0)
    total_rewards = stacked_rewards.sum(axis=1)

    aggregated_game_stats: Dict[str, float] = defaultdict(float)
    aggregated_agent_stats: List[Dict[str, float]] = [defaultdict(float) for _ in range(env.num_agents)]

    for stats in per_episode_stats:
        game_stats = stats.get("game", {}) if isinstance(stats, dict) else {}
        for key, value in game_stats.items():
            aggregated_game_stats[key] += float(value)

        agent_stats_list = stats.get("agent", []) if isinstance(stats, dict) else []
        for agent_id, agent_stats in enumerate(agent_stats_list):
            if agent_id >= env.num_agents:
                continue
            for key, value in agent_stats.items():
                aggregated_agent_stats[agent_id][key] += float(value)

    summary_table = Table(
        title=f"Evaluation Results for {resolved_game}",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Episode", justify="right")
    summary_table.add_column("Total Reward", justify="right")
    for agent_id in range(env.num_agents):
        summary_table.add_column(f"Agent {agent_id}", justify="right")

    for episode_idx, rewards in enumerate(stacked_rewards, start=1):
        row = [str(episode_idx), f"{total_rewards[episode_idx - 1]:.2f}"]
        row.extend(f"{reward:.2f}" for reward in rewards)
        summary_table.add_row(*row)

    console.print(summary_table)
    console.print(
        "Average reward per agent: " + ", ".join(f"Agent {idx}: {reward:.2f}" for idx, reward in enumerate(avg_rewards))
    )

    console.print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for key, value in sorted(aggregated_game_stats.items()):
        game_stats_table.add_row(key, f"{value / episodes:.2f}")
    console.print(game_stats_table)

    console.print("\n[bold cyan]Average Agent Stats[/bold cyan]")
    for agent_id, stats in enumerate(aggregated_agent_stats):
        agent_table = Table(title=f"Agent {agent_id}", show_header=True, header_style="bold magenta")
        agent_table.add_column("Metric")
        agent_table.add_column("Average", justify="right")
        for key, value in sorted(stats.items()):
            agent_table.add_row(key, f"{value / episodes:.2f}")
        console.print(agent_table)

    evaluation_metadata = {
        "game": resolved_game,
        "policy": resolved_policy_path,
        "checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
        "episodes": episodes,
        "average_rewards": avg_rewards.tolist(),
    }
    console.print("[dim]Metadata:[/dim]" + f" [dim]{json.dumps(evaluation_metadata, indent=2)}[/dim]")

    console.print()
    console.print("[green]Evaluation complete.[/green]")

    env.close()
