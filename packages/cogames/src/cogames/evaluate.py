"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FUT_TIMEOUT
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from cogames.utils import instantiate_or_load_policy
from mettagrid import MettaGridEnv

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig
    from mettagrid.mettagrid_c import EpisodeStats


def evaluate(
    console: Console,
    resolved_game: str,
    env_cfg: "MettaGridConfig",
    policy_class_path: str,
    policy_data_path: Optional[str],
    episodes: int,
    action_timeout_ms: int,
    seed: int = 42,
) -> None:
    env = MettaGridEnv(env_cfg=env_cfg)
    noop = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    policy = instantiate_or_load_policy(policy_class_path, policy_data_path, env)
    agent_policies = [policy.agent_policy(agent_id) for agent_id in range(env.num_agents)]

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list["EpisodeStats"] = []

    with ThreadPoolExecutor(max_workers=env.num_agents) as pool:
        progress_label = "Evaluating episodes"
        with typer.progressbar(range(episodes), label=progress_label) as progress:
            for episode_idx in progress:
                obs, _ = env.reset(seed=seed + episode_idx)
                for p in agent_policies:
                    p.reset()

                done = np.zeros(env.num_agents, dtype=bool)
                truncated = np.zeros(env.num_agents, dtype=bool)

                while not done.all() and not truncated.all():
                    # submit one callable per agent
                    futures = [pool.submit(agent_policies[i].step, obs[i]) for i in range(env.num_agents)]

                    actions = []
                    for i, fut in enumerate(futures):
                        try:
                            a = fut.result(timeout=action_timeout_ms / 1000)
                        except FUT_TIMEOUT:
                            a = noop
                            typer.echo(f"[yellow]agent {i} timed out; using noop[/yellow]")
                        except Exception as e:
                            a = noop
                            typer.echo(f"[red]agent {i} failed: {e}; using noop[/red]")
                        actions.append(np.asarray(a))

                    actions = np.stack(actions, axis=0)
                    obs, rewards, done, truncated, _ = env.step(actions)

            per_episode_rewards.append(np.array(env.get_episode_rewards(), dtype=float))
            per_episode_stats.append(deepcopy(env.get_episode_stats()))

    stacked_rewards = np.stack(per_episode_rewards)
    avg_rewards = stacked_rewards.mean(axis=0)
    total_rewards = stacked_rewards.sum(axis=1)

    aggregated_game_stats: dict[str, float] = defaultdict(float)
    aggregated_agent_stats: list[dict[str, float]] = [defaultdict(float) for _ in range(env.num_agents)]

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
        "policy": policy_class_path,
        "policy_data": policy_data_path,
        "episodes": episodes,
        "average_rewards": avg_rewards.tolist(),
    }
    console.print("[dim]Metadata:[/dim]" + f" [dim]{json.dumps(evaluation_metadata, indent=2)}[/dim]")

    console.print()
    console.print("[green]Evaluation complete.[/green]")

    env.close()
