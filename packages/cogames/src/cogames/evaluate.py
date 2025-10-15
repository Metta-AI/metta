"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from cogames.policy.interfaces import PolicySpec
from cogames.policy.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]


def _compute_policy_agent_counts(num_agents: int, policy_specs: list[PolicySpec]) -> list[int]:
    total = sum(spec.proportion for spec in policy_specs)
    if total <= 0:
        raise ValueError("Total policy proportion must be positive.")
    fractions = [spec.proportion / total for spec in policy_specs]

    ideals = [num_agents * f for f in fractions]
    counts = [math.floor(x) for x in ideals]
    remaining = num_agents - sum(counts)

    # distribute by largest remainder
    remainders = [(i, ideals[i] - counts[i]) for i in range(len(fractions))]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        counts[remainders[i][0]] += 1
    return counts


def evaluate(
    console: Console,
    resolved_game: str,
    env_cfg: MettaGridConfig,
    policy_specs: list[PolicySpec],
    episodes: int,
    action_timeout_ms: int,
    max_steps: Optional[int] = None,
    seed: int = 42,
) -> None:
    if not policy_specs:
        raise ValueError("At least one policy specification must be provided for evaluation.")

    # Load env and policies
    env = MettaGridEnv(env_cfg=env_cfg)

    policy_instances = [
        initialize_or_load_policy(spec.policy_class_path, spec.policy_data_path, env) for spec in policy_specs
    ]
    policy_counts = _compute_policy_agent_counts(env.num_agents, policy_specs)
    policy_names = [spec.name for spec in policy_specs]

    if len(policy_specs) > 1:
        console.print("\n[bold cyan]Policy Assignments[/bold cyan]")
        policy_counts_table = Table(show_header=True, header_style="bold magenta")
        policy_counts_table.add_column("Policy")
        policy_counts_table.add_column("Num Agents", justify="right")
        for policy_name, count in zip(policy_names, policy_counts, strict=True):
            policy_counts_table.add_row(policy_name, str(count))
        console.print(policy_counts_table)
        console.print()

    assignments = np.repeat(np.arange(len(policy_specs)), policy_counts)

    assert len(assignments) == env.num_agents

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list["EpisodeStats"] = []
    per_episode_assignments: list[np.ndarray] = []
    per_policy_timeouts = defaultdict(int)

    # Run episodes
    progress_label = "Evaluating episodes"
    rng = np.random.default_rng(seed)
    noop = np.array(0, dtype=env.action_space.dtype)
    with typer.progressbar(range(episodes), label=progress_label) as progress:
        for episode_idx in progress:
            obs, _ = env.reset(seed=seed + episode_idx)
            # Shuffle assignments in place
            rng.shuffle(assignments)
            agent_policies = [
                policy_instances[assignments[agent_id]].agent_policy(agent_id) for agent_id in range(env.num_agents)
            ]
            for agent_policy in agent_policies:
                agent_policy.reset()

            done = np.zeros(env.num_agents, dtype=bool)
            truncated = np.zeros(env.num_agents, dtype=bool)

            step_count = 0

            while max_steps is None or step_count < max_steps:
                actions = np.zeros(env.num_agents, dtype=env.action_space.dtype)
                for i in range(env.num_agents):
                    start_time = time.time()
                    action = agent_policies[i].step(obs[i])
                    if isinstance(action, tuple):
                        raise TypeError(
                            "AgentPolicy.step must return a single MettaGridAction under the single-discrete API. "
                            "Update the policy to emit an int-compatible action instead of a tuple."
                        )
                    end_time = time.time()
                    if (end_time - start_time) > action_timeout_ms / 1000:
                        per_policy_timeouts[assignments[i]] += 1
                        action = noop
                    actions[i] = np.asarray(action).astype(env.action_space.dtype).item()
                obs, rewards, done, truncated, _ = env.step(actions)

                step_count += 1
                if done.all() or truncated.all():
                    break

            per_episode_rewards.append(np.array(env.get_episode_rewards(), dtype=float))

            per_episode_stats.append(deepcopy(env.get_episode_stats()))
            per_episode_assignments.append(assignments.copy())

    # Report results

    aggregated_game_stats: dict[str, float] = defaultdict(float)
    aggregated_policy_stats: list[dict[str, float]] = [defaultdict(float) for _ in policy_specs]

    for episode_idx, stats in enumerate(per_episode_stats):
        game_stats = stats.get("game", {})
        for key, value in game_stats.items():
            aggregated_game_stats[key] += float(value)

        agent_stats_list = stats.get("agent", [])
        for agent_id, agent_stats in enumerate(agent_stats_list):
            if agent_id >= env.num_agents:
                continue
            assignments = per_episode_assignments[episode_idx]
            policy_idx = int(assignments[agent_id])
            for key, value in agent_stats.items():
                if any(re.match(pattern, key) for pattern in _SKIP_STATS):
                    continue
                aggregated_policy_stats[policy_idx][key] += float(value)

    console.print("\n[bold cyan]Average Policy Stats[/bold cyan]")
    for policy_idx, stats in enumerate(aggregated_policy_stats):
        policy_table = Table(title=policy_names[policy_idx], show_header=True, header_style="bold magenta")
        policy_table.add_column("Metric")
        policy_table.add_column("Average", justify="right")
        count = policy_counts[policy_idx]
        for key, value in sorted(stats.items()):
            policy_table.add_row(key, f"{value / count:.2f}")
        console.print(policy_table)

    console.print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for key, value in sorted(aggregated_game_stats.items()):
        game_stats_table.add_row(key, f"{value / episodes:.2f}")
    console.print(game_stats_table)

    console.print(f"\n[bold cyan]Average Reward per Agent on {resolved_game}[/bold cyan]")
    summary_table = Table(
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Episode", justify="right")
    for name in policy_names:
        summary_table.add_column(name, justify="right")

    total_avg_agent_reward_per_policy = defaultdict(float)
    for episode_idx, rewards in enumerate(per_episode_rewards):
        assignments = per_episode_assignments[episode_idx]
        row = [str(episode_idx)]
        episode_reward_per_policy = defaultdict(float)
        for agent_id, reward in enumerate(rewards):
            policy_idx = int(assignments[agent_id])
            episode_reward_per_policy[policy_idx] += float(reward)
        for policy_idx in range(len(policy_specs)):
            avg_reward_per_agent = episode_reward_per_policy[policy_idx] / policy_counts[policy_idx]
            row.append(str(avg_reward_per_agent))
            total_avg_agent_reward_per_policy[policy_idx] += avg_reward_per_agent

        summary_table.add_row(*row)
    summary_table.add_row(
        "Total",
        *[str(total_avg_agent_reward_per_policy[policy_idx]) for policy_idx in range(len(policy_specs))],
    )
    console.print(summary_table)

    if per_policy_timeouts:
        console.print("\n[bold cyan]Action Generation Timeouts per[/bold cyan]")
        for policy_idx, timeouts in per_policy_timeouts.items():
            console.print(f"{policy_names[policy_idx]}: {timeouts} timeouts")

    env.close()
