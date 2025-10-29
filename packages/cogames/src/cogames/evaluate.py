"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, DefaultDict, Literal, Optional

import numpy as np
import typer
import yaml  # type: ignore[import]
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from cogames.policy.interfaces import AgentPolicy, PolicySpec
from cogames.policy.utils import initialize_or_load_policy
from mettagrid import MettaGridConfig, MettaGridEnv

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]


@dataclass
class MissionEvaluationResult:
    mission_name: str
    policy_counts: list[int]
    policy_names: list[str]
    aggregated_policy_stats: list[dict[str, float]]
    aggregated_game_stats: dict[str, float]
    per_episode_rewards: list[np.ndarray]
    per_episode_assignments: list[np.ndarray]
    per_policy_timeouts: dict[int, int]
    episodes: int


class EpisodeRewardSummary(BaseModel):
    episode_index: int
    average_reward_per_policy: list[float]


class MissionPolicySummary(BaseModel):
    policy_name: str
    display_name: str
    agent_count: int
    average_metrics: dict[str, float]
    average_reward_per_agent: float
    action_timeouts: int


class MissionSummary(BaseModel):
    mission_name: str
    episodes: int
    policy_summaries: list[MissionPolicySummary]
    average_game_metrics: dict[str, float]
    episode_reward_breakdown: list[EpisodeRewardSummary]
    total_average_reward_per_policy: list[float]


class MissionResultsSummary(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    missions: list[MissionSummary]


def _build_results_summary(
    mission_results: list[MissionEvaluationResult],
    policy_specs: list[PolicySpec],
) -> MissionResultsSummary:
    if not mission_results:
        return MissionResultsSummary(missions=[])

    name_count: defaultdict[str, int] = defaultdict(int)
    display_names: list[str] = []
    for spec in policy_specs:
        name_count[spec.name] += 1
        if name_count[spec.name] > 1:
            display_names.append(f"{spec.name} ({name_count[spec.name]})")
        else:
            display_names.append(spec.name)

    summaries: list[MissionSummary] = []

    for result in mission_results:
        policy_summaries: list[MissionPolicySummary] = []

        episode_breakdown: list[EpisodeRewardSummary] = []
        cumulative_average_rewards = [0.0 for _ in policy_specs]
        for episode_idx, rewards in enumerate(result.per_episode_rewards):
            episode_assignments = result.per_episode_assignments[episode_idx]
            per_policy_totals = [0.0 for _ in policy_specs]
            for agent_id, reward in enumerate(rewards):
                policy_idx = int(episode_assignments[agent_id])
                per_policy_totals[policy_idx] += float(reward)
            per_policy_average = [
                (per_policy_totals[i] / result.policy_counts[i]) if result.policy_counts[i] > 0 else 0.0
                for i in range(len(policy_specs))
            ]
            episode_breakdown.append(
                EpisodeRewardSummary(episode_index=episode_idx, average_reward_per_policy=per_policy_average)
            )
            cumulative_average_rewards = [
                cumulative_average_rewards[i] + per_policy_average[i] for i in range(len(policy_specs))
            ]

        total_average_reward_per_policy = [
            (value / result.episodes) if result.episodes > 0 else 0.0 for value in cumulative_average_rewards
        ]

        for policy_idx, policy_name in enumerate(result.policy_names):
            agent_count = result.policy_counts[policy_idx]
            raw_metrics = result.aggregated_policy_stats[policy_idx]
            average_metrics = {
                key: (value / agent_count if agent_count > 0 else 0.0) for key, value in sorted(raw_metrics.items())
            }

            action_timeouts = result.per_policy_timeouts.get(policy_idx, 0)

            policy_summaries.append(
                MissionPolicySummary(
                    policy_name=policy_name,
                    display_name=display_names[policy_idx],
                    agent_count=agent_count,
                    average_metrics=average_metrics,
                    average_reward_per_agent=total_average_reward_per_policy[policy_idx],
                    action_timeouts=action_timeouts,
                )
            )

        average_game_metrics = {
            key: (value / result.episodes if result.episodes > 0 else 0.0)
            for key, value in sorted(result.aggregated_game_stats.items())
        }

        summaries.append(
            MissionSummary(
                mission_name=result.mission_name,
                episodes=result.episodes,
                policy_summaries=policy_summaries,
                average_game_metrics=average_game_metrics,
                episode_reward_breakdown=episode_breakdown,
                total_average_reward_per_policy=total_average_reward_per_policy,
            )
        )

    return MissionResultsSummary(missions=summaries)


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
    missions: list[tuple[str, MettaGridConfig]],
    policy_specs: list[PolicySpec],
    episodes: int,
    action_timeout_ms: int,
    max_steps: Optional[int] = None,
    seed: int = 42,
    output_format: Optional[Literal["yaml", "json"]] = None,
) -> MissionResultsSummary:
    if not missions:
        raise ValueError("At least one mission must be provided for evaluation.")
    if not policy_specs:
        raise ValueError("At least one policy specification must be provided for evaluation.")

    mission_names = [mission_name for mission_name, _ in missions]
    if len(missions) == 1:
        console.print(
            f"[cyan]Evaluating {len(policy_specs)} policies on {mission_names[0]} over {episodes} episodes[/cyan]"
        )
    else:
        console.print(f"[cyan]Evaluating {len(policy_specs)} policies over {episodes} episodes per mission[/cyan]")
        console.print("Missions:")
        for mission_name in mission_names:
            console.print(f"- {mission_name}")

    mission_results: list[MissionEvaluationResult] = []
    for mission_name, env_cfg in missions:
        mission_results.append(
            _evaluate_single_mission(
                mission_name=mission_name,
                env_cfg=env_cfg,
                policy_specs=policy_specs,
                episodes=episodes,
                action_timeout_ms=action_timeout_ms,
                max_steps=max_steps,
                seed=seed,
            )
        )

    summary = _build_results_summary(mission_results, policy_specs)
    _output_results(console, summary, output_format)
    return summary


def _output_results(
    console: Console, summary: MissionResultsSummary, output_format: Optional[Literal["yaml", "json"]]
) -> None:
    if output_format:
        if output_format == "json":
            serialized = json.dumps(summary.model_dump(mode="json"), indent=2)
        else:
            serialized = yaml.safe_dump(summary.model_dump(), sort_keys=False)
        console.print(serialized)
        return

    policy_display_names = [ps.display_name for ps in summary.missions[0].policy_summaries]

    if len(policy_display_names) > 1:
        console.print("\n[bold cyan]Policy Assignments[/bold cyan]")
        assignment_table = Table(show_header=True, header_style="bold magenta")
        assignment_table.add_column("Mission")
        assignment_table.add_column("Policy")
        assignment_table.add_column("Num Agents", justify="right")
        for mission in summary.missions:
            for policy_summary in mission.policy_summaries:
                assignment_table.add_row(
                    mission.mission_name,
                    policy_summary.display_name,
                    str(policy_summary.agent_count),
                )
        console.print(assignment_table)

    console.print("\n[bold cyan]Average Policy Stats[/bold cyan]")
    for policy_idx, policy_name in enumerate(policy_display_names):
        policy_table = Table(title=policy_name, show_header=True, header_style="bold magenta")
        policy_table.add_column("Mission")
        policy_table.add_column("Metric")
        policy_table.add_column("Average", justify="right")
        for mission in summary.missions:
            metrics = mission.policy_summaries[policy_idx].average_metrics
            if not metrics:
                policy_table.add_row(mission.mission_name, "-", "0.00")
                continue
            for key, value in metrics.items():
                policy_table.add_row(mission.mission_name, key, f"{value:.2f}")
        console.print(policy_table)

    console.print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Mission")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for mission in summary.missions:
        for key, value in mission.average_game_metrics.items():
            game_stats_table.add_row(mission.mission_name, key, f"{value:.2f}")
    console.print(game_stats_table)

    console.print("\n[bold cyan]Average Reward per Agent[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Mission")
    summary_table.add_column("Episode", justify="right")
    for display_name in policy_display_names:
        summary_table.add_column(display_name, justify="right")

    for mission in summary.missions:
        for episode in mission.episode_reward_breakdown:
            row = [mission.mission_name, str(episode.episode_index)]
            row.extend(f"{value:.2f}" for value in episode.average_reward_per_policy)
            summary_table.add_row(*row)

        total_row = [mission.mission_name, "Total"]
        total_row.extend(f"{value:.2f}" for value in mission.total_average_reward_per_policy)
        summary_table.add_row(*total_row)

    console.print(summary_table)

    if any(policy.action_timeouts for mission in summary.missions for policy in mission.policy_summaries):
        console.print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Mission")
        timeouts_table.add_column("Policy")
        timeouts_table.add_column("Timeouts", justify="right")
        for mission in summary.missions:
            for policy_summary in mission.policy_summaries:
                if policy_summary.action_timeouts > 0:
                    timeouts_table.add_row(
                        mission.mission_name,
                        policy_summary.display_name,
                        str(policy_summary.action_timeouts),
                    )
        console.print(timeouts_table)


def _evaluate_single_mission(
    mission_name: str,
    env_cfg: MettaGridConfig,
    policy_specs: list[PolicySpec],
    episodes: int,
    action_timeout_ms: int,
    max_steps: Optional[int],
    seed: int,
) -> MissionEvaluationResult:
    env = MettaGridEnv(env_cfg=env_cfg)

    policy_instances = [
        initialize_or_load_policy(spec.policy_class_path, spec.policy_data_path, env) for spec in policy_specs
    ]
    policy_counts = _compute_policy_agent_counts(env.num_agents, policy_specs)
    policy_names = [spec.name for spec in policy_specs]

    assignments = np.repeat(np.arange(len(policy_specs)), policy_counts)

    assert len(assignments) == env.num_agents

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list["EpisodeStats"] = []
    per_episode_assignments: list[np.ndarray] = []
    per_policy_timeouts: DefaultDict[int, int] = defaultdict(int)

    progress_label = f"Evaluating episodes ({mission_name})"
    rng = np.random.default_rng(seed)
    noop = np.array(0, dtype=env.action_space.dtype)
    with typer.progressbar(range(episodes), label=progress_label) as progress:
        for episode_idx in progress:
            obs, _ = env.reset(seed=seed + episode_idx)
            rng.shuffle(assignments)
            agent_policies: list[AgentPolicy] = [
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
            episode_assignments = per_episode_assignments[episode_idx]
            policy_idx = int(episode_assignments[agent_id])
            for key, value in agent_stats.items():
                if any(re.match(pattern, key) for pattern in _SKIP_STATS):
                    continue
                aggregated_policy_stats[policy_idx][key] += float(value)

    env.close()

    return MissionEvaluationResult(
        mission_name=mission_name,
        policy_counts=policy_counts,
        policy_names=policy_names,
        aggregated_policy_stats=[dict(stats) for stats in aggregated_policy_stats],
        aggregated_game_stats=dict(aggregated_game_stats),
        per_episode_rewards=per_episode_rewards,
        per_episode_assignments=per_episode_assignments,
        per_policy_timeouts=dict(per_policy_timeouts),
        episodes=episodes,
    )
