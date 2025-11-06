"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import typer
import yaml  # type: ignore[import]
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.utils import initialize_or_load_policy
from mettagrid.simulator.rollout import Rollout

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]


@dataclass
class MissionEvaluationResult:
    # Name of the mission
    mission_name: str
    # Names for each policy, not necessarily unique.
    policy_names: list[str]
    # Number of agents assigned to each policy for this mission (aligned with policy_names order).
    policy_counts: list[int]
    # Per-policy metrics summed across all agents/episodes for this mission (aligned with policy_names order).
    summed_policy_stats: list[dict[str, float]]
    # Rewards returned by the environment for each episode
    # per_episode_rewards[episode_idx][agent_id] = reward achieved by agent_id in episode episode_idx.
    per_episode_rewards: list[np.ndarray]
    # List of agent -> policy assignments for each episode
    # per_episode_assignments[episode_idx][agent_id] = policy_idx, where policy_idx is aligned with policy_names order.
    per_episode_assignments: list[np.ndarray]
    # Count of action timeouts per policy across all episodes for this mission (aligned with policy_names order).
    per_policy_timeouts: np.ndarray
    # Game metrics averaged across episodes for this mission.
    avg_game_stats: dict[str, float]
    # Total number of episodes simulated for this mission.
    episodes: int


class MissionPolicySummary(BaseModel):
    # Possibly non-unique
    policy_name: str
    # Number of agents assigned to this policy for this mission
    agent_count: int
    # Average metrics across agents assigned to this policy for this mission
    avg_agent_metrics: dict[str, float]
    # Number of action timeouts experienced for this policy for this mission
    action_timeouts: int


class MissionSummary(BaseModel):
    # Name of the mission
    mission_name: str
    # Total number of episodes simulated for this mission
    episodes: int
    # Summaries for each policy for this mission
    policy_summaries: list[MissionPolicySummary]
    # Averaged game stats across all episodes for this mission
    avg_game_stats: dict[str, float]
    # per_episode_per_policy_avg_rewards[episode_idx][policy_idx] = \
    #     average reward per policy for this episode (or None if the policy had no agents in this episode)
    per_episode_per_policy_avg_rewards: dict[int, list[float | None]]


class MissionResultsSummary(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    missions: list[MissionSummary]


def _build_results_summary(
    mission_results: list[MissionEvaluationResult],
    policy_specs: list[PolicySpec],
) -> MissionResultsSummary:
    if not mission_results:
        return MissionResultsSummary(missions=[])

    summaries: list[MissionSummary] = []

    for mr in mission_results:
        policy_summaries: list[MissionPolicySummary] = []

        per_episode_per_policy_avg_rewards: dict[int, list[float | None]] = defaultdict(list)
        for episode_idx, rewards in enumerate(mr.per_episode_rewards):
            episode_assignments = mr.per_episode_assignments[episode_idx]
            per_policy_totals = np.zeros(len(policy_specs), dtype=float)
            for agent_id, reward in enumerate(rewards):
                policy_idx = int(episode_assignments[agent_id])
                per_policy_totals[policy_idx] += float(reward)
            per_episode_per_policy_avg_rewards[episode_idx] = [
                (per_policy_totals[i] / mr.policy_counts[i]) if mr.policy_counts[i] > 0 else None
                for i in range(len(policy_specs))
            ]

        for policy_idx, policy_name in enumerate(mr.policy_names):
            agent_count = mr.policy_counts[policy_idx]
            average_metrics = {
                key: value / agent_count for key, value in sorted(mr.summed_policy_stats[policy_idx].items())
            }

            policy_summaries.append(
                MissionPolicySummary(
                    policy_name=policy_name,
                    agent_count=agent_count,
                    avg_agent_metrics=average_metrics,
                    action_timeouts=mr.per_policy_timeouts[policy_idx],
                )
            )

        summaries.append(
            MissionSummary(
                mission_name=mr.mission_name,
                episodes=mr.episodes,
                policy_summaries=policy_summaries,
                avg_game_stats=mr.avg_game_stats,
                per_episode_per_policy_avg_rewards=per_episode_per_policy_avg_rewards,
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
    _output_results(console, policy_specs, summary, output_format)
    return summary


def _output_results(
    console: Console,
    policy_specs: list[PolicySpec],
    summary: MissionResultsSummary,
    output_format: Optional[Literal["yaml", "json"]],
) -> None:
    if output_format:
        if output_format == "json":
            serialized = json.dumps(summary.model_dump(mode="json"), indent=2)
        else:
            serialized = yaml.safe_dump(summary.model_dump(), sort_keys=False)
        console.print(serialized)
        return

    name_count: defaultdict[str, int] = defaultdict(int)
    display_names: list[str] = []
    for policy_spec in policy_specs:
        name_count[policy_spec.name] += 1
        if name_count[policy_spec.name] > 1:
            display_names.append(f"{policy_spec.name} ({name_count[policy_spec.name]})")
        else:
            display_names.append(policy_spec.name)

    console.print("\n[bold cyan]Policy Assignments[/bold cyan]")
    assignment_table = Table(show_header=True, header_style="bold magenta")
    assignment_table.add_column("Mission")
    assignment_table.add_column("Policy")
    assignment_table.add_column("Num Agents", justify="right")
    for mission in summary.missions:
        for policy_summary in mission.policy_summaries:
            assignment_table.add_row(
                mission.mission_name,
                policy_summary.policy_name,
                str(policy_summary.agent_count),
            )
    console.print(assignment_table)

    console.print("\n[bold cyan]Average Policy Stats[/bold cyan]")
    for i, policy_name in enumerate(display_names):
        policy_table = Table(title=policy_name, show_header=True, header_style="bold magenta")
        policy_table.add_column("Mission")
        policy_table.add_column("Metric")
        policy_table.add_column("Average", justify="right")
        for mission in summary.missions:
            metrics = mission.policy_summaries[i].avg_agent_metrics
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
        for key, value in mission.avg_game_stats.items():
            game_stats_table.add_row(mission.mission_name, key, f"{value:.2f}")
    console.print(game_stats_table)

    console.print("\n[bold cyan]Average Reward per Agent[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Mission")
    summary_table.add_column("Episode", justify="right")
    for display_name in display_names:
        summary_table.add_column(display_name, justify="right")

    for mission in summary.missions:
        for episode_idx, avg_rewards in sorted(mission.per_episode_per_policy_avg_rewards.items(), key=lambda x: x[0]):
            row = [mission.mission_name, str(episode_idx)]
            row.extend((f"{value:.2f}" if value is not None else "-" for value in avg_rewards))
            summary_table.add_row(*row)

    console.print(summary_table)

    if any(policy.action_timeouts for mission in summary.missions for policy in mission.policy_summaries):
        console.print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Mission")
        timeouts_table.add_column("Policy")
        timeouts_table.add_column("Timeouts", justify="right")
        for mission in summary.missions:
            for i, policy_summary in enumerate(mission.policy_summaries):
                if policy_summary.action_timeouts > 0:
                    timeouts_table.add_row(
                        mission.mission_name,
                        display_names[i],
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
    # Create PolicyEnvInterface from config to get observation shape for policies that require it (e.g., LSTM)
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface

    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    policy_instances = [
        initialize_or_load_policy(
            policy_env_info,
            spec.policy_class_path,
            spec.policy_data_path,
        )
        for spec in policy_specs
    ]
    policy_counts = _compute_policy_agent_counts(env_cfg.game.num_agents, policy_specs)
    policy_names = [spec.name for spec in policy_specs]

    assignments = np.repeat(np.arange(len(policy_specs)), policy_counts)

    assert len(assignments) == env_cfg.game.num_agents

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list["EpisodeStats"] = []
    per_episode_assignments: list[np.ndarray] = []
    per_policy_timeouts: np.ndarray = np.zeros(len(policy_specs), dtype=int)

    progress_label = f"Evaluating episodes ({mission_name})"
    rng = np.random.default_rng(seed)
    with typer.progressbar(range(episodes), label=progress_label) as progress:
        for episode_idx in progress:
            rng.shuffle(assignments)
            agent_policies = [
                policy_instances[assignments[agent_id]].agent_policy(agent_id)
                for agent_id in range(env_cfg.game.num_agents)
            ]

            rollout = Rollout(
                env_cfg,
                agent_policies,
                max_action_time_ms=action_timeout_ms,
                render_mode=None,
                seed=seed + episode_idx,
            )

            rollout.run_until_done()

            per_episode_rewards.append(np.array(rollout._sim.episode_rewards, dtype=float))
            per_episode_stats.append(rollout._sim.episode_stats)
            per_episode_assignments.append(assignments.copy())

            # Aggregate timeout counts by policy
            for agent_id, timeout_count in enumerate(rollout.timeout_counts):
                if timeout_count > 0:
                    policy_idx = int(assignments[agent_id])
                    per_policy_timeouts[policy_idx] += timeout_count

    summed_game_stats: dict[str, float] = defaultdict(float)
    summed_policy_stats: list[dict[str, float]] = [defaultdict(float) for _ in policy_specs]

    for episode_idx, stats in enumerate(per_episode_stats):
        game_stats = stats.get("game", {})
        for key, value in game_stats.items():
            summed_game_stats[key] += float(value)

        agent_stats_list = stats.get("agent", [])
        for agent_id, agent_stats in enumerate(agent_stats_list):
            policy_idx = int(per_episode_assignments[episode_idx][agent_id])
            for key, value in agent_stats.items():
                if any(re.match(pattern, key) for pattern in _SKIP_STATS):
                    continue
                summed_policy_stats[policy_idx][key] += float(value)

    transpired_episodes = len(per_episode_stats)

    return MissionEvaluationResult(
        mission_name=mission_name,
        policy_counts=policy_counts,
        policy_names=policy_names,
        summed_policy_stats=[dict(stats) for stats in summed_policy_stats],
        avg_game_stats=dict({key: value / transpired_episodes for key, value in summed_game_stats.items()}),
        per_episode_rewards=per_episode_rewards,
        per_episode_assignments=per_episode_assignments,
        per_policy_timeouts=per_policy_timeouts,
        episodes=transpired_episodes,
    )
