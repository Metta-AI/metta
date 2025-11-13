"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import typer
import yaml  # type: ignore[import]
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode_rollout import MultiEpisodeRolloutResult, multi_episode_rollout

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]


class RawMissionEvaluationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Name of the mission that produced these stats
    mission_name: str
    # List of agent -> policy assignments for each episode
    # per_episode_assignments[episode_idx][agent_id] = policy_idx, where policy_idx is aligned with policy_names order.
    per_episode_assignments: list[np.ndarray]
    # Rewards returned by the environment for each episode
    # per_episode_rewards[episode_idx][agent_id] = reward achieved by agent_id in episode episode_idx.
    per_episode_rewards: list[np.ndarray]
    # List of action timeouts per episode for each policy
    # per_episode_timeouts[episode_idx][policy_idx] = number of timeouts for policy_idx in episode episode_idx.
    per_episode_timeouts: list[np.ndarray]


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
    mission_results: list[MultiEpisodeRolloutResult],
    mission_names: list[str],
    policy_specs: list[PolicySpec],
) -> MissionResultsSummary:
    if not mission_results:
        return MissionResultsSummary(missions=[])

    policy_names = [spec.name for spec in policy_specs]
    num_policies = len(policy_specs)
    summaries: list[MissionSummary] = []

    for mission_idx, mission_result in enumerate(mission_results):
        per_episode_assignments = mission_result.assignments
        policy_counts = (
            np.bincount(np.asarray(per_episode_assignments[0], dtype=int), minlength=num_policies)
            if per_episode_assignments
            else np.zeros(num_policies, dtype=int)
        )

        summed_game_stats: defaultdict[str, float] = defaultdict(float)
        summed_policy_stats: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(num_policies)]

        for stats, episode_assignments in zip(
            mission_result.stats,
            per_episode_assignments,
            strict=False,
        ):
            game_stats = stats.get("game", {})
            for key, value in game_stats.items():
                summed_game_stats[key] += float(value)

            agent_stats_list = stats.get("agent", [])
            for agent_id, agent_stats in enumerate(agent_stats_list):
                if agent_id >= len(episode_assignments):
                    continue
                policy_idx = int(episode_assignments[agent_id])
                for key, value in agent_stats.items():
                    if any(re.match(pattern, key) for pattern in _SKIP_STATS):
                        continue
                    summed_policy_stats[policy_idx][key] += float(value)

        transpired_episodes = len(mission_result.stats)
        if transpired_episodes:
            avg_game_stats = {key: value / transpired_episodes for key, value in summed_game_stats.items()}
        else:
            avg_game_stats = {}

        materialized_policy_stats = [dict(stats) for stats in summed_policy_stats]

        per_episode_per_policy_avg_rewards: dict[int, list[float | None]] = {}
        for episode_idx, (rewards, episode_assignments) in enumerate(
            zip(mission_result.rewards, per_episode_assignments, strict=False)
        ):
            per_policy_totals = np.zeros(num_policies, dtype=float)
            per_policy_counts = np.zeros(num_policies, dtype=int)
            for agent_id, reward in enumerate(rewards):
                if agent_id >= len(episode_assignments):
                    continue
                policy_idx = int(episode_assignments[agent_id])
                per_policy_totals[policy_idx] += float(reward)
                per_policy_counts[policy_idx] += 1
            per_episode_per_policy_avg_rewards[episode_idx] = [
                (per_policy_totals[i] / per_policy_counts[i]) if per_policy_counts[i] > 0 else None
                for i in range(num_policies)
            ]

        policy_summaries: list[MissionPolicySummary] = []
        for policy_idx, policy_name in enumerate(policy_names):
            agent_count = int(policy_counts[policy_idx]) if policy_idx < len(policy_counts) else 0
            average_metrics = (
                {key: value / agent_count for key, value in sorted(materialized_policy_stats[policy_idx].items())}
                if agent_count > 0
                else {}
            )
            action_timeouts = 0
            for episode_assignments, episode_timeouts in zip(
                per_episode_assignments,
                mission_result.action_timeouts,
                strict=False,
            ):
                for agent_index, timeout_count in enumerate(episode_timeouts):
                    if agent_index >= len(episode_assignments):
                        continue
                    if int(episode_assignments[agent_index]) == policy_idx:
                        action_timeouts += int(timeout_count)

            policy_summaries.append(
                MissionPolicySummary(
                    policy_name=policy_name,
                    agent_count=agent_count,
                    avg_agent_metrics=average_metrics,
                    action_timeouts=action_timeouts,
                )
            )

        summaries.append(
            MissionSummary(
                mission_name=mission_names[mission_idx],
                episodes=transpired_episodes,
                policy_summaries=policy_summaries,
                avg_game_stats=avg_game_stats,
                per_episode_per_policy_avg_rewards=per_episode_per_policy_avg_rewards,
            )
        )

    return MissionResultsSummary(missions=summaries)


def evaluate(
    console: Console,
    missions: list[tuple[str, MettaGridConfig]],
    policy_specs: list[PolicySpec],
    proportions: list[float],
    episodes: int,
    action_timeout_ms: int,
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

    mission_results: list[MultiEpisodeRolloutResult] = []
    for mission_name, env_cfg in missions:
        env_interface = PolicyEnvInterface.from_mg_cfg(env_cfg)
        policy_instances: list[MultiAgentPolicy] = [
            initialize_or_load_policy(env_interface, spec) for spec in policy_specs
        ]

        progress_label = f"Simulating ({mission_name})"
        progress_iterable = range(episodes)
        with typer.progressbar(progress_iterable, label=progress_label) as progress:
            iterator = iter(progress)

            def _progress_callback(_: int, progress_iter=iterator) -> None:
                try:
                    next(progress_iter)
                except StopIteration:
                    pass

            rollout_payload = multi_episode_rollout(
                env_cfg=env_cfg,
                policies=policy_instances,
                proportions=proportions,
                episodes=episodes,
                max_action_time_ms=action_timeout_ms,
                seed=seed,
                progress_callback=_progress_callback,
            )
        mission_results.append(rollout_payload)

    summary = _build_results_summary(mission_results, [mission_name for mission_name, _ in missions], policy_specs)
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
