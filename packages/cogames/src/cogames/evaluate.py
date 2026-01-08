"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Literal, Optional, TypeAlias

import numpy as np
import typer
import yaml  # type: ignore[import]
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult, multi_episode_rollout
from mettagrid.simulator.multi_episode.summary import MultiEpisodeRolloutSummary, build_multi_episode_rollout_summaries

MissionResultsSummary: TypeAlias = list[MultiEpisodeRolloutSummary]


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


def evaluate(
    console: Console,
    missions: list[tuple[str, MettaGridConfig]],
    policy_specs: list[PolicySpec],
    proportions: list[float],
    episodes: int,
    action_timeout_ms: int,
    seed: int = 42,
    output_format: Optional[Literal["yaml", "json"]] = None,
    save_replay: Optional[str] = None,
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
    all_replay_paths: list[str] = []
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
                save_replay=save_replay,
            )
        mission_results.append(rollout_payload)
        # Collect replay paths from this mission
        for episode in rollout_payload.episodes:
            if episode.replay_path:
                all_replay_paths.append(episode.replay_path)

    summaries = build_multi_episode_rollout_summaries(mission_results, num_policies=len(policy_specs))
    mission_names = [mission_name for mission_name, _ in missions]
    _output_results(console, policy_specs, mission_names, summaries, output_format)

    # Print replay commands if replays were saved
    if all_replay_paths:
        console.print(f"\n[bold cyan]Replays saved ({len(all_replay_paths)} episodes)![/bold cyan]")
        console.print("To watch a replay, run:")
        console.print("[bold green]cogames replay <replay_path>[/bold green]")
        console.print("\nExample:")
        console.print(f"[bold green]cogames replay {all_replay_paths[0]}[/bold green]")

    return summaries


def _output_results(
    console: Console,
    policy_specs: list[PolicySpec],
    mission_names: list[str],
    summaries: list[MultiEpisodeRolloutSummary],
    output_format: Optional[Literal["yaml", "json"]],
) -> None:
    mission_summaries = list(zip(mission_names, summaries, strict=True))
    if output_format:

        class _NamedMissionSummary(BaseModel):
            mission_name: str
            mission_summary: MultiEpisodeRolloutSummary

        class _ToDump(BaseModel):
            missions: list[_NamedMissionSummary]

        to_dump = _ToDump(
            missions=[
                _NamedMissionSummary(mission_name=mission_name, mission_summary=mission)
                for mission_name, mission in mission_summaries
            ]
        )

        if output_format == "json":
            serialized = json.dumps(to_dump.model_dump(mode="json"), indent=2)
        else:
            serialized = yaml.safe_dump(to_dump.model_dump(), sort_keys=False)
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
    for mission_name, mission in mission_summaries:
        for policy_idx, policy_summary in enumerate(mission.policy_summaries):
            assignment_table.add_row(
                mission_name,
                display_names[policy_idx],
                str(policy_summary.agent_count),
            )
    console.print(assignment_table)

    console.print("\n[bold cyan]Average Game Stats[/bold cyan]")
    game_stats_table = Table(show_header=True, header_style="bold magenta")
    game_stats_table.add_column("Mission")
    game_stats_table.add_column("Metric")
    game_stats_table.add_column("Average", justify="right")
    for mission_name, mission in mission_summaries:
        for key, value in mission.avg_game_stats.items():
            game_stats_table.add_row(mission_name, key, f"{value:.2f}")
    console.print(game_stats_table)

    console.print("\n[bold cyan]Average Policy Stats[/bold cyan]")
    for i, policy_name in enumerate(display_names):
        policy_table = Table(title=policy_name, show_header=True, header_style="bold magenta")
        policy_table.add_column("Mission")
        policy_table.add_column("Metric")
        policy_table.add_column("Average", justify="right")
        for mission_name, mission in mission_summaries:
            metrics = mission.policy_summaries[i].avg_agent_metrics
            if not metrics:
                policy_table.add_row(mission_name, "-", "-")
                continue
            for key, value in metrics.items():
                policy_table.add_row(mission_name, key, f"{value:.2f}")
        console.print(policy_table)

    console.print("\n[bold cyan]Average Per-Agent Reward [/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Mission")
    summary_table.add_column("Episode", justify="right")
    for display_name in display_names:
        summary_table.add_column(display_name, justify="right")

    for mission_name, mission in mission_summaries:
        for episode_idx, avg_rewards in sorted(mission.per_episode_per_policy_avg_rewards.items(), key=lambda x: x[0]):
            row = [mission_name, str(episode_idx)]
            row.extend((f"{value:.2f}" if value is not None else "-" for value in avg_rewards))
            summary_table.add_row(*row)

    console.print(summary_table)

    if any(policy.action_timeouts for mission in summaries for policy in mission.policy_summaries):
        console.print("\n[bold cyan]Action Generation Timeouts per Policy[/bold cyan]")
        timeouts_table = Table(show_header=True, header_style="bold magenta")
        timeouts_table.add_column("Mission")
        timeouts_table.add_column("Policy")
        timeouts_table.add_column("Timeouts", justify="right")
        for mission_name, mission in mission_summaries:
            for i, policy_summary in enumerate(mission.policy_summaries):
                if policy_summary.action_timeouts > 0:
                    timeouts_table.add_row(
                        mission_name,
                        display_names[i],
                        str(policy_summary.action_timeouts),
                    )
        console.print(timeouts_table)
