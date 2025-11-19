"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional, TypeAlias

import numpy as np
import typer
import yaml  # type: ignore[import]
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import (
    MultiEpisodeRolloutResult,
    _compute_policy_agent_counts,  # noqa: PLC2701
)
from mettagrid.simulator.multi_episode.summary import MultiEpisodeRolloutSummary, build_multi_episode_rollout_summaries
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import EpisodeStats

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
    save_replay: Optional[Path] = None,
    jobs: Optional[int] = None,
    parallel_policy: bool = False,
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

    # Determine parallelism level
    max_workers = jobs if jobs is not None and jobs > 0 else max(1, os.cpu_count() or 1)
    use_parallel = max_workers > 1 and episodes > 1

    mission_results: list[MultiEpisodeRolloutResult] = []
    all_replay_paths: list[str] = []

    for mission_name, env_cfg in missions:
        env_interface = PolicyEnvInterface.from_mg_cfg(env_cfg)
        policy_instances: list[MultiAgentPolicy] = [
            initialize_or_load_policy(env_interface, spec) for spec in policy_specs
        ]

        if parallel_policy:
            console.print("[cyan]Using per-agent subprocess wrapper (each agent in its own process)[/cyan]")

        if use_parallel:
            # Parallel episode execution
            rollout_payload = _evaluate_mission_parallel(
                console,
                mission_name,
                env_cfg,
                policy_instances,
                proportions,
                episodes,
                action_timeout_ms,
                seed,
                max_workers,
                save_replay,
                parallel_policy,
            )
        else:
            # Serial execution - use our own episode loop to support AgentPolicy-level wrapping
            rollout_payload = _evaluate_mission_serial(
                console,
                mission_name,
                env_cfg,
                policy_instances,
                proportions,
                episodes,
                action_timeout_ms,
                seed,
                save_replay,
                parallel_policy,
            )
        mission_results.append(rollout_payload)
        # Collect replay paths from this mission
        if rollout_payload.replay_paths:
            all_replay_paths.extend(rollout_payload.replay_paths)

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


def _run_single_episode(
    env_cfg: MettaGridConfig,
    policies: list[MultiAgentPolicy],
    proportions: list[float],
    episode_idx: int,
    episode_seed: int,
    max_action_time_ms: int,
    save_replay: Optional[Path],
    event_handlers: Optional[list],
    parallel_policy: bool = False,
) -> tuple[int, np.ndarray, EpisodeStats, np.ndarray, np.ndarray, list[str]]:
    """Run a single episode and return its results."""
    policy_counts = _compute_policy_agent_counts(
        env_cfg.game.num_agents, list(proportions) if proportions is not None else [1.0] * len(policies)
    )
    assignments = np.repeat(np.arange(len(policies)), policy_counts)
    rng = np.random.default_rng(episode_seed)
    rng.shuffle(assignments)

    # Create AgentPolicy instances, optionally wrapping them in subprocesses
    agent_policies: list[AgentPolicy] = []
    wrapped_agent_policies: list[AgentPolicy] = []  # Track for cleanup

    for agent_id in range(env_cfg.game.num_agents):
        policy_idx = assignments[agent_id]
        base_agent_policy = policies[policy_idx].agent_policy(agent_id)

        if parallel_policy:
            from cogames.policy.per_agent_subprocess_wrapper import wrap_agent_policy_in_subprocess

            wrapped_policy = wrap_agent_policy_in_subprocess(
                base_agent_policy,
                policies[policy_idx],
                agent_id,
            )
            agent_policies.append(wrapped_policy)
            wrapped_agent_policies.append(wrapped_policy)
        else:
            agent_policies.append(base_agent_policy)
    handlers = list(event_handlers or [])

    # Create a new replay writer for this episode if save_replay is provided
    episode_replay_writer = None
    if save_replay is not None:
        episode_replay_writer = ReplayLogWriter(str(save_replay))
        handlers.append(episode_replay_writer)

    rollout = Rollout(
        env_cfg,
        agent_policies,
        max_action_time_ms=max_action_time_ms,
        event_handlers=handlers,
    )

    rollout.run_until_done()

    rewards = np.array(rollout._sim.episode_rewards, dtype=float)
    stats = rollout._sim.episode_stats
    timeouts = np.array(rollout.timeout_counts, dtype=float)

    replay_paths = []
    if episode_replay_writer is not None:
        replay_paths.extend(episode_replay_writer.get_written_replay_paths())

    # Cleanup wrapped agent policies (shutdown their subprocesses)
    if parallel_policy:
        for wrapped_policy in wrapped_agent_policies:
            if hasattr(wrapped_policy, "shutdown"):
                wrapped_policy.shutdown()

    return episode_idx, rewards, stats, timeouts, assignments, replay_paths


def _evaluate_mission_serial(
    console: Console,
    mission_name: str,
    env_cfg: MettaGridConfig,
    policy_instances: list[MultiAgentPolicy],
    proportions: list[float],
    episodes: int,
    action_timeout_ms: int,
    base_seed: int,
    save_replay: Optional[Path],
    parallel_policy: bool = False,
) -> MultiEpisodeRolloutResult:
    """Evaluate a mission with serial episode execution."""
    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list[EpisodeStats] = []
    per_episode_assignments: list[np.ndarray] = []
    per_episode_timeouts: list[np.ndarray] = []
    all_replay_paths: list[str] = []

    # Use a separate RNG for episode seeds to ensure determinism
    seed_rng = np.random.default_rng(base_seed)
    episode_seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(episodes)]

    progress_label = f"Simulating ({mission_name})"
    with typer.progressbar(range(episodes), label=progress_label) as progress:
        for episode_idx in progress:
            _, rewards, stats, timeouts, assignments, replay_paths = _run_single_episode(
                env_cfg,
                policy_instances,
                proportions,
                episode_idx,
                episode_seeds[episode_idx],
                action_timeout_ms,
                save_replay,
                None,  # event_handlers
                parallel_policy,
            )
            per_episode_rewards.append(rewards)
            per_episode_stats.append(stats)
            per_episode_timeouts.append(timeouts)
            per_episode_assignments.append(assignments)
            all_replay_paths.extend(replay_paths)

    return MultiEpisodeRolloutResult(
        rewards=per_episode_rewards,
        stats=per_episode_stats,
        action_timeouts=per_episode_timeouts,
        assignments=per_episode_assignments,
        replay_paths=all_replay_paths,
    )


def _evaluate_mission_parallel(
    console: Console,
    mission_name: str,
    env_cfg: MettaGridConfig,
    policy_instances: list[MultiAgentPolicy],
    proportions: list[float],
    episodes: int,
    action_timeout_ms: int,
    base_seed: int,
    max_workers: int,
    save_replay: Optional[Path],
    parallel_policy: bool = False,
) -> MultiEpisodeRolloutResult:
    """Evaluate a mission with parallel episode execution."""
    console.print(f"[dim]Running {episodes} episodes in parallel (workers: {max_workers})[/dim]")

    per_episode_rewards: list[np.ndarray] = [None] * episodes  # type: ignore[list-item]
    per_episode_stats: list[EpisodeStats] = [None] * episodes  # type: ignore[list-item]
    per_episode_assignments: list[np.ndarray] = [None] * episodes  # type: ignore[list-item]
    per_episode_timeouts: list[np.ndarray] = [None] * episodes  # type: ignore[list-item]
    all_replay_paths: list[str] = []

    # Use a separate RNG for episode seeds to ensure determinism
    seed_rng = np.random.default_rng(base_seed)
    episode_seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(episodes)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_episode,
                env_cfg,
                policy_instances,
                proportions,
                episode_idx,
                episode_seeds[episode_idx],
                action_timeout_ms,
                save_replay,
                None,  # event_handlers - not used in parallel mode
                parallel_policy,
            ): episode_idx
            for episode_idx in range(episodes)
        }

        completed = 0
        with typer.progressbar(range(episodes), label=f"Simulating ({mission_name})") as progress:
            for future in as_completed(futures):
                episode_idx, rewards, stats, timeouts, assignments, replay_paths = future.result()
                per_episode_rewards[episode_idx] = rewards
                per_episode_stats[episode_idx] = stats
                per_episode_timeouts[episode_idx] = timeouts
                per_episode_assignments[episode_idx] = assignments
                all_replay_paths.extend(replay_paths)
                completed += 1
                progress.update(1)

    return MultiEpisodeRolloutResult(
        rewards=per_episode_rewards,  # type: ignore[arg-type]
        stats=per_episode_stats,  # type: ignore[arg-type]
        action_timeouts=per_episode_timeouts,  # type: ignore[arg-type]
        assignments=per_episode_assignments,  # type: ignore[arg-type]
        replay_paths=all_replay_paths,
    )


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
