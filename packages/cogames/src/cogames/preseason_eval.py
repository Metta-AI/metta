"""Preseason evaluation for CoGames policies.

This module provides evaluation against standardized baseline agents
in multiple scenarios to help assess policy performance.
"""

from __future__ import annotations

from typing import Literal, Optional

import typer
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries

# Baseline policies for preseason evaluation
BASELINE_1_NAME = "Ladybug"
BASELINE_1_CLASS = "nim_ladybug"  # cogames.policy.nim_agents.agents.LadyBugAgentsMultiPolicy
BASELINE_2_NAME = "Baseline"
BASELINE_2_CLASS = "baseline"  # cogames.policy.scripted_agent.baseline_agent.BaselinePolicy


def evaluate_preseason(
    console: Console,
    candidate_policy_spec: PolicySpec,
    episodes: int = 10,
    action_timeout_ms: int = 250,
    seed: int = 42,
    num_cogs: int = 4,
    steps: Optional[int] = None,
    output_format: Optional[Literal["yaml", "json"]] = None,
    save_replay_dir: Optional[str] = None,
) -> None:
    """Evaluate a candidate policy against preseason baselines.

    This runs three scenarios:
    1. Self-play: Candidate vs Candidate
    2. With Ladybug: 50% Candidate, 50% Ladybug
    3. With Baseline: 50% Candidate, 50% Baseline

    Args:
        console: Rich console for output
        candidate_policy_spec: The policy to evaluate
        episodes: Number of episodes per scenario
        action_timeout_ms: Max milliseconds per action
        seed: Random seed
        num_cogs: Number of agents (default: 4)
        steps: Max steps per episode (default: from mission)
        output_format: Optional output format (yaml/json)
        save_replay_dir: Optional directory to save replays
    """
    # Setup environment config
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env_cfg = mission.make_env()

    if steps is not None:
        env_cfg.game.max_steps = steps

    # Create baseline policy specs
    baseline1_spec = PolicySpec(class_path=BASELINE_1_CLASS)
    baseline2_spec = PolicySpec(class_path=BASELINE_2_CLASS)

    console.print("[cyan]Evaluating preseason scenarios on machina_1.open_world[/cyan]")
    console.print(f"Candidate: {candidate_policy_spec.name}")
    console.print(f"Episodes per scenario: {episodes}")
    console.print(f"Agents per episode: {num_cogs}\n")

    # Scenario definitions
    scenarios = [
        {
            "name": "Self-Play",
            "description": "Candidate vs Candidate",
            "policies": [candidate_policy_spec],
            "proportions": [1.0],
        },
        {
            "name": f"With {BASELINE_1_NAME}",
            "description": f"50% Candidate, 50% {BASELINE_1_NAME}",
            "policies": [candidate_policy_spec, baseline1_spec],
            "proportions": [1.0, 1.0],
        },
        {
            "name": f"With {BASELINE_2_NAME}",
            "description": f"50% Candidate, 50% {BASELINE_2_NAME}",
            "policies": [candidate_policy_spec, baseline2_spec],
            "proportions": [1.0, 1.0],
        },
    ]

    all_results = []

    for scenario in scenarios:
        console.print(f"[bold yellow]Scenario: {scenario['name']}[/bold yellow]")
        console.print(f"[dim]{scenario['description']}[/dim]")

        env_interface = PolicyEnvInterface.from_mg_cfg(env_cfg)
        policy_instances: list[MultiAgentPolicy] = [
            initialize_or_load_policy(env_interface, spec) for spec in scenario["policies"]
        ]

        progress_label = f"Running {scenario['name']}"
        progress_iterable = range(episodes)
        with typer.progressbar(progress_iterable, label=progress_label) as progress:
            iterator = iter(progress)

            def _progress_callback(_: int, progress_iter=iterator) -> None:
                try:
                    next(progress_iter)
                except StopIteration:
                    pass

            rollout_result = multi_episode_rollout(
                env_cfg=env_cfg,
                policies=policy_instances,
                proportions=scenario["proportions"],
                episodes=episodes,
                max_action_time_ms=action_timeout_ms,
                seed=seed,
                progress_callback=_progress_callback,
                save_replay=save_replay_dir,
            )

        all_results.append((scenario["name"], rollout_result))

    # Build summaries - each scenario needs its own summary with correct num_policies
    summaries = []
    for (_scenario_name, result), scenario_def in zip(all_results, scenarios, strict=True):
        num_policies_in_scenario = len(scenario_def["policies"])
        scenario_summary = build_multi_episode_rollout_summaries(
            [result],
            num_policies=num_policies_in_scenario,
        )[0]  # Get first (and only) summary
        summaries.append(scenario_summary)

    # Display results
    _display_preseason_results(console, candidate_policy_spec, scenarios, summaries, output_format)

    # Print replay info if replays were saved
    if save_replay_dir:
        total_replays = sum(len([ep for ep in result.episodes if ep.replay_path]) for _, result in all_results)
        if total_replays > 0:
            console.print(f"\n[bold cyan]Replays saved ({total_replays} episodes)![/bold cyan]")
            console.print("To watch a replay, run:")
            console.print("[bold green]cogames replay <replay_path>[/bold green]")


def _display_preseason_results(
    console: Console,
    candidate_spec: PolicySpec,
    scenarios: list[dict],
    summaries: list,
    output_format: Optional[Literal["yaml", "json"]],
) -> None:
    """Display preseason evaluation results."""
    if output_format:
        # TODO: Implement structured output
        console.print("[yellow]Structured output not yet implemented for preseason[/yellow]")
        return

    console.print("\n[bold cyan]Preseason Evaluation Results[/bold cyan]")
    console.print(f"Policy: {candidate_spec.name}\n")

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Episodes", justify="right")
    table.add_column("Avg Reward", justify="right", style="green")
    table.add_column("Min Reward", justify="right")
    table.add_column("Max Reward", justify="right")
    table.add_column("Timeouts", justify="right", style="yellow")

    for scenario, summary in zip(scenarios, summaries, strict=True):
        # Extract candidate policy stats (always policy 0)
        if summary.policy_summaries and summary.per_episode_per_policy_avg_rewards:
            policy_summary = summary.policy_summaries[0]

            # Calculate reward statistics from per-episode data
            candidate_rewards = [
                rewards[0]
                for rewards in summary.per_episode_per_policy_avg_rewards.values()
                if rewards and rewards[0] is not None
            ]

            if candidate_rewards:
                avg_reward = sum(candidate_rewards) / len(candidate_rewards)
                min_reward = min(candidate_rewards)
                max_reward = max(candidate_rewards)
            else:
                avg_reward = min_reward = max_reward = 0.0

            total_timeouts = policy_summary.action_timeouts

            table.add_row(
                scenario["name"],
                str(summary.episodes),
                f"{avg_reward:.2f}",
                f"{min_reward:.2f}",
                f"{max_reward:.2f}",
                str(total_timeouts),
            )

    console.print(table)

    # Print overall summary
    console.print("\n[bold]Preseason Score Components:[/bold]")
    console.print("• Self-Play: Tests policy consistency and cooperation")
    console.print(f"• With {BASELINE_1_NAME}: Tests adaptability to different play styles")
    console.print(f"• With {BASELINE_2_NAME}: Tests robustness across agent behaviors")
