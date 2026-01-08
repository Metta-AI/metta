from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from cogames.leaderboard import allocate_counts, make_machina1_open_world_env
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout


@dataclass(frozen=True)
class PickupScenario:
    candidate_count: int
    pool_counts: list[int]

    @property
    def replacement_count(self) -> int:
        return sum(self.pool_counts)

    def scenario_name(self) -> str:
        return f"machina1-c{self.candidate_count}-r{self.replacement_count}"


@dataclass(frozen=True)
class PickupScenarioResult:
    scenario: PickupScenario
    candidate_mean: Optional[float]
    replacement_mean: Optional[float]
    replay_paths: list[str]


def pickup(
    console: Console,
    candidate_spec: PolicySpec,
    pool_specs: list[PolicySpec],
    *,
    num_cogs: int,
    episodes: int,
    seed: int,
    map_seed: Optional[int],
    steps: Optional[int],
    action_timeout_ms: int,
    save_replay_dir: Optional[Path],
) -> None:
    env_cfg = make_machina1_open_world_env(num_cogs=num_cogs, seed=seed, map_seed=map_seed, steps=steps)
    env_interface = PolicyEnvInterface.from_mg_cfg(env_cfg)
    policy_specs = [candidate_spec, *pool_specs]
    policies = [initialize_or_load_policy(env_interface, spec) for spec in policy_specs]

    scenarios: list[PickupScenario] = [
        PickupScenario(
            candidate_count=candidate_count,
            pool_counts=allocate_counts(num_cogs - candidate_count, [1.0] * len(pool_specs)),
        )
        for candidate_count in range(num_cogs, -1, -1)
    ]

    console.print("[bold cyan]Pickup Evaluation[/bold cyan]")
    console.print(f"[dim]Mission: machina_1.open_world | cogs={num_cogs} | episodes={episodes} | seed={seed}[/dim]")
    console.print(f"Candidate: [bold]{candidate_spec.name}[/bold]")
    console.print("Pool: " + ", ".join(f"{idx + 1}:{spec.name}" for idx, spec in enumerate(pool_specs)))

    results: list[PickupScenarioResult] = []
    replacement_mean: Optional[float] = None
    total_candidate_weighted_sum = 0.0
    total_candidate_agents = 0

    with typer.progressbar(scenarios, label="Simulating") as progress:
        for scenario in progress:
            rollout = multi_episode_rollout(
                env_cfg=env_cfg,
                policies=policies,
                proportions=[scenario.candidate_count, *scenario.pool_counts],
                episodes=episodes,
                seed=seed,
                max_action_time_ms=action_timeout_ms,
                save_replay=str(save_replay_dir) if save_replay_dir else None,
            )

            candidate_means: list[float] = []
            replacement_means: list[float] = []
            replay_paths: list[str] = []

            for episode in rollout.episodes:
                if episode.replay_path:
                    replay_paths.append(episode.replay_path)
                if episode.rewards.size == 0:
                    continue
                if scenario.candidate_count == 0:
                    replacement_means.append(float(episode.rewards.mean()))
                else:
                    mask = episode.assignments == 0
                    if np.any(mask):
                        candidate_means.append(float(episode.rewards[mask].mean()))

            candidate_mean = sum(candidate_means) / len(candidate_means) if candidate_means else None
            scenario_replacement_mean = sum(replacement_means) / len(replacement_means) if replacement_means else None
            results.append(
                PickupScenarioResult(
                    scenario=scenario,
                    candidate_mean=candidate_mean,
                    replacement_mean=scenario_replacement_mean,
                    replay_paths=replay_paths,
                )
            )

            if scenario.candidate_count == 0:
                replacement_mean = scenario_replacement_mean
            elif candidate_mean is not None:
                total_candidate_weighted_sum += candidate_mean * scenario.candidate_count * episodes
                total_candidate_agents += scenario.candidate_count * episodes

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario")
    table.add_column("Candidate", justify="right")
    table.add_column("Pool Mix", justify="right")
    table.add_column("Score", justify="right")

    for result in results:
        scenario = result.scenario
        pool_mix = "/".join(str(count) for count in scenario.pool_counts)
        if scenario.candidate_count == 0:
            score = result.replacement_mean
        else:
            score = result.candidate_mean
        score_text = f"{score:.2f}" if score is not None else "-"
        table.add_row(
            scenario.scenario_name(),
            str(scenario.candidate_count),
            pool_mix,
            score_text,
        )

    console.print("\n[bold cyan]Scenario Scores[/bold cyan]")
    console.print(table)

    if replacement_mean is None:
        console.print("[yellow]No replacement baseline available (missing c0 scenario).[/yellow]")
        if save_replay_dir:
            console.print(f"[dim]Replays saved to {save_replay_dir}[/dim]")
        return

    console.print(f"\n[bold cyan]Replacement Baseline[/bold cyan] {replacement_mean:.2f}")

    vor_table = Table(show_header=True, header_style="bold magenta")
    vor_table.add_column("Candidate Count", justify="right")
    vor_table.add_column("Candidate Score", justify="right")
    vor_table.add_column("VOR", justify="right")

    for result in results:
        scenario = result.scenario
        if scenario.candidate_count == 0:
            continue
        if result.candidate_mean is None:
            candidate_score = None
            vor = None
        else:
            candidate_score = result.candidate_mean
            vor = candidate_score - replacement_mean

        candidate_text = f"{candidate_score:.2f}" if candidate_score is not None else "-"
        vor_text = f"{vor:.2f}" if vor is not None else "-"
        vor_table.add_row(str(scenario.candidate_count), candidate_text, vor_text)

    console.print("\n[bold cyan]Value Over Replacement[/bold cyan]")
    console.print(vor_table)

    overall_vor = None
    if total_candidate_agents > 0:
        overall_candidate_avg = total_candidate_weighted_sum / total_candidate_agents
        overall_vor = overall_candidate_avg - replacement_mean

    if overall_vor is not None:
        console.print(f"\n[bold cyan]Overall VOR[/bold cyan] {overall_vor:.2f}")

    if pool_specs:
        pool_names = ", ".join(spec.name for spec in pool_specs)
        console.print(f"[dim]Pool order for mix column: {pool_names}[/dim]")

    if save_replay_dir:
        console.print(f"[dim]Replays saved to {save_replay_dir}[/dim]")
