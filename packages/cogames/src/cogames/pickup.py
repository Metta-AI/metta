from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from alo.multi_episode_runner import multi_episode_rollout
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from mettagrid import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def make_machina1_open_world_env(
    *, num_cogs: int, seed: Optional[int] = None, map_seed: Optional[int] = None, steps: Optional[int] = None
) -> MettaGridConfig:
    mission = Machina1OpenWorldMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env_cfg = mission.make_env()
    if steps is not None:
        env_cfg.game.max_steps = steps

    effective_map_seed = map_seed if map_seed is not None else seed
    if effective_map_seed is not None:
        map_builder = getattr(env_cfg.game, "map_builder", None)
        if isinstance(map_builder, MapGen.Config):
            map_builder.seed = effective_map_seed

    return env_cfg


def allocate_counts(total: int, weights: list[float]) -> list[int]:
    total_weight = sum(weights)
    if total_weight == 0:
        return [0] * len(weights)
    fractions = [weight / total_weight for weight in weights]
    ideals = [total * fraction for fraction in fractions]
    counts = [int(value) for value in ideals]
    remaining = total - sum(counts)

    remainders = [ideal - count for ideal, count in zip(ideals, counts, strict=True)]
    for i in sorted(range(len(remainders)), key=remainders.__getitem__, reverse=True)[:remaining]:
        counts[i] += 1
    return counts


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
    candidate_label: Optional[str] = None,
    pool_labels: Optional[list[str]] = None,
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
    candidate_display = candidate_label or candidate_spec.name
    pool_display = pool_labels or [spec.name for spec in pool_specs]

    console.print(f"Candidate: [bold]{candidate_display}[/bold]")
    console.print("Pool: " + ", ".join(f"{idx + 1}:{label}" for idx, label in enumerate(pool_display)))

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

            candidate_sum = 0.0
            candidate_count = 0
            replacement_sum = 0.0
            replacement_count = 0
            replay_paths: list[str] = []

            for episode in rollout.episodes:
                if episode.replay_path:
                    replay_paths.append(episode.replay_path)
                if episode.rewards.size == 0:
                    continue
                if scenario.candidate_count == 0:
                    replacement_sum += float(episode.rewards.mean())
                    replacement_count += 1
                else:
                    mask = episode.assignments == 0
                    if np.any(mask):
                        candidate_sum += float(episode.rewards[mask].mean())
                        candidate_count += 1

            candidate_mean = candidate_sum / candidate_count if candidate_count else None
            scenario_replacement_mean = replacement_sum / replacement_count if replacement_count else None
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
                total_candidate_weighted_sum += candidate_mean * scenario.candidate_count * candidate_count
                total_candidate_agents += scenario.candidate_count * candidate_count

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
