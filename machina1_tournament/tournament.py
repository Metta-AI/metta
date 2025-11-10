#!/usr/bin/env python
"""Tournament system for evaluating policies using value-of-replacement.

Takes a pool of N policies and runs simulations with randomly sampled teams.
Calculates value-of-replacement: how much better/worse games are when a policy participates.

Usage:
    # Run tournament with 16 random/scripted policies
    python metta/machina1_tournament/tournament.py --num-episodes 100

    # Run with specific policy pool file
    python metta/machina1_tournament/tournament.py --policies tournament_policies.yaml --num-episodes 200

    # Quick test with fewer policies
    python metta/machina1_tournament/tournament.py --num-episodes 20 --team-size 4 --pool-size 8
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.utils import initialize_or_load_policy
from mettagrid.simulator.rollout import Rollout

try:
    from cogames.missions import get_mission_config
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "cogames" / "src"))
    from cogames.missions import get_mission_config


@dataclass
class PolicyConfig:
    """Configuration for a single policy."""

    name: str
    class_path: str
    data_path: Optional[str] = None


@dataclass
class GameResult:
    """Result of a single game."""

    episode_id: int
    policy_indices: list[int]  # Which policies played (indices into policy pool)
    total_hearts: float  # Cumulative hearts across all agents
    per_agent_hearts: list[float]  # Hearts per agent
    steps: int


@dataclass
class PolicyStats:
    """Statistics for a single policy."""

    policy_idx: int
    policy_name: str
    games_played: int
    total_hearts_when_playing: float  # Sum of total_hearts for games this policy was in
    mean_hearts_when_playing: float  # Average total_hearts when this policy played
    value_of_replacement: float  # mean_hearts_when_playing - overall_mean
    agent_positions: list[int]  # Which agent positions this policy played


def load_policy_pool(
    policy_pool_file: Optional[Path],
    pool_size: int,
    console: Console,
) -> list[PolicyConfig]:
    """Load or generate a pool of policies."""
    if policy_pool_file and policy_pool_file.exists():
        console.print(f"[cyan]Loading policy pool from {policy_pool_file}[/cyan]")
        with open(policy_pool_file) as f:
            if policy_pool_file.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        policies = [
            PolicyConfig(
                name=p["name"],
                class_path=p["class_path"],
                data_path=p.get("data_path"),
            )
            for p in data["policies"]
        ]
        return policies
    else:
        # Generate default pool with random and scripted agents
        console.print(f"[cyan]Generating default pool of {pool_size} policies[/cyan]")
        policies = []

        # Policy class templates
        templates = [
            ("mettagrid.policy.random.RandomPolicy", None, "Random"),
            (
                "cogames.policy.scripted_agent.baseline_agent.BaselineAgent",
                None,
                "Baseline",
            ),
            (
                "cogames.policy.scripted_agent.unclipping_agent.UnclippingAgent",
                None,
                "Unclipping",
            ),
        ]

        for i in range(pool_size):
            template = templates[i % len(templates)]
            policies.append(
                PolicyConfig(
                    name=f"{template[2]}_{i}",
                    class_path=template[0],
                    data_path=template[1],
                )
            )

        return policies


def run_tournament(
    mission: str,
    policy_pool: list[PolicyConfig],
    num_episodes: int,
    team_size: int,
    seed: int,
    console: Console,
    render: bool = False,
    max_steps: int = 1000,
) -> tuple[list[GameResult], list[PolicyStats]]:
    """Run the tournament and calculate value-of-replacement."""
    # Load mission
    console.print("\n[bold cyan]Tournament Configuration[/bold cyan]")
    console.print(f"Mission: {mission}")
    console.print(f"Policy Pool Size: {len(policy_pool)}")
    console.print(f"Team Size: {team_size}")
    console.print(f"Number of Episodes: {num_episodes}")
    console.print(f"Seed: {seed}")

    env_cfg = get_mission_config(mission)

    if env_cfg.game.num_agents != team_size:
        console.print(
            f"[yellow]Warning: Mission has {env_cfg.game.num_agents} agents, "
            f"but team_size is {team_size}. Adjusting mission config.[/yellow]"
        )
        env_cfg.game.num_agents = team_size

    # Initialize all policies
    console.print(f"\n[cyan]Initializing {len(policy_pool)} policies...[/cyan]")
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

    policy_instances = []
    for i, policy_cfg in enumerate(policy_pool):
        try:
            policy = initialize_or_load_policy(
                policy_env_info,
                policy_cfg.class_path,
                policy_cfg.data_path,
            )
            policy_instances.append(policy)
            console.print(f"  ✓ {i}: {policy_cfg.name}")
        except Exception as e:
            console.print(f"  ✗ {i}: {policy_cfg.name} - Error: {e}")
            raise

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Run episodes
    console.print(f"\n[cyan]Running {num_episodes} episodes...[/cyan]")
    game_results = []

    render_mode = "gui" if render else None

    for episode_id in track(range(num_episodes), description="Running episodes"):
        # Sample team_size policies without replacement
        sampled_indices = random.sample(range(len(policy_pool)), team_size)

        # Create agent policies
        agent_policies = []
        for agent_id in range(team_size):
            policy_idx = sampled_indices[agent_id]
            policy = policy_instances[policy_idx]
            agent_policy = policy.agent_policy(agent_id)
            agent_policies.append(agent_policy)

        # Run rollout
        rollout = Rollout(
            env_cfg,
            agent_policies,
            max_action_time_ms=10000,
            render_mode=render_mode,
            seed=seed + episode_id,
            pass_sim_to_policies=True,
        )

        step_count = 0
        while not rollout.is_done() and step_count < max_steps:
            rollout.step()
            step_count += 1

        # Extract hearts from episode stats
        episode_stats = rollout._sim.episode_stats
        game_stats = episode_stats.get("game", {})
        total_hearts = float(game_stats.get("chest.heart.amount", 0.0))

        # Get per-agent hearts
        agent_stats_list = episode_stats.get("agent", [])
        per_agent_hearts = []
        for agent_stats in agent_stats_list:
            # Sum all heart-related stats for this agent
            agent_heart_stats = sum(v for k, v in agent_stats.items() if "heart" in k.lower())
            per_agent_hearts.append(float(agent_heart_stats))

        game_results.append(
            GameResult(
                episode_id=episode_id,
                policy_indices=sampled_indices,
                total_hearts=total_hearts,
                per_agent_hearts=per_agent_hearts,
                steps=step_count,
            )
        )

    # Calculate statistics
    console.print("\n[cyan]Calculating value-of-replacement...[/cyan]")

    # Overall mean hearts
    overall_mean_hearts = np.mean([r.total_hearts for r in game_results])

    # Per-policy statistics
    policy_game_tracking = defaultdict(lambda: {"games": [], "positions": []})

    for result in game_results:
        for agent_pos, policy_idx in enumerate(result.policy_indices):
            policy_game_tracking[policy_idx]["games"].append(result.total_hearts)
            policy_game_tracking[policy_idx]["positions"].append(agent_pos)

    policy_stats = []
    for policy_idx in range(len(policy_pool)):
        if policy_idx in policy_game_tracking:
            games = policy_game_tracking[policy_idx]["games"]
            positions = policy_game_tracking[policy_idx]["positions"]
            total_hearts = sum(games)
            mean_hearts = np.mean(games)
            value_of_replacement = mean_hearts - overall_mean_hearts
        else:
            # Policy never played
            total_hearts = 0.0
            mean_hearts = 0.0
            value_of_replacement = 0.0
            positions = []

        policy_stats.append(
            PolicyStats(
                policy_idx=policy_idx,
                policy_name=policy_pool[policy_idx].name,
                games_played=len(
                    policy_game_tracking[policy_idx]["games"] if policy_idx in policy_game_tracking else []
                ),
                total_hearts_when_playing=total_hearts,
                mean_hearts_when_playing=mean_hearts,
                value_of_replacement=value_of_replacement,
                agent_positions=positions,
            )
        )

    # Sort by value of replacement
    policy_stats.sort(key=lambda x: x.value_of_replacement, reverse=True)

    return game_results, policy_stats


def display_results(
    game_results: list[GameResult],
    policy_stats: list[PolicyStats],
    console: Console,
) -> None:
    """Display tournament results."""
    console.print("\n[bold green]Tournament Results[/bold green]")

    # Overall statistics
    total_hearts = sum(r.total_hearts for r in game_results)
    mean_hearts = np.mean([r.total_hearts for r in game_results])
    std_hearts = np.std([r.total_hearts for r in game_results])

    console.print("\n[cyan]Overall Statistics[/cyan]")
    console.print(f"  Total Episodes: {len(game_results)}")
    console.print(f"  Total Hearts: {total_hearts:.1f}")
    console.print(f"  Mean Hearts/Game: {mean_hearts:.2f}")
    console.print(f"  Std Hearts/Game: {std_hearts:.2f}")

    # Policy rankings
    console.print("\n[bold cyan]Policy Rankings (by Value of Replacement)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Policy Name", style="white")
    table.add_column("Games", justify="right")
    table.add_column("Mean Hearts", justify="right")
    table.add_column("Value of Replacement", justify="right")
    table.add_column("Positions", justify="left")

    for rank, stats in enumerate(policy_stats, 1):
        # Color code value of replacement
        vor = stats.value_of_replacement
        if vor > 0:
            vor_str = f"[green]+{vor:.2f}[/green]"
        elif vor < 0:
            vor_str = f"[red]{vor:.2f}[/red]"
        else:
            vor_str = f"{vor:.2f}"

        # Summarize positions
        if stats.agent_positions:
            pos_counts = [stats.agent_positions.count(i) for i in range(max(stats.agent_positions) + 1)]
            pos_str = ",".join(str(c) for c in pos_counts)
        else:
            pos_str = "-"

        table.add_row(
            str(rank),
            stats.policy_name,
            str(stats.games_played),
            f"{stats.mean_hearts_when_playing:.2f}",
            vor_str,
            pos_str,
        )

    console.print(table)


def save_results(
    game_results: list[GameResult],
    policy_stats: list[PolicyStats],
    output_file: Path,
    console: Console,
) -> None:
    """Save tournament results to file."""
    results = {
        "tournament_summary": {
            "num_episodes": len(game_results),
            "overall_mean_hearts": np.mean([r.total_hearts for r in game_results]),
            "overall_std_hearts": np.std([r.total_hearts for r in game_results]),
        },
        "policy_rankings": [
            {
                "rank": i + 1,
                "policy_idx": stats.policy_idx,
                "policy_name": stats.policy_name,
                "games_played": stats.games_played,
                "mean_hearts_when_playing": stats.mean_hearts_when_playing,
                "value_of_replacement": stats.value_of_replacement,
            }
            for i, stats in enumerate(policy_stats)
        ],
        "game_results": [
            {
                "episode_id": r.episode_id,
                "policy_indices": r.policy_indices,
                "total_hearts": r.total_hearts,
                "steps": r.steps,
            }
            for r in game_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/green]")


def main(
    mission: str = typer.Option("machina_1", "--mission", "-m", help="Mission name"),
    num_episodes: int = typer.Option(100, "--num-episodes", "-n", help="Number of tournament episodes"),
    team_size: int = typer.Option(4, "--team-size", "-t", help="Number of agents per team"),
    pool_size: int = typer.Option(16, "--pool-size", "-p", help="Size of policy pool (if generating default)"),
    policies: Optional[Path] = typer.Option(None, "--policies", help="Path to policy pool YAML/JSON file"),  # noqa: B008
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    render: bool = typer.Option(False, "--render", "-r", help="Render games (slow)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results (JSON)"),  # noqa: B008
    max_steps: int = typer.Option(1000, "--max-steps", help="Max steps per episode"),
) -> None:
    """Run a tournament to evaluate policies using value-of-replacement metric."""
    console = Console()

    console.print("[bold cyan]Policy Tournament System[/bold cyan]")
    console.print("Value of Replacement = Mean Hearts when Playing - Overall Mean\n")

    # Load policy pool
    policy_pool = load_policy_pool(policies, pool_size, console)

    if len(policy_pool) < team_size:
        console.print(f"[red]Error: Policy pool size ({len(policy_pool)}) must be >= team size ({team_size})[/red]")
        raise typer.Exit(1)

    # Run tournament
    game_results, policy_stats = run_tournament(
        mission=mission,
        policy_pool=policy_pool,
        num_episodes=num_episodes,
        team_size=team_size,
        seed=seed,
        console=console,
        render=render,
        max_steps=max_steps,
    )

    # Display results
    display_results(game_results, policy_stats, console)

    # Save results if requested
    if output:
        save_results(game_results, policy_stats, output, console)


if __name__ == "__main__":
    typer.run(main)
