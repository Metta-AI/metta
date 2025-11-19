#!/usr/bin/env python
"""Train an agent on CoGs vs Clips missions, then run tournament on the same mission.

This script:
1. Trains a ViT policy agent on a specific CoGs vs Clips eval mission (e.g., extractor_hub_30)
2. Saves checkpoints at regular intervals (default every 10 epochs)
3. Runs tournament with all checkpoints + baseline policies on the SAME mission

Usage:
    # Train and run tournament on default mission (extractor_hub_30)
    python metta/machina1_tournament/train_and_tourney.py --num-episodes 100

    # Train on a specific mission
    python metta/machina1_tournament/train_and_tourney.py --mission collect_resources_classic --num-episodes 100
    
    # Just run tournament with existing checkpoints
    python metta/machina1_tournament/train_and_tourney.py \\
        --skip-training --run relh.main.1110.2 --mission extractor_hub_30
"""

import json
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table

from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.runner import SimulationRunConfig, run_simulations
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

try:
    from cogames.cli.mission import get_mission
    from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "cogames" / "src"))
    from cogames.cli.mission import get_mission
    from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS


def get_mission_identifier(mission_name: str) -> str:
    """Convert mission name to proper identifier for get_mission().

    For eval missions (like 'extractor_hub_30'), prepends 'evals.' site prefix.
    For other missions, returns as-is (they should already be in site.mission format).
    """
    # Check if it's an eval mission
    eval_mission_names = {m.name for m in EVAL_MISSIONS}
    if mission_name in eval_mission_names:
        return f"evals.{mission_name}"

    # If it already contains a dot, assume it's in site.mission format
    if "." in mission_name:
        return mission_name

    # Otherwise, assume it's an eval mission
    return f"evals.{mission_name}"


@dataclass
class PolicyConfig:
    """Configuration for a single policy."""

    name: str
    class_path: Optional[str] = None  # Optional for .mpt files which embed architecture
    data_path: Optional[str] = None


@dataclass
class GameResult:
    """Result of a single game."""

    episode_id: int
    policy_indices: list[int]
    total_hearts: float
    per_agent_hearts: list[float]
    steps: int


@dataclass
class PolicyStats:
    """Statistics for a single policy."""

    policy_idx: int
    policy_name: str
    games_played: int
    total_hearts_when_playing: float
    mean_hearts_when_playing: float
    value_of_replacement: float
    agent_positions: list[int]


@dataclass
class SelfPlayStats:
    """Self-play statistics for a single policy."""

    policy_idx: int
    policy_name: str
    mean_reward: float
    std_reward: float
    episodes_played: int


def train_with_checkpoints(
    checkpoint_dir: Path,
    max_epochs: int,
    checkpoint_interval: int,
    num_cogs: int,
    mission: str,
    console: Console,
) -> list[Path]:
    """Train agent and save checkpoints at specified intervals.

    Returns list of checkpoint paths.
    """
    console.print("\n[bold cyan]Training Agent with Checkpoints[/bold cyan]")
    console.print(f"Mission: {mission}")
    console.print(f"Training for {max_epochs} epochs")
    console.print(f"Checkpoints every {checkpoint_interval} epochs")
    console.print(f"Number of cogs: {num_cogs}")
    console.print(f"Output directory: {checkpoint_dir}")

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total timesteps based on epochs
    # Each epoch is ~1024 steps (batch_size / num_envs)
    # For small maps recipe with default settings, epochs are small
    timesteps_per_epoch = 1024  # Actual steps per epoch for this configuration
    total_timesteps = max_epochs * timesteps_per_epoch

    # Build training command using the recipe system
    # Train on a single specific mission to match evaluation
    # Note: checkpointer.epoch_interval controls checkpoint frequency
    # Device is controlled by system.device, not trainer.device
    cmd = [
        "uv",
        "run",
        "./tools/run.py",
        "recipes.prod.cvc.fixed_maps.train",
        f"run={checkpoint_dir.name}",
        f"num_cogs={num_cogs}",
        f"mission={mission}",  # Train on specific mission
        'variants=["lonely_heart","heart_chorus","pack_rat"]',
        f"trainer.total_timesteps={total_timesteps}",
        f"checkpointer.epoch_interval={checkpoint_interval}",
        "evaluator.evaluate_local=true",  # Enable local evaluation
    ]

    console.print(f"\n[yellow]Running: {' '.join(cmd)}[/yellow]\n")

    # Run training
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Training failed with error: {e}[/red]")
        raise

    # Find all checkpoint files
    # The recipe system saves to train_dir/{run_name}/checkpoints/
    train_dir = Path("./train_dir")
    actual_checkpoint_dir = train_dir / checkpoint_dir.name / "checkpoints"

    console.print(f"[cyan]Looking for checkpoints in: {actual_checkpoint_dir}[/cyan]")

    # List what's actually in train_dir for debugging
    if train_dir.exists():
        console.print(f"[cyan]Contents of {train_dir}:[/cyan]")
        for item in sorted(train_dir.iterdir()):
            if item.is_dir():
                console.print(f"  📁 {item.name}/")
                # Check if it has a checkpoints subdirectory
                checkpoints_subdir = item / "checkpoints"
                if checkpoints_subdir.exists():
                    num_checkpoints = len(list(checkpoints_subdir.glob("*.pt")))
                    console.print(f"     └─ checkpoints/ ({num_checkpoints} .pt files)")

    if not actual_checkpoint_dir.exists():
        console.print(f"[red]Error: Checkpoint directory {actual_checkpoint_dir} does not exist![/red]")
        console.print("[red]Training may have failed or saved checkpoints to a different location.[/red]")
        return []

    # Look for policy checkpoint files (.mpt), not trainer state files (.pt)
    # Explicitly exclude trainer_state.pt
    all_mpt_files = sorted(actual_checkpoint_dir.glob("*.mpt"))
    checkpoints = [f for f in all_mpt_files if f.name != "trainer_state.mpt"]

    if not checkpoints:
        console.print("[yellow]Warning: No policy checkpoints found after training![/yellow]")
        console.print(f"[yellow]Searched in: {actual_checkpoint_dir}[/yellow]")
        console.print(f"[yellow]Found {len(all_mpt_files)} .mpt files total, but none were policy checkpoints[/yellow]")

        # List what we did find for debugging
        if all_mpt_files:
            console.print("[cyan]Files found:[/cyan]")
            for f in all_mpt_files:
                console.print(f"  - {f.name}")
        return []

    console.print(f"\n[green]Training complete! Found {len(checkpoints)} policy checkpoints[/green]")
    for cp in checkpoints:
        console.print(f"  ✓ {cp.name}")

    # Select checkpoints at desired intervals
    # For simplicity, take the last 5 checkpoints if we have that many
    selected_checkpoints = checkpoints[-min(5, len(checkpoints)) :]

    return selected_checkpoints


def build_policy_pool(
    checkpoint_paths: list[Path],
    console: Console,
) -> list[PolicyConfig]:
    """Build policy pool including trained checkpoints and baselines."""
    policies = []

    # Add trained checkpoint policies
    # For .mpt files, don't specify class_path - it's embedded in the checkpoint
    for i, checkpoint_path in enumerate(checkpoint_paths):
        policies.append(
            PolicyConfig(
                name=f"Trained_CP{i + 1}",
                class_path=None,  # Architecture is embedded in .mpt file
                data_path=str(checkpoint_path),
            )
        )

    # Add baseline policies for comparison
    baseline_templates = [
        ("mettagrid.policy.random_agent.RandomMultiAgentPolicy", None, "Random"),
        ("cogames.policy.scripted_agent.baseline_agent.BaselinePolicy", None, "Baseline"),
        ("cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy", None, "Unclipping"),
        ("cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy", None, "Thinky"),
        ("cogames.policy.nim_agents.agents.RaceCarAgentsMultiPolicy", None, "RaceCar"),
        ("cogames.policy.nim_agents.agents.LadyBugAgentsMultiPolicy", None, "LadyBug"),
        ("cogames.policy.nim_agents.agents.RandomAgentsMultiPolicy", None, "NimRandom"),
    ]

    # Add a few of each baseline
    for i in range(2):  # 2 of each baseline type
        for class_path, data_path, name in baseline_templates:
            policies.append(
                PolicyConfig(
                    name=f"{name}_{i + 1}",
                    class_path=class_path,
                    data_path=data_path,
                )
            )

    console.print(f"\n[cyan]Policy Pool ({len(policies)} policies):[/cyan]")
    for i, policy in enumerate(policies):
        console.print(f"  {i}: {policy.name}")

    return policies


def run_tournament(
    mission: str,
    policy_pool: list[PolicyConfig],
    num_episodes: int,
    team_size: int,
    seed: int,
    console: Console,
    max_steps: int = 1000,
    variants: Optional[list[str]] = None,
) -> tuple[list[GameResult], list[PolicyStats], list]:
    """Run tournament on CoGs vs Clips missions."""
    console.print("\n[bold cyan]Tournament Configuration[/bold cyan]")
    console.print(f"Mission: {mission}")
    console.print(f"Policy Pool Size: {len(policy_pool)}")
    console.print(f"Team Size: {team_size}")
    console.print(f"Number of Episodes: {num_episodes}")

    # Get mission config once to initialize policies (don't pass variants_arg - mission has them)
    mission_id = get_mission_identifier(mission)
    _, env_cfg_template, _ = get_mission(mission_id, variants_arg=None)

    if env_cfg_template.game.num_agents != team_size:
        env_cfg_template.game.num_agents = team_size

    # Initialize policies
    console.print(f"\n[cyan]Initializing {len(policy_pool)} policies...[/cyan]")
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg_template)

    policy_instances = []
    for i, policy_cfg in enumerate(policy_pool):
        try:
            # For .mpt files (PolicyArtifact), use CheckpointManager which loads the embedded architecture
            if policy_cfg.class_path is None and policy_cfg.data_path is not None:
                checkpoint_path = Path(policy_cfg.data_path)
                if checkpoint_path.suffix == ".mpt":
                    # Load PolicyArtifact with embedded architecture
                    # CheckpointManager returns a metta.agent.policy.Policy which already
                    # implements agent_policy() method and is compatible with Rollout
                    uri = f"file://{checkpoint_path.resolve()}"
                    policy = CheckpointManager.load_from_uri(uri, policy_env_info, torch.device("cpu"))
                else:
                    raise ValueError(f"class_path is None but data_path is not .mpt: {policy_cfg.data_path}")
            else:
                # For regular policies (non-.mpt), use the standard loading mechanism
                policy = initialize_or_load_policy(
                    policy_env_info,
                    PolicySpec(
                        class_path=policy_cfg.class_path,
                        data_path=policy_cfg.data_path,
                    ),
                )

            policy_instances.append(policy)
            console.print(f"  ✓ {i}: {policy_cfg.name}")
        except Exception as e:
            console.print(f"  ✗ {i}: {policy_cfg.name} - Error: {e}")
            raise

    random.seed(seed)
    np.random.seed(seed)

    # Get fresh env_cfg for tournament
    _, env_cfg, _ = get_mission(mission_id, variants_arg=None)
    if env_cfg.game.num_agents != team_size:
        env_cfg.game.num_agents = team_size

    # Run episodes using run_simulations
    console.print(f"\n[cyan]Running {num_episodes} episodes with run_simulations...[/cyan]")

    # Use uniform proportions for all policies
    proportions = [1.0] * len(policy_instances)

    def progress_callback(msg: str) -> None:
        console.print(f"  {msg}")

    # Create simulation configuration
    simulation_config = SimulationRunConfig(
        env=env_cfg,
        num_episodes=num_episodes,
        proportions=proportions,
        max_action_time_ms=10000,
    )

    # Create policy initializers that return the already-initialized instances
    def make_policy_initializer(policy_instance):
        def initializer(env_interface):
            return policy_instance

        return initializer

    policy_initializers = [make_policy_initializer(policy) for policy in policy_instances]

    # Run simulations
    simulation_results = run_simulations(
        policy_initializers=policy_initializers,
        simulations=[simulation_config],
        replay_dir=None,  # No replay recording for tournament
        seed=seed,
        on_progress=progress_callback,
    )

    # Extract the rollout result from the first simulation
    rollout_result = simulation_results[0].results

    # Convert multi_episode_rollout results to GameResult format
    game_results = []
    for episode_id, episode in enumerate(rollout_result.episodes):
        episode_stats = episode.stats
        episode_assignments = episode.assignments

        game_stats = episode_stats.get("game", {})
        total_hearts = float(game_stats.get("chest.heart.amount", 0.0))

        agent_stats_list = episode_stats.get("agent", [])
        per_agent_hearts = []
        for agent_stats in agent_stats_list:
            agent_heart_stats = sum(v for k, v in agent_stats.items() if "heart" in k.lower())
            per_agent_hearts.append(float(agent_heart_stats))

        game_results.append(
            GameResult(
                episode_id=episode_id,
                policy_indices=episode_assignments.tolist(),
                total_hearts=total_hearts,
                per_agent_hearts=per_agent_hearts,
                steps=0,  # multi_episode_rollout doesn't track steps
            )
        )

    # Calculate statistics
    console.print("\n[cyan]Calculating value-of-replacement...[/cyan]")
    overall_mean_hearts = np.mean([r.total_hearts for r in game_results])

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
            total_hearts = 0.0
            mean_hearts = 0.0
            value_of_replacement = 0.0
            positions = []

        games_count = len(policy_game_tracking[policy_idx]["games"]) if policy_idx in policy_game_tracking else 0

        policy_stats.append(
            PolicyStats(
                policy_idx=policy_idx,
                policy_name=policy_pool[policy_idx].name,
                games_played=games_count,
                total_hearts_when_playing=total_hearts,
                mean_hearts_when_playing=mean_hearts,
                value_of_replacement=value_of_replacement,
                agent_positions=positions,
            )
        )

    policy_stats.sort(key=lambda x: x.value_of_replacement, reverse=True)

    return game_results, policy_stats, policy_instances


def run_self_play(
    mission: str,
    policy_pool: list[PolicyConfig],
    policy_instances: list,
    team_size: int,
    seed: int,
    console: Console,
    num_self_play_episodes: int = 5,
    max_steps: int = 1000,
    variants: Optional[list[str]] = None,
) -> list[SelfPlayStats]:
    """Run self-play episodes for each policy.

    Each policy plays against copies of itself to measure pure self-play performance.
    """
    # Don't pass variants_arg if using a mission that already has them
    # Eval missions already have all the variants baked in
    console.print("\n[bold cyan]Self-Play Evaluation[/bold cyan]")
    console.print(f"Mission: {mission}")
    console.print(f"Episodes per policy: {num_self_play_episodes}")

    mission_id = get_mission_identifier(mission)
    self_play_stats = []

    for policy_idx, (policy_cfg, policy) in enumerate(zip(policy_pool, policy_instances, strict=True)):
        console.print(f"\n[cyan]Self-play for {policy_cfg.name}...[/cyan]")
        episode_rewards = []

        for episode_id in range(num_self_play_episodes):
            # Create fresh env_cfg for each episode to avoid state pollution
            # Don't pass variants_arg - the mission already has them
            _, env_cfg, _ = get_mission(mission_id, variants_arg=None)

            if env_cfg.game.num_agents != team_size:
                env_cfg.game.num_agents = team_size

            # All agents use the same policy
            agent_policies = [policy.agent_policy(agent_id) for agent_id in range(team_size)]

            rollout = Rollout(
                env_cfg,
                agent_policies,
                max_action_time_ms=10000,
                render_mode=None,
                seed=seed + policy_idx * 1000 + episode_id,
                pass_sim_to_policies=True,
            )

            step_count = 0
            while not rollout.is_done() and step_count < max_steps:
                rollout.step()
                step_count += 1

            episode_stats = rollout._sim.episode_stats
            game_stats = episode_stats.get("game", {})
            total_hearts = float(game_stats.get("chest.heart.amount", 0.0))
            episode_rewards.append(total_hearts)

            # Debug logging
            console.print(f"    Episode {episode_id + 1}: {total_hearts:.2f} hearts in {step_count} steps")

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        self_play_stats.append(
            SelfPlayStats(
                policy_idx=policy_idx,
                policy_name=policy_cfg.name,
                mean_reward=mean_reward,
                std_reward=std_reward,
                episodes_played=num_self_play_episodes,
            )
        )

        console.print(
            f"  ✓ {policy_cfg.name}: mean={mean_reward:.2f}, std={std_reward:.2f} (n={num_self_play_episodes})"
        )

    # Sort by mean reward
    self_play_stats.sort(key=lambda x: x.mean_reward, reverse=True)

    return self_play_stats


def display_results(
    game_results: list[GameResult],
    policy_stats: list[PolicyStats],
    self_play_stats: list[SelfPlayStats],
    console: Console,
) -> None:
    """Display tournament and self-play results."""
    console.print("\n[bold green]Tournament Results[/bold green]")

    total_hearts = sum(r.total_hearts for r in game_results)
    mean_hearts = np.mean([r.total_hearts for r in game_results])
    std_hearts = np.std([r.total_hearts for r in game_results])

    console.print("\n[cyan]Overall Statistics[/cyan]")
    console.print(f"  Total Episodes: {len(game_results)}")
    console.print(f"  Total Hearts: {total_hearts:.1f}")
    console.print(f"  Mean Hearts/Game: {mean_hearts:.2f}")
    console.print(f"  Std Hearts/Game: {std_hearts:.2f}")

    console.print("\n[bold cyan]Policy Rankings (by Value of Replacement)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Policy Name", style="white")
    table.add_column("Games", justify="right")
    table.add_column("Mean Hearts", justify="right")
    table.add_column("Value of Replacement", justify="right")

    for rank, stats in enumerate(policy_stats, 1):
        vor = stats.value_of_replacement
        if vor > 0:
            vor_str = f"[green]+{vor:.2f}[/green]"
        elif vor < 0:
            vor_str = f"[red]{vor:.2f}[/red]"
        else:
            vor_str = f"{vor:.2f}"

        table.add_row(
            str(rank),
            stats.policy_name,
            str(stats.games_played),
            f"{stats.mean_hearts_when_playing:.2f}",
            vor_str,
        )

    console.print(table)

    # Display self-play results
    console.print("\n[bold cyan]Self-Play Performance[/bold cyan]")
    sp_table = Table(show_header=True, header_style="bold magenta")
    sp_table.add_column("Rank", justify="right", style="cyan")
    sp_table.add_column("Policy Name", style="white")
    sp_table.add_column("Mean Reward", justify="right")
    sp_table.add_column("Std Dev", justify="right")
    sp_table.add_column("Episodes", justify="right")

    for rank, stats in enumerate(self_play_stats, 1):
        sp_table.add_row(
            str(rank),
            stats.policy_name,
            f"{stats.mean_reward:.2f}",
            f"{stats.std_reward:.2f}",
            str(stats.episodes_played),
        )

    console.print(sp_table)


def main(
    skip_training: bool = typer.Option(False, "--skip-training", help="Skip training, use existing checkpoints"),
    run: Optional[str] = typer.Option(
        None,
        "--run",
        help="Run name for training checkpoint directory (defaults to mission name)",
    ),
    max_epochs: int = typer.Option(146500, "--max-epochs", help="Maximum training epochs (~150M steps)"),
    checkpoint_interval: int = typer.Option(50, "--checkpoint-interval", help="Save checkpoint every N epochs"),
    num_episodes: int = typer.Option(50, "--num-episodes", "-n", help="Number of tournament episodes"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),  # noqa: B008
    mission: str = typer.Option(
        "extractor_hub_30",
        "--mission",
        "-m",
        help="CoGs vs Clips eval mission name (e.g., extractor_hub_30, collect_resources_classic)",
    ),
    num_cogs: int = typer.Option(4, "--num-cogs", help="Number of cogs/agents in the mission"),
) -> None:
    """Train agent with checkpoints, then run tournament on CoGs vs Clips missions."""
    console = Console()

    console.print("[bold cyan]Train & Tournament System[/bold cyan]")
    console.print(f"Training and evaluating on CoGs vs Clips mission: {mission}\n")

    # Default run name to mission name to avoid conflicts between missions
    if run is None:
        run = f"tournament_{mission}"

    checkpoint_dir = Path(run)

    # Step 1: Training (or skip if requested)
    if skip_training:
        console.print("[yellow]Skipping training, looking for existing checkpoints...[/yellow]")
        # Look in the train_dir structure
        train_dir = Path("./train_dir")
        actual_checkpoint_dir = train_dir / checkpoint_dir.name / "checkpoints"

        if not actual_checkpoint_dir.exists():
            # Fallback: try the provided directory directly
            actual_checkpoint_dir = checkpoint_dir

        # Get all .mpt files and exclude trainer_state.mpt
        all_mpt = sorted(actual_checkpoint_dir.glob("*.mpt"))
        checkpoint_paths = [f for f in all_mpt if f.name != "trainer_state.mpt"]

        if not checkpoint_paths:
            console.print(f"[red]Error: No policy checkpoints found in {actual_checkpoint_dir}[/red]")
            if all_mpt:
                console.print(
                    f"[yellow]Found {len(all_mpt)} .mpt files, but they appear to be trainer checkpoints[/yellow]"
                )
            raise typer.Exit(1)
        console.print(f"Found {len(checkpoint_paths)} policy checkpoints in {actual_checkpoint_dir}")
        # Take last 5 checkpoints
        checkpoint_paths = checkpoint_paths[-min(5, len(checkpoint_paths)) :]
    else:
        checkpoint_paths = train_with_checkpoints(
            checkpoint_dir=checkpoint_dir,
            max_epochs=max_epochs,
            checkpoint_interval=checkpoint_interval,
            num_cogs=num_cogs,
            mission=mission,
            console=console,
        )

    # Verify we have checkpoints before proceeding
    if not checkpoint_paths:
        console.print("[red]Error: No checkpoints available for tournament![/red]")
        console.print("[yellow]Training may have failed. Check the training logs above.[/yellow]")
        raise typer.Exit(1)

    # Step 2: Build policy pool with checkpoints
    policy_pool = build_policy_pool(checkpoint_paths, console)

    # Step 3: Run tournament
    game_results, policy_stats, policy_instances = run_tournament(
        mission=mission,
        policy_pool=policy_pool,
        num_episodes=num_episodes,
        team_size=num_cogs,  # Use num_cogs for team size
        seed=seed,
        console=console,
    )

    # Step 4: Run self-play evaluation
    self_play_stats = run_self_play(
        mission=mission,
        policy_pool=policy_pool,
        policy_instances=policy_instances,
        team_size=num_cogs,
        seed=seed,
        console=console,
        num_self_play_episodes=5,
    )

    # Step 5: Display results
    display_results(game_results, policy_stats, self_play_stats, console)

    # Step 6: Save results if requested
    if output:
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
            "self_play_rankings": [
                {
                    "rank": i + 1,
                    "policy_idx": stats.policy_idx,
                    "policy_name": stats.policy_name,
                    "mean_reward": stats.mean_reward,
                    "std_reward": stats.std_reward,
                    "episodes_played": stats.episodes_played,
                }
                for i, stats in enumerate(self_play_stats)
            ],
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    typer.run(main)
