#!/usr/bin/env -S uv run

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import importlib.metadata
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from packaging.version import Version

from cogames import evaluate as evaluate_module
from cogames import game, utils
from cogames import play as play_module
from cogames import train as train_module
from cogames.policy.utils import parse_policy_spec, resolve_policy_class_path, resolve_policy_data_path

# Always add current directory to Python path
sys.path.insert(0, ".")

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("cogames.main")

app = typer.Typer(
    help="CoGames - Multi-agent cooperative and competitive games",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """Show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        # No command provided, show help
        print(ctx.get_help())


mission_argument = typer.Argument(
    None,
    help="Name of the mission. Can be in the format 'map_name' or 'map_name.custom_mission' or 'path/to/mission.yaml'.",
    callback=lambda ctx, value: game.require_mission_argument(ctx, value, console),
)


@app.command("missions", help="List all available missions, or describe a specific mission")
@app.command("games", hidden=True)
def games_cmd(
    mission_name: str = mission_argument,
) -> None:
    mission_name, env_cfg = utils.get_mission_config(console, mission_name)
    try:
        game.describe_mission(mission_name, env_cfg, console)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="play", no_args_is_help=True, help="Play a game")
def play_cmd(
    mission_name: str = mission_argument,
    policy_class_path: str = typer.Option(
        "cogames.policy.random.RandomPolicy",
        "--policy",
        help="Path to policy class",
        callback=resolve_policy_class_path,
    ),
    policy_data_path: Optional[str] = typer.Option(
        None,
        "--policy-data",
        help="Path to policy weights file or directory",
        callback=resolve_policy_data_path,
    ),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of steps to run", min=1),
    render: Literal["gui", "text", "none"] = typer.Option(
        "gui", "--render", "-r", help="Render mode: 'gui', 'text', or 'none' (no rendering)"
    ),
) -> None:
    mission_name, env_cfg = utils.get_mission_config(console, mission_name)

    console.print(f"[cyan]Playing {mission_name}[/cyan]")
    console.print(f"Max Steps: {steps}, Interactive: {interactive}, Render: {render}")

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_class_path=policy_class_path,
        policy_data_path=policy_data_path,
        max_steps=steps,
        seed=42,
        render=render,
        verbose=interactive,  # Use interactive flag for verbose output
    )


@app.command("make-mission", help="Create a new mission configuration")
@app.command("make-game", hidden=True)
def make_mission(
    base_mission: str = mission_argument,
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents", min=1),
    width: int = typer.Option(10, "--width", "-w", help="Map width", min=1),
    height: int = typer.Option(10, "--height", "-h", help="Map height", min=1),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (yml or json)"),  # noqa: B008
) -> None:
    try:
        _, env_cfg = utils.get_mission_config(console, base_mission)

        # Update map dimensions
        env_cfg.game.map_builder.width = width  # type: ignore[attr-defined]
        env_cfg.game.map_builder.height = height  # type: ignore[attr-defined]
        env_cfg.game.num_agents = num_agents

        if output:
            game.save_mission_config(env_cfg, output)
            console.print(f"[green]Game configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="train", help="Train a policy on a mission")
def train_cmd(
    mission_name: str = mission_argument,
    policy_class_path: str = typer.Option(
        "cogames.policy.simple.SimplePolicy",
        "--policy",
        help="Path to policy class",
        callback=resolve_policy_class_path,
    ),
    initial_weights_path: Optional[str] = typer.Option(
        None,
        "--initial-weights",
        help="Path to initial policy weights .pt file",
    ),
    checkpoints_path: str = typer.Option(
        "./train_dir",
        "--checkpoints",
        help="Path to save training data",
    ),
    steps: int = typer.Option(10_0000_000_000, "--steps", "-s", help="Number of training steps", min=1),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to train on (e.g. 'auto', 'cpu', 'cuda')",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for training", min=0),
    batch_size: int = typer.Option(4096, "--batch-size", help="Batch size for training", min=1),
    minibatch_size: int = typer.Option(4096, "--minibatch-size", help="Minibatch size for training", min=1),
    num_workers: Optional[int] = typer.Option(
        None,
        "--num-workers",
        help="Number of worker processes (defaults to number of CPU cores)",
        min=1,
    ),
    parallel_envs: Optional[int] = typer.Option(
        None,
        "--parallel-envs",
        help="Number of parallel environments",
        min=1,
    ),
) -> None:
    mission_name, env_cfg = utils.get_mission_config(console, mission_name)

    torch_device = utils.resolve_training_device(console, device)

    try:
        train_module.train(
            env_cfg=env_cfg,
            policy_class_path=policy_class_path,
            initial_weights_path=initial_weights_path,
            device=torch_device,
            num_steps=steps,
            checkpoints_path=Path(checkpoints_path),
            seed=seed,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            vector_num_workers=num_workers,
            vector_num_envs=parallel_envs,
            vector_batch_size=batch_size,
            game_name=mission_name,
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


@app.command(
    name="evaluate",
    no_args_is_help=True,
    help="Evaluate one or more policies on a mission",
)
@app.command("eval", hidden=True)
def evaluate_cmd(
    mission_name: str = mission_argument,
    policies: list[str] = typer.Argument(  # noqa: B008
        None,
        help=(
            "List of policies in the form '{policy_class_path}[:policy_data_path][:proportion]'. "
            "Provide multiple options for mixed populations."
        ),
    ),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes", min=1),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        help="Max milliseconds afforded to generate each action before noop is used by default",
        min=1,
    ),
) -> None:
    if not policies:
        console.print("[red]Error: No policies provided[/red]")
        raise typer.Exit(1)
    policy_specs = [parse_policy_spec(spec) for spec in policies]  # noqa: F821

    resolved_game, env_cfg = utils.get_mission_config(console, mission_name)

    console.print(f"[cyan]Evaluating {len(policy_specs)} policies on {resolved_game} over {episodes} episodes[/cyan]")

    evaluate_module.evaluate(
        console,
        resolved_game=resolved_game,
        env_cfg=env_cfg,
        policy_specs=policy_specs,
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
    )


@app.command(name="version", help="Show version information")
def version_cmd() -> None:
    def public_version(dist_name: str) -> str:
        return str(Version(importlib.metadata.version(dist_name)).public)

    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="right", style="bold cyan")
    table.add_column("", justify="right")

    for dist_name in ["mettagrid", "pufferlib-core", "cogames"]:
        table.add_row(dist_name, public_version(dist_name))

    console.print(table)


if __name__ == "__main__":
    app()
