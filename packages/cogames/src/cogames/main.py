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
from cogames.cogs_vs_clips.scenarios import make_game
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


@app.command("games", help="List all available games or describe a specific game")
def games_cmd(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to describe"),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save game configuration to file (YAML or JSON)"),  # noqa: B008
) -> None:
    if game_name is None:
        # List all games
        game.list_games(console)
    else:
        # Get the game configuration
        try:
            game_config = game.get_game(game_name)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

        # Save configuration if requested
        if save:
            try:
                game.save_game_config(game_config, save)
                console.print(f"[green]Game configuration saved to: {save}[/green]")
            except ValueError as e:
                console.print(f"[red]Error saving configuration: {e}[/red]")
                raise typer.Exit(1) from e
        else:
            # Otherwise describe the game
            try:
                game.describe_game(game_name, console)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from e


@app.command(name="play", no_args_is_help=True, help="Play a game")
def play_cmd(
    game_name: str = typer.Argument(
        None,
        help="Name of the game to play",
        callback=lambda ctx, value: game.require_game_argument(ctx, value, console),
    ),
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
    render: Literal["gui", "text"] = typer.Option("gui", "--render", "-r", help="Render mode: 'gui' or 'text'"),
) -> None:
    resolved_game, env_cfg = utils.get_game_config(console, game_name)

    console.print(f"[cyan]Playing {resolved_game}[/cyan]")
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


@app.command("make-game", help="Create a new game configuration")
def make_scenario(
    base_game: Optional[str] = typer.Argument(None, help="Base game to use as template"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents", min=1),
    width: int = typer.Option(10, "--width", "-w", help="Map width", min=1),
    height: int = typer.Option(10, "--height", "-h", help="Map height", min=1),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (YAML or JSON)"),  # noqa: B008
) -> None:
    try:
        # If base_game specified, use it as template
        if base_game:
            resolved_game, error = utils.resolve_game(base_game)
            if error:
                console.print(f"[red]Error: {error}[/red]")
                console.print("Creating from scratch instead...")
            else:
                console.print(f"[cyan]Using {resolved_game} as template[/cyan]")
        else:
            console.print("[cyan]Creating new game from scratch[/cyan]")

        # Use cogs_vs_clips make_game for now

        # Create game with specified parameters
        new_config = make_game(
            num_cogs=num_agents,
            num_assemblers=1,
            num_chests=1,
        )

        # Update map dimensions
        new_config.game.map_builder.width = width  # type: ignore[attr-defined]
        new_config.game.map_builder.height = height  # type: ignore[attr-defined]
        new_config.game.num_agents = num_agents

        if output:
            game.save_game_config(new_config, output)
            console.print(f"[green]Game configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="train", help="Train a policy on a game")
def train_cmd(
    game_name: str = typer.Argument(
        None,
        help="Name of the game to train on",
        callback=lambda ctx, value: game.require_game_argument(ctx, value, console),
    ),
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
    steps: int = typer.Option(10000, "--steps", "-s", help="Number of training steps", min=1),
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
) -> None:
    resolved_game, env_cfg = utils.get_game_config(console, game_name)

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
            game_name=resolved_game,
            vector_num_workers=num_workers,
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


@app.command(
    name="evaluate",
    no_args_is_help=True,
    help="Evaluate one or more policies on a game",
)
@app.command("eval", hidden=True)
def evaluate_cmd(
    game_name: str = typer.Argument(
        None,
        help="Name of the game to evaluate",
        callback=lambda ctx, value: game.require_game_argument(ctx, value, console),
    ),
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

    resolved_game, env_cfg = utils.get_game_config(console, game_name)

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
