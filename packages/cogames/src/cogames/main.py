"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Always add current directory to Python path
sys.path.insert(0, ".")

import typer
from rich.console import Console

logger = logging.getLogger("cogames.main")

app = typer.Typer(help="CoGames - Multi-agent cooperative and competitive games")
console = Console()

# Mapping of shorthand policy names to full class paths
POLICY_SHORTCUTS = {
    "random": "cogames.policy.random.RandomPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
}


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.

    Args:
        policy: Either a shorthand like "random", "simple", "lstm"
                or a full class path like "cogames.policy.random.RandomPolicy"

    Returns:
        Full class path to the policy
    """
    # If it's a shorthand, expand it
    if policy in POLICY_SHORTCUTS:
        return POLICY_SHORTCUTS[policy]
    # Otherwise assume it's already a full class path
    return policy


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """Show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        # No command provided, show help
        print(ctx.get_help())


@app.command("games")
def games_cmd(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to describe"),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save game configuration to file (YAML or JSON)"),  # noqa: B008
) -> None:
    """List all available games or describe a specific game."""
    from cogames import game

    if game_name is None:
        # List all games
        table = game.list_games(console)
        console.print(table)
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


@app.command(name="play")
def play_cmd(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to play"),
    policy_class_path: str = typer.Option(
        "cogames.policy.random.RandomPolicy", "--policy", help="Path to policy class"
    ),
    policy_data_path: Optional[str] = typer.Option(None, "--policy-data", help="Path to initial policy weights"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of steps to run"),
) -> None:
    """Play a game."""
    from cogames import game, utils

    # If no game specified, list games
    if game_name is None:
        console.print("[yellow]No game specified. Available games:[/yellow]")
        table = game.list_games(console)
        console.print(table)
        console.print("\n[dim]Usage: cogames play <game>[/dim]")
        return

    # Resolve game name
    resolved_game, error = utils.resolve_game(game_name)
    if error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)
    assert resolved_game is not None
    env_cfg = game.get_game(resolved_game)

    # Resolve policy shorthand
    full_policy_path = resolve_policy_class_path(policy_class_path)

    console.print(f"[cyan]Playing {resolved_game}[/cyan]")
    console.print(f"Max Steps: {steps}, Interactive: {interactive}")

    from cogames import play as play_module

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_class_path=full_policy_path,
        policy_data_path=policy_data_path,
        max_steps=steps,
        seed=42,
        verbose=interactive,  # Use interactive flag for verbose output
    )


@app.command("make-game")
def make_scenario(
    base_game: Optional[str] = typer.Argument(None, help="Base game to use as template"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents"),
    width: int = typer.Option(10, "--width", "-w", help="Map width"),
    height: int = typer.Option(10, "--height", "-h", help="Map height"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (YAML or JSON)"),  # noqa: B008
) -> None:
    """Create a new game configuration."""
    from cogames import game, utils

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
        from cogames.cogs_vs_clips.scenarios import make_game

        # Create game with specified parameters
        new_config = make_game(
            num_cogs=num_agents,
            num_assemblers=1,
            num_chests=1,
        )

        # Update map dimensions
        new_config.game.map_builder.width = width
        new_config.game.map_builder.height = height
        new_config.game.num_agents = num_agents

        if output:
            game.save_game_config(new_config, output)
            console.print(f"[green]Game configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="train")
def train_cmd(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to train on"),
    policy_class_path: str = typer.Option(
        "cogames.policy.simple.SimplePolicy", "--policy", help="Path to policy class"
    ),
    initial_weights_path: Optional[str] = typer.Option(
        None, "--initial-weights", help="Path to initial policy weights"
    ),
    checkpoints_path: str = typer.Option(
        "./train_dir",
        "--checkpoints",
        help="Path to save training data",
    ),
    steps: int = typer.Option(10000, "--steps", "-s", help="Number of training steps"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to train on (e.g. 'auto', 'cpu', 'cuda')",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for training"),
    batch_size: int = typer.Option(4096, "--batch-size", help="Batch size for training"),
    minibatch_size: int = typer.Option(4096, "--minibatch-size", help="Minibatch size for training"),
) -> None:
    """Train a policy on a game."""
    import torch

    from cogames import game, utils
    from cogames import train as train_module

    # If no game specified, list games
    if game_name is None:
        console.print("[yellow]No game specified. Available games:[/yellow]")
        table = game.list_games(console)
        console.print(table)
        console.print("\n[dim]Usage: cogames train <game>[/dim]")
        return

    # Resolve game name
    resolved_game, error = utils.resolve_game(game_name)
    if error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)
    assert resolved_game is not None
    env_cfg = game.get_game(resolved_game)

    # Resolve policy shorthand
    full_policy_path = resolve_policy_class_path(policy_class_path)

    def resolve_training_device(requested: str) -> torch.device:
        normalized = requested.strip().lower()

        def cuda_usable() -> bool:
            cuda_backend = getattr(torch.backends, "cuda", None)
            if cuda_backend is None or not cuda_backend.is_built():
                return False
            if not hasattr(torch._C, "_cuda_getDeviceCount"):
                return False
            return torch.cuda.is_available()

        if normalized == "auto":
            if cuda_usable():
                return torch.device("cuda")
            console.print("[yellow]CUDA not available; falling back to CPU for training.[/yellow]")
            return torch.device("cpu")

        try:
            candidate = torch.device(requested)
        except (RuntimeError, ValueError):
            console.print(f"[yellow]Warning: Unknown device '{requested}'. Falling back to CPU.[/yellow]")
            return torch.device("cpu")

        if candidate.type == "cuda" and not cuda_usable():
            console.print("[yellow]CUDA requested but unavailable. Training will run on CPU instead.[/yellow]")
            return torch.device("cpu")

        return candidate

    torch_device = resolve_training_device(device)

    try:
        train_module.train(
            env_cfg=env_cfg,
            policy_class_path=full_policy_path,
            initial_weights_path=initial_weights_path,
            device=torch_device,
            num_steps=steps,
            checkpoints_path=Path(checkpoints_path),
            seed=seed,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            game_name=resolved_game,
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


@app.command()
def evaluate(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to evaluate"),
    policy: Optional[str] = typer.Argument(None, help="Path to policy checkpoint or 'random' for random policy"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes"),
) -> None:
    """Evaluate a policy on a game."""
    console.print("[red]Coming soon...[/red]")


if __name__ == "__main__":
    app()
