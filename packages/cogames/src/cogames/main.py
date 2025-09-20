"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from cogames import game, play, train, utils

app = typer.Typer(help="CoGames - Multi-agent cooperative and competitive games")
console = Console()


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """Show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        # No command provided, show help
        print(ctx.get_help())


@app.command()
def games(game_name: Optional[str] = typer.Argument(None, help="Name of the game to describe")) -> None:
    """List all available games or describe a specific game."""
    if game_name is None:
        # List all games
        table = game.list_games(console)
        console.print(table)
    else:
        # Describe specific game
        try:
            game.describe_game(game_name, console)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e


@app.command(name="play")
def play_cmd(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to play"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of steps to run"),
    render: bool = typer.Option(True, "--render/--no-render", help="Whether to render the game"),
) -> None:
    """Play a game."""
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

    try:
        console.print(f"[cyan]Playing {resolved_game}[/cyan]")
        console.print(f"Steps: {steps}, Render: {render}, Interactive: {interactive}")

        play.play_game(
            game_name=resolved_game,
            scenario_name=resolved_game,  # For backward compatibility
            policy="random",
            steps=steps,
            episodes=1,
            interactive=interactive,
            render=render,
            console=console,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except ImportError as e:
        console.print(f"[red]Error: Failed to import mettagrid: {e}[/red]")
        console.print("Make sure mettagrid is properly installed.")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error running game: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def make_scenario(
    base_game: Optional[str] = typer.Argument(None, help="Base game to use as template"),
    name: str = typer.Option("custom_game", "--name", "-n", help="Name for the new game"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents"),
    width: int = typer.Option(10, "--width", "-w", help="Map width"),
    height: int = typer.Option(10, "--height", "-h", help="Map height"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (YAML or JSON)"),  # noqa: B008
) -> None:
    """Create a new game configuration."""
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
            num_base_extractors=1,
            num_wilderness_extractors=1,
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
    algorithm: str = typer.Option("ppo", "--algorithm", "-a", help="Training algorithm (ppo, a2c, dqn)"),
    steps: int = typer.Option(10000, "--steps", "-s", help="Number of training steps"),
    save_path: Optional[Path] = typer.Option(None, "--save", help="Path to save trained model"),  # noqa: B008
    wandb_project: Optional[str] = typer.Option(None, "--wandb", help="W&B project name for logging"),  # noqa: B008
) -> None:
    """Train a policy on a game."""
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

    try:
        train.train_policy(
            game_name=resolved_game,
            scenario_name=resolved_game,  # For backward compatibility
            algorithm=algorithm,
            steps=steps,
            save_path=save_path,
            wandb_project=wandb_project,
            console=console,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def evaluate(
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to evaluate"),
    policy: Optional[str] = typer.Argument(None, help="Path to policy checkpoint or 'random' for random policy"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes"),
    render: bool = typer.Option(False, "--render", "-r", help="Render evaluation episodes"),
    save_video: Optional[Path] = typer.Option(None, "--video", help="Save evaluation video to path"),  # noqa: B008
) -> None:
    """Evaluate a policy on a game."""
    # If no game specified, list games
    if game_name is None:
        console.print("[yellow]No game specified. Available games:[/yellow]")
        table = game.list_games(console)
        console.print(table)
        console.print("\n[dim]Usage: cogames evaluate <game> [policy][/dim]")
        return

    # Resolve game name
    resolved_game, error = utils.resolve_game(game_name)
    if error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)

    # Default policy to random if not specified
    if policy is None:
        policy = "random"
        console.print("[yellow]No policy specified, using random policy[/yellow]")

    try:
        console.print(f"[cyan]Evaluating policy on {resolved_game}[/cyan]")
        console.print(f"Policy: {policy}")
        console.print(f"Episodes: {episodes}")
        console.print(f"Render: {render}")

        if save_video:
            console.print(f"Video will be saved to: {save_video}")

        play.evaluate_policy(
            game_name=resolved_game,
            scenario_name=resolved_game,  # For backward compatibility
            policy=policy,
            episodes=episodes,
            render=render,
            save_video=save_video,
            console=console,
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except ImportError as e:
        console.print(f"[red]Error: Failed to import mettagrid: {e}[/red]")
        console.print("Make sure mettagrid is properly installed.")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
