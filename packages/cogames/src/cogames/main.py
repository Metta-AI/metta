"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from cogames.evaluate import PolicySpec

# Always add current directory to Python path
sys.path.insert(0, ".")

import typer
from rich.console import Console

logger = logging.getLogger("cogames.main")

app = typer.Typer(help="CoGames - Multi-agent cooperative and competitive games")
console = Console()


def _resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand or full class path.
    Args:
        policy: Either a shorthand like "random", "simple", "lstm"
                or a full class path like "cogames.policy.random.RandomPolicy"
    Returns:
        Full class path to the policy
    """
    return {
        "random": "cogames.policy.random.RandomPolicy",
        "simple": "cogames.policy.simple.SimplePolicy",
        "lstm": "cogames.policy.lstm.LSTMPolicy",
        "claude": "cogames.policy.claude.ClaudePolicy",
    }.get(policy, policy)


def _require_game_argument(ctx: typer.Context, value: Optional[str]) -> str:
    if value is not None:
        return value

    from cogames import game

    console.print("[yellow]No game specified. Available games:[/yellow]")
    table = game.list_games(console)
    if table is not None:
        console.print(table)
    console.print(f"\n[dim]Usage: {ctx.command_path} <game>[/dim]")
    raise typer.Exit(0)


def _resolve_policy_data_path(policy_data_path: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint path if provided."""
    if policy_data_path is None:
        return None
    path = Path(policy_data_path)
    if path.is_file():
        return str(path)
    if not path.exists():
        console.print(f"[red]Checkpoint path not found: {path}[/red]")
        raise typer.Exit(1)

    last_touched_checkpoint_file = max(
        (p for p in path.rglob("*.pt")), key=lambda target: target.stat().st_mtime, default=None
    )
    if not last_touched_checkpoint_file:
        console.print(f"[red]No checkpoint files (*.pt) found in directory: {path}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Using checkpoint: {last_touched_checkpoint_file}[/green]")
    return str(last_touched_checkpoint_file)


def _parse_policy_option(spec: str) -> "PolicySpec":
    """Parse a policy CLI option into its components.

    Args:
        spec: string in the form ``policy_class:proportion[:policy_data]``.

    Returns:
        A list of PolicySpec objects

    Raises:
        typer.BadParameter: If the specification is malformed or invalid.
    """
    from cogames.evaluate import PolicySpec

    raw = spec.strip()
    if not raw:
        raise typer.BadParameter("Policy specification cannot be empty.")

    parts = raw.split(":", maxsplit=2)
    if len(parts) < 2:
        raise typer.BadParameter("Policy specification must include both class path and proportion separated by ':'")

    raw_class_path, raw_fraction = parts[0].strip(), parts[1].strip()
    raw_policy_data = parts[2].strip() if len(parts) == 3 else None

    if not raw_class_path:
        raise typer.BadParameter("Policy class path cannot be empty.")

    try:
        fraction = float(raw_fraction)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid proportion value '{raw_fraction}'.") from exc

    if fraction <= 0:
        raise typer.BadParameter("Policy proportion must be a positive number.")

    resolved_class_path = _resolve_policy_class_path(raw_class_path)
    resolved_policy_data = _resolve_policy_data_path(raw_policy_data)

    return PolicySpec(
        policy_class_path=resolved_class_path,
        proportion=fraction,
        policy_data_path=resolved_policy_data,
    )


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
    from cogames import game

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
        callback=_require_game_argument,
    ),
    policy_class_path: str = typer.Option(
        "cogames.policy.random.RandomPolicy",
        "--policy",
        help="Path to policy class",
        callback=_resolve_policy_class_path,
    ),
    policy_data_path: Optional[str] = typer.Option(
        None, "--policy-data", help="Path to initial policy weights", callback=_resolve_policy_data_path
    ),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of steps to run", min=1),
    render: Literal["gui", "text"] = typer.Option("gui", "--render", "-r", help="Render mode: 'gui' or 'text'"),
) -> None:
    from cogames import utils

    resolved_game, env_cfg = utils.get_game_config(console, game_name)

    console.print(f"[cyan]Playing {resolved_game}[/cyan]")
    console.print(f"Max Steps: {steps}, Interactive: {interactive}, Render: {render}")

    from cogames import play as play_module

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


@app.command(name="evaluate", no_args_is_help=True, help="Evaluate a policy on a game")
def evaluate_cmd(
    game_name: str = typer.Argument(
        None,
        help="Name of the game to evaluate",
        callback=_require_game_argument,
    ),
    policy_class_path: str = typer.Option(
        "cogames.policy.random.RandomPolicy",
        "--policy",
        help="Path to policy class",
        callback=_resolve_policy_class_path,
    ),
    policy_data_path: Optional[str] = typer.Option(
        None, "--policy-data", help="Path to policy weights", callback=_resolve_policy_data_path
    ),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes", min=1),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        help="Max milliseconds afforded to generate each action before noop is used by default",
        min=1,
    ),
) -> None:
    from cogames import utils

    resolved_game, env_cfg = utils.get_game_config(console, game_name)
    console.print(f"[cyan]Evaluating on {resolved_game}[/cyan]")
    console.print(f"Episodes: {episodes}")

    from cogames import evaluate as evaluate_module

    policy_specs = [
        evaluate_module.PolicySpec(
            policy_class_path=policy_class_path,
            proportion=1.0,
            policy_data_path=policy_data_path,
        )
    ]
    evaluate_module.evaluate(
        console,
        resolved_game=resolved_game,
        env_cfg=env_cfg,
        policy_specs=policy_specs,
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
    )


@app.command(name="evaluate-many", no_args_is_help=True, help="Evaluate many policies together on a game")
def evaluate_many_cmd(
    game_name: str = typer.Argument(
        None,
        help="Name of the game to evaluate",
        callback=_require_game_argument,
    ),
    policies: list[str] = typer.Argument(  # noqa: B008
        help=(
            "List of policies in the form 'class_path:proportion[:policy_data_path]'. "
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
    policy_specs = [_parse_policy_option(spec) for spec in policies]

    from cogames import utils

    resolved_game, env_cfg = utils.get_game_config(console, game_name)
    from cogames import evaluate as evaluate_module

    console.print(f"[cyan]Evaluating {len(policy_specs)} policies on {resolved_game}[/cyan]")
    console.print(f"Episodes: {episodes}")

    evaluate_module.evaluate(
        console,
        resolved_game=resolved_game,
        env_cfg=env_cfg,
        policy_specs=policy_specs,
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
    )


@app.command("make-game", help="Create a new game configuration")
def make_scenario(
    base_game: Optional[str] = typer.Argument(None, help="Base game to use as template"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents", min=1),
    width: int = typer.Option(10, "--width", "-w", help="Map width", min=1),
    height: int = typer.Option(10, "--height", "-h", help="Map height", min=1),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (YAML or JSON)"),  # noqa: B008
) -> None:
    from cogames import utils

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
        new_config.game.map_builder.width = width  # type: ignore[attr-defined]
        new_config.game.map_builder.height = height  # type: ignore[attr-defined]
        new_config.game.num_agents = num_agents

        if output:
            from cogames import game as game_module

            game_module.save_game_config(new_config, output)
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
        callback=_require_game_argument,
    ),
    policy_class_path: str = typer.Option(
        "cogames.policy.simple.SimplePolicy",
        "--policy",
        help="Path to policy class",
        callback=_resolve_policy_class_path,
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
) -> None:
    from cogames import train as train_module
    from cogames import utils

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
        )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


if __name__ == "__main__":
    app()
