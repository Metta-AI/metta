"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import contextlib
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Optional, Sequence

# Always add current directory to Python path
sys.path.insert(0, ".")

import torch
import typer
from rich.console import Console

from cogames import game, play, serialization, train, utils
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol

logger = logging.getLogger("cogames.main")

app = typer.Typer(help="CoGames - Multi-agent cooperative and competitive games")
console = Console()
policy_app = typer.Typer(help="Policy utilities")
app.add_typer(policy_app, name="policy")


@contextlib.contextmanager
def _command_timeout(ctx: typer.Context):
    timeout: Optional[int] = None
    if ctx is not None:
        ctx.ensure_object(dict)
        timeout = ctx.obj.get("timeout")
    with utils.cli_timeout(timeout):
        yield


def _ensure_config(value: Any) -> MettaGridConfig:
    if isinstance(value, MettaGridConfig):
        return value
    if isinstance(value, (str, Path)):
        return game.get_game(str(value))
    if isinstance(value, dict):
        return MettaGridConfig.model_validate(value)
    msg = f"Unsupported curriculum item type: {type(value)!r}."
    raise ValueError(msg)


def _load_curriculum_configs(source: Any, max_items: int) -> Sequence[MettaGridConfig]:
    queue: list[Any] = [source]
    configs: list[MettaGridConfig] = []

    while queue and len(configs) < max_items:
        item = queue.pop(0)

        if isinstance(item, MettaGridConfig) or isinstance(item, (str, Path)) or isinstance(item, dict):
            configs.append(_ensure_config(item))
            continue

        if callable(item):
            produced = item()
            queue.append(produced)
            continue

        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for produced in item:
                queue.append(produced)
                if len(configs) + len(queue) >= max_items:
                    break
            continue

        msg = f"Curriculum source produced unsupported type: {type(item)!r}."
        raise ValueError(msg)

    if not configs:
        raise ValueError("Curriculum did not yield any MettaGridConfig instances.")

    return configs


def _resolve_initial_weights(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            (candidate for candidate in path.iterdir() if candidate.is_file()),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        for candidate in reversed(candidates):
            if candidate.suffix in {".pt", ".pth", ".ckpt"}:
                return candidate
        raise ValueError(f"No checkpoint files found in directory: {path}")
    raise ValueError(f"Initial weights path not found: {path}")


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", help="Abort the command if it runs longer than the given seconds"),
    ] = None,
) -> None:
    """Show help when no command is provided."""
    ctx.ensure_object(dict)
    ctx.obj["timeout"] = timeout if timeout and timeout > 0 else None

    if ctx.invoked_subcommand is None:
        # No command provided, show help
        print(ctx.get_help())


@app.command("games")
def games_cmd(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to describe"),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save game configuration to file (YAML or JSON)"),  # noqa: B008
) -> None:
    """List all available games or describe a specific game."""
    with _command_timeout(ctx):
        if game_name is None:
            table = game.list_games(console)
            console.print(table)
            return

        game_config = game.get_game(game_name)

        if save:
            game.save_game_config(game_config, save)
            console.print(f"[green]Game configuration saved to: {save}[/green]")
            return

        game.describe_game(game_name, console)


@app.command(name="play")
def play_cmd(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to play"),
    policy_class_path: str = typer.Option(
        "cogames.examples.random_policy.RandomPolicy", "--policy", help="Path to policy class"
    ),
    policy_data_path: Optional[str] = typer.Option(None, "--policy-data", help="Path to initial policy weights"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of steps to run"),
) -> None:
    """Play a game."""
    with _command_timeout(ctx):
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
            raise typer.BadParameter(error, param_name="game_name")
        assert resolved_game is not None
        env_cfg = game.get_game(resolved_game)

        console.print(f"[cyan]Playing {resolved_game}[/cyan]")
        console.print(f"Max Steps: {steps}, Interactive: {interactive}")

        play.play(
            console,
            env_cfg=env_cfg,
            policy_class_path=policy_class_path,
            policy_data_path=policy_data_path,
            max_steps=steps,
            seed=42,
            verbose=interactive,  # Use interactive flag for verbose output
        )


@app.command("make-game")
def make_scenario(
    ctx: typer.Context,
    base_game: Optional[str] = typer.Argument(None, help="Base game to use as template"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents"),
    width: int = typer.Option(10, "--width", "-w", help="Map width"),
    height: int = typer.Option(10, "--height", "-h", help="Map height"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (YAML or JSON)"),  # noqa: B008
) -> None:
    """Create a new game configuration."""
    with _command_timeout(ctx):
        if base_game:
            resolved_game, error = utils.resolve_game(base_game)
            if error:
                raise typer.BadParameter(error, param_name="base_game")
            console.print(f"[cyan]Using {resolved_game} as template[/cyan]")
        else:
            console.print("[cyan]Creating new game from scratch[/cyan]")

        from cogames.cogs_vs_clips.scenarios import make_game

        new_config = make_game(
            num_cogs=num_agents,
            num_assemblers=1,
            num_chests=1,
        )
        new_config.game.map_builder.width = width
        new_config.game.map_builder.height = height
        new_config.game.num_agents = num_agents

        if output:
            game.save_game_config(new_config, output)
            console.print(f"[green]Game configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")


@app.command(name="train")
def train_cmd(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to train on"),
    policy_class_path: Annotated[
        str,
        typer.Option("--policy", help="Path to policy class"),
    ] = "cogames.examples.simple_policy.SimplePolicy",
    initial_weights_path: Annotated[
        Optional[Path],
        typer.Option(
            "--initial-weights",
            help="Path to initial policy weights (file or directory for latest checkpoint)",
        ),
    ] = None,
    checkpoints_path: Annotated[
        str,
        typer.Option("--checkpoints", help="Path to save training data"),
    ] = "./experiments",
    steps: Annotated[int, typer.Option("--steps", "-s", help="Number of training steps")] = 10000,
    device: Annotated[str, typer.Option("--device", help="Device to train on")] = "cuda",
    seed: Annotated[int, typer.Option("--seed", help="Seed for training")] = 42,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-size",
            help="Batch size for PPO updates (defaults to num-envs * rollout-length)",
        ),
    ] = None,
    minibatch_size: Annotated[
        Optional[int],
        typer.Option("--minibatch-size", help="Minibatch size for PPO updates (defaults to batch-size)"),
    ] = None,
    num_envs: Annotated[int, typer.Option("--num-envs", help="Number of vectorized environments")] = 128,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of environment workers")] = 8,
    use_rnn: Annotated[
        bool,
        typer.Option("--use-rnn/--no-use-rnn", help="Enable recurrent policies"),
    ] = False,
    curriculum: Annotated[
        Optional[str],
        typer.Option(
            "--curriculum",
            help="Python path to a callable or iterable returning MettaGridConfig instances",
        ),
    ] = None,
    checkpoint_interval: Annotated[
        int,
        typer.Option("--checkpoint-interval", help="Steps between automatic checkpoints"),
    ] = 200,
    vector_backend: Annotated[
        Literal["multiprocessing", "serial", "ray"],
        typer.Option(
            "--vector-backend",
            help="Vector environment backend to use",
            case_sensitive=False,
        ),
    ] = "multiprocessing",
) -> None:
    """Train a policy on a game."""
    with _command_timeout(ctx):
        backend = vector_backend.lower()
        if game_name is None and curriculum is None:
            raise typer.BadParameter("provide a game or curriculum", param_name="game_name")

        env_cfgs: list[MettaGridConfig] = []

        if game_name is not None:
            resolved_game, error = utils.resolve_game(game_name)
            if error:
                raise typer.BadParameter(error, param_name="game_name")
            assert resolved_game is not None
            env_cfgs.append(game.get_game(resolved_game))

        if curriculum is not None:
            curriculum_source = load_symbol(curriculum)
            max_items = max(num_envs, 32)
            env_cfgs.extend(_load_curriculum_configs(curriculum_source, max_items))

        if not env_cfgs:
            raise typer.BadParameter("curriculum did not yield any configurations", param_name="curriculum")

        resolved_initial = _resolve_initial_weights(initial_weights_path)

        effective_batch = batch_size or max(num_envs * 32, 512)
        effective_minibatch = minibatch_size or effective_batch

        if effective_minibatch > effective_batch:
            raise typer.BadParameter("minibatch must be <= batch size", param_name="minibatch_size")

        if num_workers < 1:
            raise typer.BadParameter("num-workers must be >= 1", param_name="num_workers")

        if num_envs % num_workers != 0:
            raise typer.BadParameter(
                "num-envs must be divisible by num-workers",
                param_name="num_envs",
            )

        if backend == "ray" and not hasattr(train.pufferlib.vector, "Ray"):
            raise typer.BadParameter("Ray backend is not available", param_name="vector_backend")

        device_obj = torch.device(device)

        train.train(
            env_cfgs=env_cfgs,
            policy_class_path=policy_class_path,
            initial_weights_path=resolved_initial,
            device=device_obj,
            num_steps=steps,
            checkpoints_path=Path(checkpoints_path),
            seed=seed,
            batch_size=effective_batch,
            minibatch_size=effective_minibatch,
            num_envs=num_envs,
            num_workers=num_workers,
            use_rnn=use_rnn,
            checkpoint_interval=checkpoint_interval,
            vector_backend=backend,
        )

        console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


@app.command()
def evaluate(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to evaluate"),
    policy: Optional[str] = typer.Argument(None, help="Path to policy checkpoint or 'random' for random policy"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes"),
) -> None:
    """Evaluate a policy on a game."""
    with _command_timeout(ctx):
        console.print("[red]Coming soon...[/red]")


@policy_app.command("export")
def policy_export(
    policy_class: Annotated[str, typer.Argument(help="Policy class path")],
    checkpoint_path: Annotated[Path, typer.Argument(help="Existing checkpoint to bundle")],
    output_dir: Annotated[Path, typer.Argument(help="Destination directory")],
) -> None:
    artifact = serialization.bundle_policy(policy_class, checkpoint_path, output_dir)
    console.print(f"[green]Policy bundle created at {artifact.weights_path.parent}[/green]")


@policy_app.command("load")
def policy_load(
    bundle_dir: Annotated[Path, typer.Argument(help="Path to policy bundle directory")],
    game_name: Annotated[str, typer.Argument(help="Game name to instantiate environment")],
    device: str = typer.Option("cpu", "--device", help="Device for the policy"),
) -> None:
    env_cfg = game.get_game(game_name)
    env = MettaGridEnv(env_cfg=env_cfg)
    policy = serialization.load_policy_from_bundle(bundle_dir, env, torch.device(device))
    policy.reset()
    console.print(f"[green]Loaded policy {policy.__class__.__name__} on {device}[/green]")


@policy_app.command("inspect")
def policy_inspect(bundle_dir: Annotated[Path, typer.Argument(help="Path to policy bundle directory")]) -> None:
    metadata = serialization.inspect_bundle(bundle_dir)
    console.print_json(data=metadata)


if __name__ == "__main__":
    app()
