"""CogWorks CLI with Biomes defaults."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Sequence

import typer
from rich.console import Console

from cogames import game
from cogames.main import (
    BASE_RUNS_DIR as _COGAMES_BASE_RUNS_DIR,
)
from cogames.main import (
    _collect_configs,
    _command_timeout,
    _dump_game_configs,
)
from cogames.main import (
    train_cmd as _cogames_train_cmd,
)

if TYPE_CHECKING:
    from mettagrid import MettaGridConfig  # type: ignore[import]

DEFAULT_MAP_NAMES: tuple[str, ...] = (
    "machina_1",
    "machina_2",
    "machina_3",
    "machina_4",
    "machina_5",
    "machina_6",
    "machina_7",
)
DEFAULT_RUN_DIR: Path = (_COGAMES_BASE_RUNS_DIR / "cogworks_biomes").resolve()


app = typer.Typer(
    help="CogWorks tooling with Biomes map defaults",
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


def _resolve_games(explicit: Optional[Sequence[str]]) -> list[str]:
    if explicit:
        return list(explicit)
    return list(DEFAULT_MAP_NAMES)


def _ensure_default_curricula(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    env_cfgs: list["MettaGridConfig"] = []
    env_names: list[str] = []
    for name in DEFAULT_MAP_NAMES:
        try:
            cfg = game.get_game(name)
        except ValueError as exc:  # pragma: no cover - validation handled in CLI path
            raise typer.BadParameter(f"Default CogWorks map '{name}' is unavailable: {exc}") from exc
        env_cfgs.append(cfg)
        env_names.append(name)
    _dump_game_configs(env_cfgs, env_names, destination)


@app.callback()
def main(
    ctx: typer.Context,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", help="Abort the command if it runs longer than the given seconds"),
    ] = None,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["timeout"] = timeout if timeout and timeout > 0 else None
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("curricula")
def curricula_cmd(
    ctx: typer.Context,
    games: Annotated[
        Optional[list[str]],
        typer.Option("--game", "-g", help="Specific games to export (defaults to CogWorks Biomes maps)"),
    ] = None,
    curriculum: Annotated[
        Optional[str],
        typer.Option(
            "--curriculum",
            help="Python path to a callable or iterable returning MettaGridConfig instances",
        ),
    ] = None,
    max_items: Annotated[
        int,
        typer.Option(
            "--max-items",
            help="Maximum number of curriculum items to export",
        ),
    ] = 128,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Directory to write map configurations"),
    ] = None,
) -> None:
    with _command_timeout(ctx):
        selected_games = _resolve_games(games)
        env_cfgs, env_names = _collect_configs(
            selected_games,
            curriculum,
            max_items,
            fallback_folder=None,
            game_param="game",
        )

        if not env_cfgs:
            raise typer.BadParameter("no games or curriculum items to export")

        if output_dir is not None:
            destination = output_dir.expanduser().resolve()
        else:
            DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
            destination = (DEFAULT_RUN_DIR / "curricula").resolve()

        _dump_game_configs(env_cfgs, env_names, destination)

        console.print(f"[green]Exported {len(env_cfgs)} CogWorks maps to: {destination}[/green]")


@app.command("train")
def train_cmd(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to train on"),
    policy_class_path: Annotated[
        str,
        typer.Option("--policy", help="Path to policy class"),
    ] = "cogames.policy.simple.SimplePolicy",
    initial_weights_path: Annotated[
        Optional[Path],
        typer.Option(
            "--initial-weights",
            help="Path to initial policy weights (file or directory for latest checkpoint)",
        ),
    ] = None,
    checkpoints_path: Annotated[
        Optional[Path],
        typer.Option("--checkpoints", help="Path to save training data"),
    ] = None,
    steps: Annotated[int, typer.Option("--steps", "-s", help="Number of training steps")] = 10000,
    device: Annotated[str, typer.Option("--device", help="Device to train on (e.g. 'auto', 'cpu', 'cuda')")] = "auto",
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
    num_envs: Annotated[Optional[int], typer.Option("--num-envs", help="Number of vectorized environments")] = None,
    num_workers: Annotated[Optional[int], typer.Option("--num-workers", help="Number of environment workers")] = None,
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
    run_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--run-dir",
            help="Base directory for this training run",
        ),
    ] = None,
    map_dump_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--map-dump-dir",
            help="Directory to export map configurations (defaults to run-dir/maps or checkpoints/maps)",
        ),
    ] = None,
) -> None:
    target_run_dir = (run_dir or DEFAULT_RUN_DIR).expanduser().resolve()
    fallback_dir = (target_run_dir / "curricula").resolve()
    _ensure_default_curricula(fallback_dir)

    _cogames_train_cmd(
        ctx,
        game_name=game_name,
        policy_class_path=policy_class_path,
        initial_weights_path=initial_weights_path,
        checkpoints_path=checkpoints_path,
        steps=steps,
        device=device,
        seed=seed,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        num_envs=num_envs,
        num_workers=num_workers,
        use_rnn=use_rnn,
        curriculum=curriculum,
        checkpoint_interval=checkpoint_interval,
        vector_backend=vector_backend,
        run_dir=target_run_dir,
        map_dump_dir=map_dump_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
