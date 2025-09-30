"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import contextlib
import logging
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Optional, Sequence, Tuple

# Always add current directory to Python path
sys.path.insert(0, ".")

import torch
import typer
from rich.console import Console

from cogames import curriculum as curriculum_utils
from cogames import game, play, serialization, train, utils
from mettagrid import MettaGridConfig, MettaGridEnv
from mettagrid.util.module import load_symbol

logger = logging.getLogger("cogames.main")

app = typer.Typer(help="CoGames - Multi-agent cooperative and competitive games")
console = Console()
policy_app = typer.Typer(help="Policy utilities")
app.add_typer(policy_app, name="policy")

BASE_RUNS_DIR = (Path(__file__).resolve().parent / "runs").resolve()


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


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return sanitized or "map"


def _dump_game_configs(configs: Sequence[MettaGridConfig], names: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.yaml"):
        existing.unlink()
    for index, (config_obj, raw_name) in enumerate(zip(configs, names, strict=False)):
        base_name = raw_name or f"map_{index:03d}"
        file_stem = _sanitize_filename(base_name)
        candidate = output_dir / f"{file_stem}.yaml"
        if candidate.exists():
            candidate = output_dir / f"{file_stem}_{index:03d}.yaml"
        game.save_game_config(config_obj, candidate)


def _default_device(explicit: Optional[str]) -> str:
    if explicit is not None:
        normalized = explicit.lower()

        if normalized == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            console.print("[yellow]CUDA/MPS unavailable; training will run on CPU.[/yellow]")
            return "cpu"

        try:
            requested = torch.device(explicit)
        except (RuntimeError, ValueError):
            console.print(f"[yellow]Unknown device '{explicit}'. Falling back to CPU.[/yellow]")
            return "cpu"

        if requested.type == "cuda" and not torch.cuda.is_available():
            console.print("[yellow]CUDA requested but unavailable. Training will run on CPU instead.[/yellow]")
            return "cpu"

        if requested.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            console.print("[yellow]MPS requested but unavailable. Training will run on CPU instead.[/yellow]")
            return "cpu"

        return str(requested)

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_run_dir(run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir.expanduser().resolve()
    return (BASE_RUNS_DIR / "default").resolve()


def _suggest_parallelism(
    device: str,
    requested_envs: Optional[int],
    requested_workers: Optional[int],
) -> Tuple[int, int]:
    if requested_envs is not None and requested_workers is not None:
        return requested_envs, requested_workers

    cpu_count = os.cpu_count() or 4

    if requested_envs is not None:
        envs = max(1, requested_envs)
    else:
        if device == "cuda":
            envs = min(max(cpu_count * 2, 8), 32)
        else:
            envs = min(max(cpu_count, 2), 16)

    if requested_workers is not None:
        workers = max(1, requested_workers)
    else:
        workers = max(1, min(envs, max(1, cpu_count // 2)))

    if envs % workers != 0:
        for candidate in range(workers, 0, -1):
            if envs % candidate == 0:
                workers = candidate
                break
        else:
            workers = 1
            envs = max(envs, workers)

    return envs, workers


def _collect_configs(
    game_names: Iterable[str],
    curriculum_path: Optional[str],
    max_items: int,
    fallback_folder: Optional[Path],
    game_param: str,
) -> Tuple[list[MettaGridConfig], list[str]]:
    configs: list[MettaGridConfig] = []
    names: list[str] = []

    for game_name in game_names:
        resolved_game, error = utils.resolve_game(game_name)
        if error:
            raise typer.BadParameter(error, param_name=game_param)
        if resolved_game is None:
            raise typer.BadParameter(f"game '{game_name}' not found", param_name=game_param)
        configs.append(game.get_game(resolved_game))
        names.append(resolved_game)

    if curriculum_path is not None:
        curriculum_source = load_symbol(curriculum_path)
        curriculum_cfgs = _load_curriculum_configs(curriculum_source, max_items)
        start_index = len(names)
        for offset, cfg in enumerate(curriculum_cfgs):
            cfg_name = getattr(getattr(cfg, "game", None), "name", None)
            label = str(cfg_name) if cfg_name else f"curriculum_{start_index + offset:03d}"
            configs.append(cfg)
            names.append(label)
    elif fallback_folder is not None and fallback_folder.exists():
        try:
            folder_cfgs, folder_names = curriculum_utils.load_map_folder_with_names(fallback_folder)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            pass
        else:
            configs.extend(folder_cfgs)
            names.extend(folder_names)

    for index in range(len(names), len(configs)):
        names.append(f"map_{index:03d}")

    return configs, names


def _filter_uniform_agent_count(
    configs: Sequence[MettaGridConfig],
    names: Sequence[str],
) -> Tuple[list[MettaGridConfig], list[str]]:
    if not configs:
        return list(configs), list(names)

    counts = [cfg.game.num_agents for cfg in configs]
    if len(set(counts)) <= 1:
        return list(configs), list(names)

    most_common_count = Counter(counts).most_common(1)[0][0]
    filtered_cfgs: list[MettaGridConfig] = []
    filtered_names: list[str] = []
    dropped = 0
    for cfg, name, count in zip(configs, names, counts, strict=False):
        if count == most_common_count:
            filtered_cfgs.append(cfg)
            filtered_names.append(name)
        else:
            dropped += 1

    if dropped:
        console.print(
            "[yellow]Skipping {dropped} map(s) with mismatched agent counts. "
            "Training will use configs with {agents} agent(s).[/yellow]".format(
                dropped=dropped,
                agents=most_common_count,
            )
        )

    return filtered_cfgs, filtered_names


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

        try:
            game_config = game.get_game(game_name)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_name="game_name") from exc

        if save:
            game.save_game_config(game_config, save)
            console.print(f"[green]Game configuration saved to: {save}[/green]")
            return

        game.describe_game(game_name, console)


@app.command("curricula")
def curricula_cmd(
    ctx: typer.Context,
    games: Annotated[
        Optional[list[str]],
        typer.Option("--game", "-g", help="Specific games to export (defaults to all Cogs vs Clips games)"),
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
    """Export game or curriculum maps into a folder for reuse."""

    with _command_timeout(ctx):
        selected_games = games or list(game.get_all_games().keys())
        env_cfgs, env_names = _collect_configs(
            selected_games,
            curriculum,
            max_items,
            fallback_folder=None,
            game_param="game_name",
        )

        if not env_cfgs:
            raise typer.BadParameter("no games or curriculum items to export", param_name="game")

        if output_dir is not None:
            destination = output_dir.expanduser().resolve()
        else:
            destination = (_resolve_run_dir(None) / "curricula").resolve()

        _dump_game_configs(env_cfgs, env_names, destination)

        console.print(f"[green]Exported {len(env_cfgs)} maps to: {destination}[/green]")


@app.command("clean")
def clean_cmd(
    ctx: typer.Context,
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Base run directory to clean (defaults to cogames runs directory)"),
    ] = None,
    remove_curricula: Annotated[
        bool,
        typer.Option("--curricula/--no-curricula", help="Remove exported curriculum maps"),
    ] = True,
    remove_checkpoints: Annotated[
        bool,
        typer.Option("--checkpoints/--no-checkpoints", help="Remove training checkpoints"),
    ] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show actions without deleting files")] = False,
) -> None:
    """Remove cached curricula and checkpoint artifacts for runs."""

    if not remove_curricula and not remove_checkpoints:
        console.print("[yellow]Nothing to clean: enable --curricula and/or --checkpoints.")
        return

    with _command_timeout(ctx):
        resolved_run_dir = _resolve_run_dir(run_dir)
        targets: list[Path] = []

        if remove_curricula:
            targets.append((resolved_run_dir / "curricula").resolve())
        if remove_checkpoints:
            targets.append((resolved_run_dir / "checkpoints").resolve())

        seen: set[Path] = set()
        removed_any = False

        for target in targets:
            if target in seen:
                continue
            seen.add(target)

            if not target.exists():
                console.print(f"[yellow]Skipping missing path: {target}")
                continue

            if not target.is_dir():
                console.print(f"[yellow]Skipping non-directory path: {target}")
                continue

            if dry_run:
                console.print(f"[cyan]Would remove: {target}")
                continue

            shutil.rmtree(target)
            console.print(f"[green]Removed: {target}")
            removed_any = True

        if dry_run:
            console.print("[green]Dry run complete. No changes made.")
        elif not removed_any:
            console.print("[yellow]No matching directories were removed.")


@app.command(name="play")
def play_cmd(
    ctx: typer.Context,
    game_name: Optional[str] = typer.Argument(None, help="Name of the game to play"),
    policy_class_path: str = typer.Option(
        "cogames.policy.random.RandomPolicy", "--policy", help="Path to policy class"
    ),
    policy_data_path: Optional[str] = typer.Option(None, "--policy-data", help="Path to initial policy weights"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of steps to run"),
) -> None:
    """Play a game."""
    with _command_timeout(ctx):
        if game_name is None:
            console.print("[yellow]No game specified. Available games:[/yellow]")
            table = game.list_games(console)
            console.print(table)
            console.print("\n[dim]Usage: cogames play <game>[/dim]")
            return

        resolved_game, error = utils.resolve_game(game_name)
        if error:
            raise typer.BadParameter(error, param_name="game_name")
        assert resolved_game is not None
        env_cfg = game.get_game(resolved_game)

        full_policy_path = resolve_policy_class_path(policy_class_path)

        console.print(f"[cyan]Playing {resolved_game}[/cyan]")
        console.print(f"Max Steps: {steps}, Interactive: {interactive}")

        play.play(
            console,
            env_cfg=env_cfg,
            policy_class_path=full_policy_path,
            policy_data_path=policy_data_path,
            max_steps=steps,
            seed=42,
            verbose=interactive,
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
        try:
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
        except typer.BadParameter:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for CLI errors
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc


@app.command(name="train")
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
    """Train a policy on a game."""
    with _command_timeout(ctx):
        resolved_run_dir = _resolve_run_dir(run_dir)
        resolved_run_dir.mkdir(parents=True, exist_ok=True)

        backend = vector_backend.lower()
        resolved_device_name = _default_device(device)
        resolved_num_envs, resolved_num_workers = _suggest_parallelism(resolved_device_name, num_envs, num_workers)
        if vector_backend.lower() == "multiprocessing" and sys.platform == "darwin":
            resolved_num_workers = 1

        fallback_curricula = resolved_run_dir / "curricula"
        base_games = [game_name] if game_name is not None else []
        if not base_games and curriculum is None and not fallback_curricula.exists():
            base_games = ["assembler_1_simple"]
        env_cfgs, env_names = _collect_configs(
            base_games,
            curriculum,
            max(resolved_num_envs, 32),
            fallback_folder=fallback_curricula,
            game_param="game_name",
        )
        env_cfgs, env_names = _filter_uniform_agent_count(env_cfgs, env_names)

        representative_game = env_names[0] if env_names else None

        if env_cfgs:
            import math

            available = len(env_cfgs)
            if num_envs is None:
                resolved_num_envs = max(1, min(resolved_num_envs, available))
            else:
                resolved_num_envs = max(1, min(num_envs, available))

            if num_workers is None:
                resolved_num_workers = max(1, min(resolved_num_workers, resolved_num_envs))
            else:
                resolved_num_workers = max(1, min(num_workers, resolved_num_envs))

            if resolved_num_envs % resolved_num_workers != 0:
                gcd = math.gcd(resolved_num_envs, resolved_num_workers)
                resolved_num_workers = gcd or 1

            env_cfgs = env_cfgs[:resolved_num_envs]
            env_names = env_names[:resolved_num_envs]

        if not env_cfgs:
            if game_name is None and curriculum is None:
                console.print("[yellow]No game or curriculum specified. Available games:[/yellow]")
                table = game.list_games(console)
                console.print(table)
                console.print("\n[dim]Usage: cogames train <game>[/dim]")
            msg = (
                "No game or curriculum configurations found. Provide a game name, specify --curriculum, "
                "or generate maps with 'cogames curricula'."
            )
            raise typer.BadParameter(msg)

        while len(env_names) < len(env_cfgs):
            env_names.append(f"map_{len(env_names):03d}")

        resolved_initial = None
        if initial_weights_path is not None:
            resolved_initial = _resolve_initial_weights(initial_weights_path)

        default_batch = max(resolved_num_envs * 32, 512)
        if batch_size is not None:
            effective_batch = max(resolved_num_envs, min(batch_size, steps))
        else:
            effective_batch = min(default_batch, max(resolved_num_envs, steps))

        if minibatch_size is not None:
            effective_minibatch = max(1, min(minibatch_size, effective_batch))
        else:
            effective_minibatch = effective_batch

        if effective_minibatch > effective_batch:
            raise typer.BadParameter("minibatch must be <= batch size", param_name="minibatch_size")

        if backend == "ray" and not hasattr(train.pufferlib.vector, "Ray"):
            raise typer.BadParameter("Ray backend is not available", param_name="vector_backend")

        device_obj = torch.device(resolved_device_name)

        if checkpoints_path is not None:
            checkpoints_dir = checkpoints_path.expanduser().resolve()
        else:
            checkpoints_dir = (resolved_run_dir / "checkpoints").resolve()
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if map_dump_dir is not None:
            maps_dir = map_dump_dir if map_dump_dir.is_absolute() else (resolved_run_dir / map_dump_dir).resolve()
        else:
            maps_dir = (resolved_run_dir / "curricula").resolve()

        _dump_game_configs(env_cfgs, env_names, maps_dir)

        full_policy_path = resolve_policy_class_path(policy_class_path)

        train.train(
            env_cfgs=env_cfgs,
            policy_class_path=full_policy_path,
            initial_weights_path=resolved_initial,
            device=device_obj,
            num_steps=steps,
            checkpoints_path=checkpoints_dir,
            seed=seed,
            batch_size=effective_batch,
            minibatch_size=effective_minibatch,
            num_envs=resolved_num_envs,
            num_workers=resolved_num_workers,
            use_rnn=use_rnn,
            checkpoint_interval=checkpoint_interval,
            vector_backend=backend,
            game_name=representative_game,
        )

        console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_dir}[/green]")


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


if __name__ == "__main__":  # pragma: no cover
    app()
