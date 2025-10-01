"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import contextlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

# Always add current directory to Python path
sys.path.insert(0, ".")

import torch
import typer
from rich.console import Console

from cogames import curriculum as curriculum_utils
from cogames import game, play, serialization, train, utils
from click.core import ParameterSource
from mettagrid import MettaGridConfig

from cogames.env import make_hierarchical_env
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


# Mapping of shorthand policy names to full class paths
POLICY_SHORTCUTS = {
    "random": "cogames.policy.random.RandomPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
}


def resolve_policy_class_path(policy: str) -> str:
    """Return the full class path for a policy shorthand (``random``, ``simple``, ``lstm``)."""
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
        try:
            env_cfgs, env_names = curriculum_utils.collect_curriculum_configs(
                selected_games,
                curriculum_path=curriculum,
                max_items=max_items,
                fallback_folder=None,
                game_param="game_name",
            )
        except curriculum_utils.CurriculumArgumentError as exc:
            raise typer.BadParameter(str(exc), param_name=exc.param_name or "game") from exc

        if not env_cfgs:
            raise typer.BadParameter("no games or curriculum items to export", param_name="game")

        if output_dir is not None:
            destination = output_dir.expanduser().resolve()
        else:
            destination = (utils.resolve_run_dir(BASE_RUNS_DIR, None) / "curricula").resolve()

        curriculum_utils.dump_game_configs(env_cfgs, env_names, destination)

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
        resolved_run_dir = utils.resolve_run_dir(BASE_RUNS_DIR, run_dir)
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
        "cogames.policy.lstm.LSTMPolicy", "--policy", help="Path to policy class"
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
    num_variants: Annotated[int, typer.Option("--num-variants", help="Number of map variants to generate")] = 1,
    variant_key: Annotated[
        Optional[str],
        typer.Option("--key", help="Dot path to MettaGridConfig field to sweep (e.g. 'game.map.width')"),
    ] = None,
    variant_min: Annotated[Optional[float], typer.Option("--min", help="Minimum value for sweep")] = None,
    variant_max: Annotated[Optional[float], typer.Option("--max", help="Maximum value for sweep")] = None,
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

            base_config = make_game(
                num_cogs=num_agents,
                num_assemblers=1,
                num_chests=1,
            )

            def apply_common_fields(cfg: MettaGridConfig) -> MettaGridConfig:
                cfg.game.map_builder.width = width
                cfg.game.map_builder.height = height
                cfg.game.num_agents = num_agents
                return cfg

            def cast_variant_value(current: Any, raw: float) -> Any:
                if isinstance(current, bool):
                    return bool(raw)
                if isinstance(current, int) and not isinstance(current, bool):
                    return int(round(raw))
                if isinstance(current, float):
                    return float(raw)
                return raw

            def resolve_attr_value(obj: Any, path: str) -> Any:
                target = obj
                for part in path.split("."):
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        raise typer.BadParameter(f"Unknown config key '{path}'.", param_name="key")
                return target

            def assign_attr(obj: Any, path: str, raw_value: float) -> Any:
                parts = path.split(".")
                target = obj
                for part in parts[:-1]:
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        raise typer.BadParameter(f"Unknown config key '{path}'.", param_name="key")

                attr = parts[-1]
                if not hasattr(target, attr):
                    raise typer.BadParameter(f"Unknown config key '{path}'.", param_name="key")

                current_value = getattr(target, attr)
                casted_value = cast_variant_value(current_value, raw_value)
                setattr(target, attr, casted_value)
                return casted_value

            if num_variants < 1:
                raise typer.BadParameter("--num-variants must be >= 1", param_name="num_variants")

            if num_variants > 1 and variant_key is None:
                raise typer.BadParameter("Provide --key when generating multiple variants", param_name="key")

            if variant_key is not None:
                if variant_min is None and variant_max is None:
                    raise typer.BadParameter("Provide --min and/or --max when using --key", param_name="min")
                if variant_min is not None and variant_max is not None and variant_min > variant_max:
                    raise typer.BadParameter("--min must be <= --max", param_name="min")

            variant_values: list[Optional[float]]
            if variant_key:
                if num_variants == 1:
                    probe_cfg = apply_common_fields(base_config.model_copy(deep=True))
                    current_value = resolve_attr_value(probe_cfg, variant_key)
                    base_value = float(current_value) if isinstance(current_value, (int, float)) else 0.0
                    if variant_min is not None:
                        base_value = variant_min
                    if variant_max is not None:
                        base_value = variant_max
                    variant_values = [base_value]
                else:
                    start = variant_min if variant_min is not None else variant_max or 0.0
                    end = variant_max if variant_max is not None else variant_min or start
                    denominator = max(num_variants - 1, 1)
                    span = end - start
                    variant_values = [start + span * (idx / denominator) for idx in range(num_variants)]
            else:
                variant_values = [None] * num_variants

            variants: list[MettaGridConfig] = []
            for raw_value in variant_values:
                cfg = apply_common_fields(base_config.model_copy(deep=True))
                if variant_key and raw_value is not None:
                    assign_attr(cfg, variant_key, raw_value)
                variants.append(cfg)

            base_label = getattr(variants[0].game, "name", base_game or "map")
            safe_base_label = curriculum_utils.sanitize_map_name(str(base_label))

            if num_variants > 1:
                if output is None:
                    raise typer.BadParameter(
                        "Provide --output pointing to a directory when generating variants.",
                        param_name="output",
                    )
                output_dir = output
                if output_dir.suffix:
                    raise typer.BadParameter(
                        "--output must be a directory (no filename) when generating variants.",
                        param_name="output",
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                for idx, cfg in enumerate(variants):
                    filename = output_dir / f"{safe_base_label}_{idx:03d}.yaml"
                    game.save_game_config(cfg, filename)
                if variant_key:
                    formatted = ", ".join(f"{val:g}" for val in variant_values if val is not None)
                    console.print(
                        f"[green]Generated {len(variants)} variants at: {output_dir}[/green]"
                        + (f" [dim](values: {formatted})[/dim]" if formatted else "")
                    )
                else:
                    console.print(f"[green]Generated {len(variants)} variants at: {output_dir}[/green]")
            else:
                cfg = variants[0]
                if output:
                    if output.is_dir() or not output.suffix:
                        output.mkdir(parents=True, exist_ok=True)
                        target_path = (output / f"{safe_base_label}.yaml").resolve()
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        target_path = output.resolve()
                    game.save_game_config(cfg, target_path)
                    message = f"[green]Game configuration saved to: {target_path}[/green]"
                    if variant_key and variant_values[0] is not None:
                        message += f" [dim]({variant_key}={variant_values[0]:g})[/dim]"
                    console.print(message)
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
    ] = "cogames.policy.lstm.LSTMPolicy",
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
        resolved_run_dir = utils.resolve_run_dir(BASE_RUNS_DIR, run_dir)
        resolved_run_dir.mkdir(parents=True, exist_ok=True)

        backend = vector_backend.lower()

        def resolve_training_device(requested: str) -> torch.device:
            normalized = requested.strip().lower()

            def cuda_usable() -> bool:
                cuda_backend = getattr(torch.backends, "cuda", None)
                if cuda_backend is None or not getattr(cuda_backend, "is_built", lambda: False)():
                    return False
                try:
                    return torch.cuda.is_available()
                except (AssertionError, RuntimeError):
                    return False

            def mps_usable() -> bool:
                return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

            if normalized == "auto":
                if cuda_usable():
                    return torch.device("cuda")
                if mps_usable():
                    return torch.device("mps")
                console.print("[yellow]CUDA/MPS unavailable; training will run on CPU.[/yellow]")
                return torch.device("cpu")

            try:
                candidate = torch.device(requested)
            except (RuntimeError, ValueError):
                console.print(f"[yellow]Warning: Unknown device '{requested}'. Falling back to CPU.[/yellow]")
                return torch.device("cpu")

            if candidate.type == "cuda" and not cuda_usable():
                console.print("[yellow]CUDA requested but unavailable. Training will run on CPU instead.[/yellow]")
                return torch.device("cpu")

            if candidate.type == "mps" and not mps_usable():
                console.print("[yellow]MPS requested but unavailable. Training will run on CPU instead.[/yellow]")
                return torch.device("cpu")

            return candidate

        torch_device = resolve_training_device(device)
        steps_source = ctx.get_parameter_source("steps") if ctx else None
        batch_source = ctx.get_parameter_source("batch_size") if ctx else None
        minibatch_source = ctx.get_parameter_source("minibatch_size") if ctx else None

        if torch_device.type == "cuda":
            if steps_source in (None, ParameterSource.DEFAULT):
                steps = 50_000_000
                console.print(
                    "[cyan]Auto-adjusting training steps to 50,000,000 for CUDA runs. "
                    "Override with --steps if needed.[/cyan]"
                )
            if batch_source in (None, ParameterSource.DEFAULT):
                batch_size = 524_288
                console.print(
                    "[cyan]Auto-adjusting batch size to 524,288 for CUDA runs. "
                    "Override with --batch-size if needed.[/cyan]"
                )
            if minibatch_source in (None, ParameterSource.DEFAULT):
                minibatch_size = 16_384
                console.print(
                    "[cyan]Auto-adjusting minibatch size to 16,384 for CUDA runs. "
                    "Override with --minibatch-size if needed.[/cyan]"
                )
        resolved_device_type = torch_device.type
        resolved_num_envs, resolved_num_workers = utils.suggest_parallelism(
            resolved_device_type,
            num_envs,
            num_workers,
        )
        if vector_backend.lower() == "multiprocessing" and sys.platform == "darwin":
            resolved_num_workers = 1

        fallback_curricula = resolved_run_dir / "curricula"
        base_games = [game_name] if game_name is not None else []
        if not base_games and curriculum is None and not fallback_curricula.exists():
            base_games = list(curriculum_utils.DEFAULT_BIOME_GAMES)
        try:
            env_cfgs, env_names = curriculum_utils.collect_curriculum_configs(
                base_games,
                curriculum_path=curriculum,
                max_items=max(resolved_num_envs, 32),
                fallback_folder=fallback_curricula,
                game_param="game_name",
            )
        except curriculum_utils.CurriculumArgumentError as exc:
            raise typer.BadParameter(str(exc), param_name=exc.param_name or "game_name") from exc
        env_cfgs, env_names, dropped = utils.filter_uniform_agent_count(env_cfgs, env_names)
        if dropped:
            agent_count = env_cfgs[0].game.num_agents if env_cfgs else "?"
            console.print(
                "[yellow]Skipping {dropped} map(s) with mismatched agent counts. "
                "Training will use configs with {agents} agent(s).[/yellow]".format(
                    dropped=dropped,
                    agents=agent_count,
                )
            )

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
            resolved_initial = utils.resolve_initial_weights(initial_weights_path)

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

        if checkpoints_path is not None:
            checkpoints_dir = checkpoints_path.expanduser().resolve()
        else:
            checkpoints_dir = (resolved_run_dir / "checkpoints").resolve()
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if map_dump_dir is not None:
            maps_dir = map_dump_dir if map_dump_dir.is_absolute() else (resolved_run_dir / map_dump_dir).resolve()
        else:
            maps_dir = (resolved_run_dir / "curricula").resolve()

        if not maps_dir.exists() or not any(maps_dir.glob("*.yaml")):
            curriculum_utils.dump_game_configs(env_cfgs, env_names, maps_dir)

        full_policy_path = resolve_policy_class_path(policy_class_path)

        train.train(
            env_cfgs=env_cfgs,
            policy_class_path=full_policy_path,
            initial_weights_path=resolved_initial,
            device=torch_device,
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
    env = make_hierarchical_env(env_cfg)
    policy = serialization.load_policy_from_bundle(bundle_dir, env, torch.device(device))
    policy.reset()
    console.print(f"[green]Loaded policy {policy.__class__.__name__} on {device}[/green]")


@policy_app.command("inspect")
def policy_inspect(bundle_dir: Annotated[Path, typer.Argument(help="Path to policy bundle directory")]) -> None:
    metadata = serialization.inspect_bundle(bundle_dir)
    console.print_json(data=metadata)


if __name__ == "__main__":  # pragma: no cover
    app()
