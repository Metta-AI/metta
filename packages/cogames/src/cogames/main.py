#!/usr/bin/env -S uv run

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Literal, Optional

import typer
import yaml
from packaging.version import Version
from rich.table import Table

from cogames import curricula, game
from cogames import evaluate as evaluate_module
from cogames import play as play_module
from cogames import train as train_module
from cogames.cli.base import console
from cogames.cli.mission import describe_mission, get_mission_name_and_config
from cogames.cli.policy import get_policy_spec, get_policy_specs, policy_arg_example, policy_arg_w_proportion_example
from cogames.device import resolve_training_device
from mettagrid import MettaGridConfig, MettaGridEnv

# Always add current directory to Python path
sys.path.insert(0, ".")

logger = logging.getLogger("cogames.main")


app = typer.Typer(
    help="CoGames - Multi-agent cooperative and competitive games",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command("missions", help="List all available missions, or describe a specific mission")
@app.command("games", hidden=True)
@app.command("mission", hidden=True)
def games_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Argument(None, help="Name of the mission"),
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None, "--format", help="Output mission configuration in YAML or JSON."
    ),
    save: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save",
        "-s",
        help="Save mission configuration to file (YAML or JSON)",
    ),
) -> None:
    resolved_mission, env_cfg = get_mission_name_and_config(ctx, mission)

    if save is not None:
        try:
            game.save_mission_config(env_cfg, save)
            console.print(f"[green]Mission configuration saved to: {save}[/green]")
        except ValueError as exc:  # pragma: no cover - user input
            console.print(f"[red]Error saving configuration: {exc}[/red]")
            raise typer.Exit(1) from exc
        return

    if format_ is not None:
        try:
            data = env_cfg.model_dump(mode="json")
            if format_ == "json":
                console.print(json.dumps(data, indent=2))
            else:
                console.print(yaml.safe_dump(data, sort_keys=False))
        except Exception as exc:  # pragma: no cover - serialization errors
            console.print(f"[red]Error formatting configuration: {exc}[/red]")
            raise typer.Exit(1) from exc
        return

    try:
        describe_mission(resolved_mission, env_cfg)
    except ValueError as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(name="play", help="Play a game")
def play_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Argument(None, help="Mission name"),
    policy: Optional[str] = typer.Argument("noop", help=f"Policy ({policy_arg_example})"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Run in interactive mode"),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of steps to run", min=1),
    render: Literal["gui", "text", "none"] = typer.Option(
        "gui", "--render", "-r", help="Render mode: 'gui', 'text', or 'none' (no rendering)"
    ),
) -> None:
    resolved_mission, env_cfg = get_mission_name_and_config(ctx, mission)
    policy_spec = get_policy_spec(ctx, policy)
    console.print(f"[cyan]Playing {resolved_mission}[/cyan]")
    console.print(f"Max Steps: {steps}, Interactive: {interactive}, Render: {render}")

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        max_steps=steps,
        seed=42,
        render=render,
        verbose=interactive,
        game_name=resolved_mission,
    )


@app.command("make-mission", help="Create a new mission configuration")
@app.command("make-game", hidden=True)
def make_mission(
    ctx: typer.Context,
    base_mission: Optional[str] = typer.Argument(None, help="Base mission to start configuring from"),
    num_agents: int = typer.Option(2, "--agents", "-a", help="Number of agents", min=1),
    width: int = typer.Option(10, "--width", "-w", help="Map width", min=1),
    height: int = typer.Option(10, "--height", "-h", help="Map height", min=1),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (yml or json)"),  # noqa: B008
) -> None:
    try:
        resolved_mission, env_cfg = get_mission_name_and_config(ctx, base_mission)

        # Update map dimensions
        env_cfg.game.map_builder.width = width  # type: ignore[attr-defined]
        env_cfg.game.map_builder.height = height  # type: ignore[attr-defined]
        env_cfg.game.num_agents = num_agents

        # Validate the environment configuration
        _ = MettaGridEnv(env_cfg)

        if output:
            game.save_mission_config(env_cfg, output)
            console.print(f"[green]Game configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(name="train", help="Train a policy on a mission")
def train_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Argument(None, help="Name of the mission to train on"),
    policy: Optional[str] = typer.Argument("simple", help=f"Policy ({policy_arg_example})"),
    checkpoints_path: str = typer.Option(
        "./train_dir",
        "--checkpoints",
        help="Path to save training data",
    ),
    steps: int = typer.Option(10_000_000_000, "--steps", "-s", help="Number of training steps", min=1),
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
    vector_batch_size: Optional[int] = typer.Option(
        None,
        "--vector-batch-size",
        help="Override vectorized environment batch size",
        min=1,
    ),
) -> None:
    rotation_aliases = {"training_rotation", "training_facility_rotation", "training_cycle"}
    rotation_easy_aliases = {"training_rotation_easy"}
    rotation_shaped_aliases = {"training_rotation_shaped"}
    rotation_easy_shaped_aliases = {"training_rotation_easy_shaped"}

    env_cfg: Optional[MettaGridConfig] = None
    curriculum_supplier: Optional[Callable[[], MettaGridConfig]] = None
    resolved_mission = mission

    if mission in rotation_aliases:
        curriculum_supplier = curricula.training_rotation()
    elif mission in rotation_easy_aliases:
        curriculum_supplier = curricula.training_rotation_easy()
    elif mission in rotation_shaped_aliases:
        curriculum_supplier = curricula.training_rotation_shaped()
    elif mission in rotation_easy_shaped_aliases:
        curriculum_supplier = curricula.training_rotation_easy_shaped()
    else:
        resolved_mission, env_cfg = get_mission_name_and_config(ctx, mission)

    policy_spec = get_policy_spec(ctx, policy)
    torch_device = resolve_training_device(console, device)

    try:
        train_module.train(
            env_cfg=env_cfg,
            policy_class_path=policy_spec.policy_class_path,
            initial_weights_path=policy_spec.policy_data_path,
            device=torch_device,
            num_steps=steps,
            checkpoints_path=Path(checkpoints_path),
            seed=seed,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            vector_num_workers=num_workers,
            vector_num_envs=parallel_envs,
            vector_batch_size=vector_batch_size,
            game_name=resolved_mission,
            env_cfg_supplier=curriculum_supplier,
        )

    except ValueError as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Training complete. Checkpoints saved to: {checkpoints_path}[/green]")


@app.command(
    name="eval",
    help="Evaluate one or more policies on a mission",
)
@app.command("evaluate", hidden=True)
def evaluate_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Argument(None, help="Name of the mission"),
    policies: Optional[list[str]] = typer.Argument(  # noqa: B008
        None, help=f"Policies to evaluate: ({policy_arg_w_proportion_example}...)"
    ),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes", min=1),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        help="Max milliseconds afforded to generate each action before noop is used by default",
        min=1,
    ),
) -> None:
    resolved_mission, env_cfg = get_mission_name_and_config(ctx, mission)
    policy_specs = get_policy_specs(ctx, policies)

    console.print(
        f"[cyan]Evaluating {len(policy_specs)} policies on {resolved_mission} over {episodes} episodes[/cyan]"
    )

    evaluate_module.evaluate(
        console,
        resolved_game=resolved_mission,
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
