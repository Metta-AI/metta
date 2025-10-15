#!/usr/bin/env -S uv run

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional, TypeVar

import typer
import yaml
from click.core import ParameterSource
from packaging.version import Version
from rich.table import Table

from cogames import evaluate as evaluate_module
from cogames import game
from cogames import play as play_module
from cogames import train as train_module
from cogames.cli.base import console
from cogames.cli.mission import describe_mission, get_mission_name_and_config, get_mission_names_and_configs
from cogames.cli.policy import get_policy_spec, get_policy_specs, policy_arg_example, policy_arg_w_proportion_example
from cogames.curricula import make_rotation
from cogames.device import resolve_training_device
from mettagrid import MettaGridEnv

# Always add current directory to Python path
sys.path.insert(0, ".")

logger = logging.getLogger("cogames.main")


T = TypeVar("T")


app = typer.Typer(
    help="CoGames - Multi-agent cooperative and competitive games",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command("missions", help="List all available missions, or describe a specific mission")
@app.command("games", hidden=True)
@app.command("mission", hidden=True)
def games_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Name of the mission"),
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
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Name of the mission"),
    policy: str = typer.Option("noop", "--policy", "-p", help=f"Policy ({policy_arg_example})"),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of steps to run", min=1),
    render: Literal["gui", "unicode", "none"] = typer.Option(
        "gui", "--render", "-r", help="Render mode: 'gui', 'unicode' (interactive terminal), or 'none'"
    ),
) -> None:
    resolved_mission, env_cfg = get_mission_name_and_config(ctx, mission)
    policy_spec = get_policy_spec(ctx, policy)
    console.print(f"[cyan]Playing {resolved_mission}[/cyan]")
    console.print(f"Max Steps: {steps}, Render: {render}")

    if ctx.get_parameter_source("steps") in (
        ParameterSource.COMMANDLINE,
        ParameterSource.ENVIRONMENT,
        ParameterSource.PROMPT,
    ):
        env_cfg.game.max_steps = steps

    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_spec=policy_spec,
        max_steps=steps,
        seed=42,
        render=render,
        game_name=resolved_mission,
    )


@app.command("make-mission", help="Create a new mission configuration")
@app.command("make-game", hidden=True)
def make_mission(
    ctx: typer.Context,
    base_mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Base mission to start configuring from"),
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
            console.print(f"[green]Modified {resolved_mission} configuration saved to: {output}[/green]")
        else:
            console.print("\n[yellow]To save this configuration, use the --output option.[/yellow]")

    except Exception as exc:  # pragma: no cover - user input
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command(name="train", help="Train a policy on a mission")
def train_cmd(
    ctx: typer.Context,
    missions: Optional[list[str]] = typer.Option(None, "--mission", "-m", help="Missions to train on"),  # noqa B008
    policy: str = typer.Option("simple", "--policy", "-p", help=f"Policy ({policy_arg_example})"),
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
    selected_missions = get_mission_names_and_configs(ctx, missions)
    if len(selected_missions) == 1:
        mission_name, env_cfg = selected_missions[0]
        supplier = None
        console.print(f"Training on mission: {mission_name}\n")
    elif len(selected_missions) > 1:
        env_cfg = None
        supplier = make_rotation(selected_missions)
        console.print("Training on missions:\n" + "\n".join(f"- {m}" for m, _ in selected_missions) + "\n")
    else:
        # Should not get here
        raise ValueError("Please specify at least one mission")

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
            env_cfg_supplier=supplier,
            missions_arg=missions,
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
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Name of the mission"),
    policies: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--policy",
        "-p",
        help=f"Policies to evaluate: ({policy_arg_w_proportion_example}...)",
    ),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of evaluation episodes", min=1),
    action_timeout_ms: int = typer.Option(
        250,
        "--action-timeout-ms",
        help="Max milliseconds afforded to generate each action before noop is used by default",
        min=1,
    ),
    steps: Optional[int] = typer.Option(1000, "--steps", "-s", help="Max steps per episode", min=1),
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
        max_steps=steps,
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
