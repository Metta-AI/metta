#!/usr/bin/env -S uv run

"""CLI for CoGames - collection of environments for multi-agent cooperative and competitive games."""

import importlib.metadata
import importlib.util
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, TypeVar

import typer
import yaml  # type: ignore[import]
from click.core import ParameterSource
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from cogames import evaluate as evaluate_module
from cogames import game, verbose
from cogames import play as play_module
from cogames import train as train_module
from cogames.cli.base import console
from cogames.cli.leaderboard import leaderboard_cmd, submissions_cmd
from cogames.cli.login import DEFAULT_COGAMES_SERVER, perform_login
from cogames.cli.mission import (
    describe_mission,
    get_mission_name_and_config,
    list_evals,
    list_missions,
    list_variants,
)
from cogames.cli.policy import (
    get_policy_spec,
    get_policy_specs_with_proportions,
    policy_arg_example,
    policy_arg_w_proportion_example,
)
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER, submit_command
from cogames.cli.utils import init_suppress_warnings
from cogames.curricula import make_rotation
from cogames.device import resolve_training_device
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.policy.loader import discover_and_register_policies
from mettagrid.policy.policy_registry import get_policy_registry
from mettagrid.renderer.renderer import RenderMode

# Always add current directory to Python path
sys.path.insert(0, ".")

init_suppress_warnings()
logger = logging.getLogger("cogames.main")


T = TypeVar("T")


def _resolve_mettascope_script() -> Path:
    spec = importlib.util.find_spec("mettagrid")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("mettagrid package is not available; cannot locate MettaScope.")

    package_dir = Path(spec.origin).resolve().parent
    search_roots = (package_dir, *package_dir.parents)

    for root in search_roots:
        candidate = root / "nim" / "mettascope" / "src" / "mettascope.nim"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"MettaScope sources not found relative to installed mettagrid package (searched from {package_dir})."
    )


app = typer.Typer(
    help="CoGames - Multi-agent cooperative and competitive games",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    callback=lambda: discover_and_register_policies("cogames.policy"),
)


@app.command(name="tutorial", help="Print instructions on how to play CvC and runs cogames play --mission tutorial")
def tutorial_cmd(
    ctx: typer.Context,
) -> None:
    """Run the CoGames tutorial."""
    # Suppress logs during tutorial to keep instructions visible
    logging.getLogger().setLevel(logging.ERROR)

    console.print(
        Panel.fit(
            "[bold cyan]MISSION BRIEFING: Tutorial Sector[/bold cyan]\n\n"
            "Welcome, Cognitive. This simulation mirrors frontline HEART ops.\n"
            "We will launch the Mettascope visual interface now.\n\n"
            "When you are ready to deploy, press Enter below.",
            title="Mission Briefing",
            border_style="green",
        )
    )

    Prompt.ask("[dim]Press Enter to launch simulation[/dim]", default="", show_default=False)
    console.print("[dim]Initializing Mettascope...[/dim]")

    # Load tutorial mission
    from cogames.cogs_vs_clips.tutorial_missions import TutorialMission

    # Create environment config
    env_cfg = TutorialMission.make_env()
    # Force 1 agent for tutorial
    env_cfg.game.num_agents = 1

    def run_tutorial_steps():
        # Wait a moment for the window to appear
        time.sleep(3)

        tutorial_steps = (
            {
                "title": "Step 1 — Interface & Controls",
                "lines": (
                    "Left Pane (Intel): Shows details for selected objects (Stations, Tiles, Cogs).",
                    "Right Pane (Vibe Deck): Select icons here to change your Cog's broadcast resonance.",
                    "Zoom/Pan: Scroll or pinch to zoom the arena; drag to pan.",
                    "Click your Cog (or their portrait) to focus the camera on them.",
                ),
            },
            {
                "title": "Step 2 — Movement & Energy",
                "lines": (
                    "Use WASD or Arrow Keys to move your Cog.",
                    "Every move costs Energy. Watch your battery bar on the Cog or in the HUD.",
                    "If low, rest (skip turn) or find a Charger [yellow]+[/yellow].",
                ),
            },
            {
                "title": "Step 3 — Extraction",
                "lines": (
                    "Primary interaction mode is WALKING INTO things.",
                    "Locate an Extractor station:",
                    "  [yellow]C[/yellow] (Carbon), [yellow]O[/yellow] (Oxygen),",
                    "  [yellow]G[/yellow] (Germanium), [yellow]S[/yellow] (Silicon).",
                    "Walk into it to extract resources.",
                    "Note: Silicon ([yellow]S[/yellow]) costs 20 energy!",
                ),
            },
            {
                "title": "Step 4 — Crafting (Assembler)",
                "lines": (
                    "Click the central Assembler [yellow]&[/yellow] to see the HEART recipe in the Left Pane.",
                    "Set your Vibe (Right Pane) to match the requirement (usually [red]heart_a[/red]).",
                    "Walk into the Assembler to craft. Inputs are taken from your inventory instantly.",
                ),
            },
            {
                "title": "Step 5 — Deposit (Chest)",
                "lines": (
                    "Go to the Chest [yellow]C[/yellow] (usually near the center).",
                    "Switch your Vibe to [red]heart_b[/red] (Deposit Mode).",
                    "Walk into the Chest to deposit the HEART and complete the objective.",
                    "Note: To pull resources out of the Chest, you must vibe the matching resource *_a protocol.",
                ),
            },
        )

        for idx, step in enumerate(tutorial_steps):
            console.print()
            console.print(f"[bold cyan]{step['title']}[/bold cyan]")
            console.print()
            for line in step["lines"]:
                console.print(f"  • {line}")
            console.print()
            if idx < len(tutorial_steps) - 1:
                Prompt.ask("[dim]Press Enter for next step[/dim]", default="", show_default=False)

        console.print(
            "[bold green]REFERENCE DOSSIERS[/bold green]\n"
            "- [link=packages/cogames/MISSION.md]MISSION.md[/link]: Machina VII deployment orders.\n"
            "- [link=packages/cogames/README.md]README.md[/link]: System overview and CLI quick start.\n"
            "- [link=packages/cogames/TECHNICAL_MANUAL.md]TECHNICAL_MANUAL.md[/link]: FACE sensor/command schematics."
        )
        console.print()
        console.print("[dim]Tutorial briefing complete. Good luck, Cognitive.[/dim]")

    # Start tutorial interaction in a background thread
    tutorial_thread = threading.Thread(target=run_tutorial_steps, daemon=True)
    tutorial_thread.start()

    # Run play (blocks main thread)
    play_module.play(
        console,
        env_cfg=env_cfg,
        policy_spec=get_policy_spec(ctx, "noop"),  # Default to noop, assuming human control
        game_name="tutorial",
        render_mode="gui",
    )


@app.command("missions", help="List all available missions, or describe a specific mission")
def missions_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Name of the mission (optional)"),
) -> None:
    """List all missions or describe a specific one."""
    if mission:
        resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission)
        describe_mission(resolved_mission, env_cfg, mission_cfg)
    else:
        # Show table of all missions
        console.print("[bold]Available Missions[/bold]")
        list_missions(console)
        console.print()
        console.print("Run [green]cogames missions --mission [NAME][/green] for details.")


@app.command("evals", help="List all available evaluations")
def evals_cmd(
    ctx: typer.Context,
) -> None:
    """List all evaluations."""
    list_evals(console)


@app.command("variants", help="List all available variants")
def variants_cmd(
    ctx: typer.Context,
) -> None:
    """List all variants."""
    list_variants(console)


@app.command("policies", help="List all available policies")
def policies_cmd(
    ctx: typer.Context,
) -> None:
    """List all available policies."""
    registry = get_policy_registry()

    table = Table(title="Available Policies", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Description")

    for name, entry in registry.list_policies().items():
        table.add_row(name, entry.cls.__name__, entry.description or "")

    console.print(table)


@app.command(name="train", help="Train a policy on a mission")
def train_cmd(
    ctx: typer.Context,
    mission: list[str] = typer.Option(..., "--mission", "-m", help="Name of the mission (can be used multiple times)"),  # noqa: B008
    policy: str = typer.Option(..., "--policy", "-p", help=f"Policy ({policy_arg_example})"),  # noqa: B008
    cogs: Optional[int] = typer.Option(None, "--cogs", "-c", help="Number of cogs (agents)"),  # noqa: B008
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        help="Mission variant (can be used multiple times, e.g., --variant solar_flare --variant dark_side)",
    ),
    steps: int = typer.Option(10000, "--steps", "-s", help="Number of training steps"),  # noqa: B008
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cpu, cuda, mps)"),  # noqa: B008
    batch_size: int = typer.Option(4096, "--batch-size", "-b", help="Batch size per update"),  # noqa: B008
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w", help="Number of worker processes"),  # noqa: B008
    seed: int = typer.Option(42, "--seed", help="Random seed"),  # noqa: B008
    wandb_project: str = typer.Option("cogames", "--wandb-project", help="W&B project name"),  # noqa: B008
    wandb_entity: Optional[str] = typer.Option(None, "--wandb-entity", help="W&B entity/team name"),  # noqa: B008
    wandb_group: Optional[str] = typer.Option(None, "--wandb-group", help="W&B run group"),  # noqa: B008
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name", help="W&B run name"),  # noqa: B008
    checkpoint_dir: Path = typer.Option(  # noqa: B008
        Path("train_dir"), "--checkpoint-dir", help="Directory to save checkpoints"
    ),
    eval_interval: Optional[int] = typer.Option(  # noqa: B008
        None, "--eval-interval", help="Steps between evaluations (default: 10% of total steps)"
    ),
    save_interval: Optional[int] = typer.Option(  # noqa: B008
        None, "--save-interval", help="Steps between checkpoints (default: 20% of total steps)"
    ),
) -> None:
    # Resolve device
    device_str = resolve_training_device(device)
    console.print(f"[cyan]Training on {device_str}[/cyan]")

    # Create curriculum from missions
    # Parse missions, variants, and cogs
    # We need to handle the fact that cogs/variant are applied to ALL missions if provided once,
    # or we'd need a complex syntax. For now, apply to all.

    task_names = mission  # Renamed for clarity

    # Create rotation
    try:
        rotation = make_rotation(task_names, variant, cogs, seed=seed)
    except ValueError as exc:
        console.print(f"[red]Error creating curriculum: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"Training curriculum: {[t.name for t in rotation.tasks]}")

    # Setup policy
    policy_spec = get_policy_spec(ctx, policy)

    # Train
    train_module.train(
        console,
        rotation=rotation,
        policy_spec=policy_spec,
        total_steps=steps,
        device=device_str,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_name=wandb_name,
        checkpoint_dir=checkpoint_dir,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )


@app.command(name="eval", help="Evaluate a policy on a mission")
def eval_cmd(
    ctx: typer.Context,
    mission: list[str] = typer.Option(..., "--mission", "-m", help="Name of the mission (can be used multiple times)"),  # noqa: B008
    policy: list[str] = typer.Option(..., "--policy", "-p", help=f"Policy ({policy_arg_w_proportion_example})"),  # noqa: B008
    cogs: Optional[int] = typer.Option(None, "--cogs", "-c", help="Number of cogs (agents)"),  # noqa: B008
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        help="Mission variant (can be used multiple times, e.g., --variant solar_flare --variant dark_side)",
    ),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of episodes per mission"),  # noqa: B008
    steps: int = typer.Option(1000, "--steps", "-s", help="Max steps per episode"),  # noqa: B008
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results (json/yaml)"),  # noqa: B008
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (json/yaml)"),  # noqa: A002, B008
    seed: int = typer.Option(42, "--seed", help="Random seed"),  # noqa: B008
    render: RenderMode = typer.Option("none", "--render", "-r", help="Render mode"),  # noqa: B008
    action_timeout_ms: int = typer.Option(  # noqa: B008
        250, "--action-timeout-ms", help="Timeout per action in ms (for mettascope)"
    ),
) -> None:
    # Parse policies
    policy_specs = get_policy_specs_with_proportions(ctx, policy)

    # Create rotation
    try:
        rotation = make_rotation(mission, variant, cogs, seed=seed)
    except ValueError as exc:
        console.print(f"[red]Error creating evaluation set: {exc}[/red]")
        raise typer.Exit(1) from exc

    # Run evaluation
    results = evaluate_module.evaluate(
        console,
        rotation=rotation,
        policy_specs=policy_specs,
        num_episodes=episodes,
        max_steps=steps,
        seed=seed,
        render_mode=render,
    )

    # Output results
    if output or format:
        # Serialize results
        results_dict = results.to_dict()

        # Determine format
        fmt = format
        if not fmt and output:
            if output.suffix in (".json", ".yaml", ".yml"):
                fmt = output.suffix.lstrip(".")
            else:
                fmt = "json"

        if not fmt:
            fmt = "json"

        if fmt == "json":
            content = json.dumps(results_dict, indent=2)
        elif fmt in ("yaml", "yml"):
            content = yaml.dump(results_dict, sort_keys=False)
        else:
            console.print(f"[red]Unknown format: {fmt}[/red]")
            raise typer.Exit(1)

        if output:
            output.write_text(content)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            console.print(content)
    else:
        # Print table
        evaluate_module.print_results(console, results)


@app.command(name="login", help="Login to CoGames server")
def login_cmd(
    ctx: typer.Context,
    server: str = typer.Option(DEFAULT_COGAMES_SERVER, "--server", help="CoGames server URL"),  # noqa: B008
    force: bool = typer.Option(False, "--force", "-f", help="Force re-login"),  # noqa: B008
) -> None:
    perform_login(server, force=force)


@app.command(name="submit", help="Submit a policy to the leaderboard")
def submit_cmd(
    ctx: typer.Context,
    policy: str = typer.Argument(..., help=f"Policy ({policy_arg_example})"),  # noqa: B008
    mission: str = typer.Option(..., "--mission", "-m", help="Mission to submit for"),  # noqa: B008
    server: str = typer.Option(DEFAULT_SUBMIT_SERVER, "--server", help="Submission server URL"),  # noqa: B008
) -> None:
    policy_spec = get_policy_spec(ctx, policy)
    submit_command(console, policy_spec, mission, server)


@app.command(name="describe", help="Describe a mission and its configuration")
def describe_cmd(
    ctx: typer.Context,
    mission: str = typer.Argument(..., help="Mission name (e.g., hello_world.open_world)"),  # noqa: B008
    cogs: Optional[int] = typer.Option(None, "--cogs", "-c", help="Number of cogs (agents)"),  # noqa: B008
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        help="Mission variant (can be used multiple times, e.g., --variant solar_flare --variant dark_side)",
    ),
) -> None:
    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission, variant, cogs)
    describe_mission(resolved_mission, env_cfg, mission_cfg)


@app.command(name="play", help="Play a game")
def play_cmd(
    ctx: typer.Context,
    mission: Optional[str] = typer.Option(None, "--mission", "-m", help="Name of the mission"),  # noqa: B008
    cogs: Optional[int] = typer.Option(None, "--cogs", "-c", help="Number of cogs (agents)"),  # noqa: B008
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        help="Mission variant (can be used multiple times, e.g., --variant solar_flare --variant dark_side)",
    ),
    policy: str = typer.Option("noop", "--policy", "-p", help=f"Policy ({policy_arg_example})"),  # noqa: B008
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of steps to run", min=1),  # noqa: B008
    render: RenderMode = typer.Option("gui", "--render", "-r", help="Render mode"),  # noqa: B008
    seed: int = typer.Option(42, "--seed", help="Seed for the simulator and policy", min=0),  # noqa: B008
    map_seed: Optional[int] = typer.Option(  # noqa: B008
        None,
        "--map-seed",
        help="Override MapGen seed for procedural maps (defaults to --seed if not set)",
        min=0,
    ),
    print_cvc_config: bool = typer.Option(  # noqa: B008
        False, "--print-cvc-config", help="Print Mission config (CVC config) and exit"
    ),
    print_mg_config: bool = typer.Option(False, "--print-mg-config", help="Print MettaGridConfig and exit"),  # noqa: B008
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        help=(
            "Directory to save replay. Directory will be created if it doesn't exist. "
            "Replay will be saved with a unique UUID-based filename."
        ),
    ),
) -> None:
    resolved_mission, env_cfg, mission_cfg = get_mission_name_and_config(ctx, mission, variant, cogs)

    if print_cvc_config or print_mg_config:
        try:
            verbose.print_configs(console, env_cfg, mission_cfg, print_cvc_config, print_mg_config)
        except Exception as exc:
            console.print(f"[red]Error printing config: {exc}[/red]")
            raise typer.Exit(1) from exc

    # Optionally override MapGen seed so maps are reproducible across runs.
    # This uses --map-seed if provided, otherwise reuses the main --seed.

    effective_map_seed: Optional[int] = map_seed if map_seed is not None else seed
    if effective_map_seed is not None:
        map_builder = getattr(env_cfg.game, "map_builder", None)
        if isinstance(map_builder, MapGen.Config) and map_builder.seed is None:
            map_builder.seed = effective_map_seed

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
        seed=seed,
        render_mode=render,
        game_name=resolved_mission,
        save_replay=save_replay_dir,
    )


@app.command(name="replay", help="Replay a saved game using MettaScope")
def replay_cmd(
    replay_path: Path = typer.Argument(..., help="Path to the replay file"),  # noqa: B008
) -> None:
    if not replay_path.exists():
        console.print(f"[red]Error: Replay file not found: {replay_path}[/red]")
        raise typer.Exit(1)

    try:
        mettascope_path = _resolve_mettascope_script()
    except FileNotFoundError as exc:
        console.print(f"[red]Error locating MettaScope: {exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[cyan]Launching MettaScope to replay: {replay_path}[/cyan]")

    try:
        # Run nim with mettascope and replay argument
        cmd = ["nim", "r", "-d:fidgetUseCached", str(mettascope_path), f"--replay:{replay_path}"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Error running MettaScope: {exc}[/red]")
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        console.print("[red]Error: 'nim' command not found. Please ensure Nim is installed and in your PATH.[/red]")
        raise typer.Exit(1) from exc


@app.command("make-mission", help="Create a new mission configuration")
@app.command("make-game", hidden=True)
def make_mission_cmd(
    ctx: typer.Context,
    base_mission: str = typer.Option("training_facility_1", "--mission", "-m", help="Base mission to modify"),  # noqa: B008
    agents: int = typer.Option(2, "--agents", "-a", help="Number of agents"),  # noqa: B008
    width: int = typer.Option(10, "--width", "-w", help="Map width"),  # noqa: B008
    height: int = typer.Option(10, "--height", "-h", help="Map height"),  # noqa: B008
    output: Path = typer.Option(Path("mission.yaml"), "--output", "-o", help="Output file path"),  # noqa: B008
) -> None:
    try:
        # Load base mission
        _, env_cfg, _ = get_mission_name_and_config(ctx, base_mission)

        # Modify configuration
        env_cfg.game.num_agents = agents

        # Update map size if possible
        if hasattr(env_cfg.game, "map_builder") and hasattr(env_cfg.game.map_builder, "width"):
            env_cfg.game.map_builder.width = width
            env_cfg.game.map_builder.height = height

        # Save
        game.save_mission_config(env_cfg, output)
        console.print(f"[green]Mission configuration saved to {output}[/green]")

@app.command(
    name="evaluate",
    help="Evaluate one or more policies on one or more missions",
)
@app.command("eval", hidden=True)
def evaluate_cmd(
    ctx: typer.Context,
    missions: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--mission",
        "-m",
        help="Missions to evaluate (supports wildcards, e.g., --mission training_facility.*)",
    ),
    mission_set: Optional[str] = typer.Option(
        None,
        "--mission-set",
        "-S",
        help="Predefined mission set: eval_missions, integrated_evals, spanning_evals, diagnostic_evals, all",
    ),
    cogs: Optional[int] = typer.Option(None, "--cogs", "-c", help="Number of cogs (agents)"),
    variant: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--variant",
        "-v",
        help="Mission variant (can be used multiple times, e.g., --variant solar_flare --variant dark_side)",
    ),
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
    seed: int = typer.Option(42, "--seed", help="Base random seed for evaluation", min=0),
    map_seed: Optional[int] = typer.Option(
        None,
        "--map-seed",
        help="Override MapGen seed for procedural maps (defaults to --seed if not set)",
        min=0,
    ),
    format_: Optional[Literal["yaml", "json"]] = typer.Option(
        None,
        "--format",
        help="Output results in YAML or JSON format",
    ),
    save_replay_dir: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--save-replay-dir",
        help=(
            "Directory to save replays. Directory will be created if it doesn't exist. "
            "Each replay will be saved with a unique UUID-based filename."
        ),
    ),
) -> None:
    # Handle mission set expansion
    if mission_set and missions:
        console.print("[red]Error: Cannot use both --mission-set and --mission[/red]")
        raise typer.Exit(1)

    if mission_set:
        from cogames.cli.mission import load_mission_set

        try:
            mission_objs = load_mission_set(mission_set)
            missions = [m.full_name() for m in mission_objs]
            console.print(f"[cyan]Using mission set '{mission_set}' ({len(missions)} missions)[/cyan]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        # Default to 4 cogs for mission sets unless explicitly specified
        if cogs is None:
            cogs = 4

    selected_missions = get_mission_names_and_configs(ctx, missions, variants_arg=variant, cogs=cogs, steps=steps)

    # Optionally override MapGen seed so maps are reproducible across runs.
    # This uses --map-seed if provided, otherwise reuses the main --seed.
    from mettagrid.mapgen.mapgen import MapGen

    effective_map_seed: Optional[int] = map_seed if map_seed is not None else seed
    if effective_map_seed is not None:
        for _, env_cfg in selected_missions:
            map_builder = getattr(env_cfg.game, "map_builder", None)
            if isinstance(map_builder, MapGen.Config):
                map_builder.seed = effective_map_seed

    policy_specs = get_policy_specs_with_proportions(ctx, policies)

    console.print(
        f"[cyan]Preparing evaluation for {len(policy_specs)} policies across {len(selected_missions)} mission(s)[/cyan]"
    )

    evaluate_module.evaluate(
        console,
        missions=selected_missions,
        policy_specs=[spec.to_policy_spec() for spec in policy_specs],
        proportions=[spec.proportion for spec in policy_specs],
        action_timeout_ms=action_timeout_ms,
        episodes=episodes,
        seed=seed,
        output_format=format_,
        save_replay=save_replay_dir,
    )


@app.command("version", help="Show version information")
def version_cmd() -> None:
    def public_version(dist_name: str) -> str:
        return str(Version(importlib.metadata.version(dist_name)).public)

    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="right", style="bold cyan")
    table.add_column("", justify="right")

    for dist_name in ["mettagrid", "pufferlib-core", "cogames"]:
        table.add_row(dist_name, public_version(dist_name))

    console.print(table)


@app.command(name="policies", help="Show default policies and their shorthand names")
def policies_cmd() -> None:
    policy_registry = get_policy_registry()
    table = Table(show_header=False, box=None, show_lines=False, pad_edge=False)
    table.add_column("", justify="left", style="bold cyan")
    table.add_column("", justify="right")

    for policy_name, policy_path in policy_registry.items():
        table.add_row(policy_name, policy_path)
    table.add_row("custom", "path.to.your.PolicyClass")

    console.print(table)


@app.command(name="login", help="Authenticate with CoGames server")
def login_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--server",
        "-s",
        help="CoGames server URL",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Get a new token even if one already exists",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Authentication timeout in seconds",
        min=1,
    ),
) -> None:
    from urllib.parse import urlparse

    # Check if we already have a token
    from cogames.auth import BaseCLIAuthenticator

    temp_auth = BaseCLIAuthenticator(
        token_file_name="cogames.yaml",
        token_storage_key="login_tokens",
    )

    if temp_auth.has_saved_token(server) and not force:
        console.print(f"[green]Already authenticated with {urlparse(server).hostname}[/green]")
        return

    # Perform authentication
    console.print(f"[cyan]Authenticating with {server}...[/cyan]")
    if perform_login(auth_server_url=server, force=force, timeout=timeout):
        console.print("[green]Authentication successful![/green]")
    else:
        console.print("[red]Authentication failed![/red]")
        raise typer.Exit(1)


app.command(name="submissions", help="List your submissions on the leaderboard")(submissions_cmd)

app.command(
    name="leaderboard",
    help="Show leaderboard entries (public or your submissions) with per-sim scores",
)(leaderboard_cmd)


@app.command(name="submit", help="Submit a policy to CoGames competitions")
def submit_cmd(
    ctx: typer.Context,
    policy: str = typer.Option(
        ...,
        "--policy",
        "-p",
        help=f"Policy specification: {policy_arg_example}",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Policy name for the submission",
    ),
    include_files: Optional[list[str]] = typer.Option(  # noqa: B008
        None,
        "--include-files",
        "-f",
        help="Files or directories to include in submission (can be specified multiple times)",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        help="Login/authentication server URL",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        help="Submission server URL",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run validation only without submitting",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip policy validation in isolated environment",
    ),
) -> None:
    """Submit a policy to CoGames competitions.

    This command validates your policy, creates a submission package,
    and uploads it to the CoGames server.

    The policy will be tested in an isolated environment before submission
    (unless --skip-validation is used).
    """
    submit_command(
        ctx=ctx,
        policy=policy,
        name=name,
        include_files=include_files,
        login_server=login_server,
        server=server,
        dry_run=dry_run,
        skip_validation=skip_validation,
    )


@app.command(name="docs", help="Print documentation")
def docs_cmd(
    doc_name: Optional[str] = typer.Argument(None, help="Document name to print"),
) -> None:
    """Print a documentation file.

    Available documents:
      - readme: README.md - CoGames overview and documentation
      - mission: MISSION.md - Mission briefing for Machina VII Deployment
      - technical_manual: TECHNICAL_MANUAL.md - Technical manual for Cogames
      - scripted_agent: Scripted agent policy documentation
      - evals: Evaluation missions documentation
      - mapgen: Cogs vs Clips map generation documentation
    """
    # Hardcoded mapping of document names to file paths and descriptions
    package_root = Path(__file__).parent.parent.parent
    docs_map: dict[str, tuple[Path, str]] = {
        "readme": (package_root / "README.md", "CoGames overview and documentation"),
        "mission": (package_root / "MISSION.md", "Mission briefing for Machina VII Deployment"),
        "technical_manual": (package_root / "TECHNICAL_MANUAL.md", "Technical manual for Cogames"),
        "scripted_agent": (
            Path(__file__).parent / "policy" / "scripted_agent" / "README.md",
            "Scripted agent policy documentation",
        ),
        "evals": (
            Path(__file__).parent / "cogs_vs_clips" / "evals" / "README.md",
            "Evaluation missions documentation",
        ),
        "mapgen": (
            Path(__file__).parent / "cogs_vs_clips" / "cogs_vs_clips_mapgen.md",
            "Cogs vs Clips map generation documentation",
        ),
    }

    # If no argument provided, show available documents
    if doc_name is None:
        from rich.table import Table

        console.print("\n[bold cyan]Available Documents:[/bold cyan]\n")
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1))
        table.add_column("Document", style="blue", no_wrap=True)
        table.add_column("Description", style="white")

        for name, (_, description) in sorted(docs_map.items()):
            table.add_row(name, description)

        console.print(table)
        console.print("\nUsage: [bold]cogames docs <document_name>[/bold]")
        console.print("Example: [bold]cogames docs mission[/bold]")
        return

    if doc_name not in docs_map:
        available = ", ".join(sorted(docs_map.keys()))
        console.print(f"[red]Error: Unknown document '{doc_name}'[/red]")
        console.print(f"\nAvailable documents: {available}")
        raise typer.Exit(1)

    doc_path, _ = docs_map[doc_name]

    if not doc_path.exists():
        console.print(f"[red]Error: Document file not found: {doc_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"cogames: {importlib.metadata.version('cogames')}")
        console.print(f"mettagrid: {importlib.metadata.version('mettagrid')}")
        console.print(f"pufferlib: {importlib.metadata.version('pufferlib')}")
    except Exception as exc:
        console.print(f"[red]Error retrieving version info: {exc}[/red]")


if __name__ == "__main__":
    app()
