from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from tribal_village_env.build import ensure_nim_library_current
from tribal_village_env.environment import TribalVillageEnv

# Optional CoGames training integration
try:
    from tribal_village_env.cogames.cli import attach_train_command
except ImportError:  # CoGames not installed; CLI should still work for play mode
    attach_train_command = None  # type: ignore[assignment]

app = typer.Typer(
    help="CLI for playing Tribal Village",
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()

# Attempt to register the PufferLib trainer if CoGames is installed locally.
if attach_train_command is not None:
    attach_train_command(app, command_name="train", require_cogames=False, console_fallback=console)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_gui() -> None:
    project_root = _project_root()
    cmd = ["nim", "r", "-d:release", "tribal_village.nim"]

    console.print("[cyan]Launching Tribal Village GUI via Nim...[/cyan]")
    subprocess.run(cmd, cwd=project_root, check=True)


def _run_ansi(steps: int, max_steps: Optional[int], random_actions: bool) -> None:
    config: dict[str, object] = {"render_mode": "ansi"}
    if max_steps is not None:
        config["max_steps"] = max_steps

    env = TribalVillageEnv(config=config)

    def _make_actions() -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id in range(env.num_agents):
            key = f"agent_{agent_id}"
            if random_actions:
                actions[key] = int(env.single_action_space.sample())
            else:
                actions[key] = 0
        return actions

    try:
        env.reset()
        console.print(env.render())

        for step in range(steps):
            actions = _make_actions()
            _, _, terminated, truncated, _ = env.step(actions)
            console.print(env.render())

            if all(terminated.values()) or all(truncated.values()):
                console.print(f"[yellow]Episode ended at step {step + 1}[/yellow]")
                break
    finally:
        env.close()


def _options():
    return {
        "render": typer.Option("gui", "--render", "-r", help="Render mode: gui (default) or ansi (text-only)"),
        "steps": typer.Option(128, "--steps", "-s", help="Steps to run when using ANSI render", min=1),
        "max_steps": typer.Option(None, "--max-steps", help="Override max steps in ANSI mode", min=1),
        "random_actions": typer.Option(
            True, "--random-actions/--no-random-actions", help="Use random actions in ANSI mode (otherwise no-op)"
        ),
    }


@app.command("play", help="Play Tribal Village using the Nim GUI or ANSI renderer")
def play(
    render: str = _options()["render"],
    steps: int = _options()["steps"],
    max_steps: Optional[int] = _options()["max_steps"],
    random_actions: bool = _options()["random_actions"],
) -> None:
    ensure_nim_library_current()

    render_mode = render.lower()
    if render_mode not in {"gui", "ansi"}:
        console.print("[red]Invalid render mode. Use 'gui' or 'ansi'.[/red]")
        raise typer.Exit(1)

    if render_mode == "gui":
        _run_gui()
    else:
        _run_ansi(steps=steps, max_steps=max_steps, random_actions=random_actions)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    render: str = _options()["render"],
    steps: int = _options()["steps"],
    max_steps: Optional[int] = _options()["max_steps"],
    random_actions: bool = _options()["random_actions"],
) -> None:
    """Default to play when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(play, render=render, steps=steps, max_steps=max_steps, random_actions=random_actions)


if __name__ == "__main__":
    app()
