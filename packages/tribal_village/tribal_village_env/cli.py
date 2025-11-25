from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from tribal_village_env.build import ensure_nim_library_current
from tribal_village_env.environment import TribalVillageEnv

app = typer.Typer(
    help="CLI for playing Tribal Village",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_gui() -> None:
    project_root = _project_root()
    cmd = ["nim", "r", "-d:release", "tribal_village.nim"]

    try:
        console.print("[cyan]Launching Tribal Village GUI via Nim...[/cyan]")
        subprocess.run(cmd, cwd=project_root, check=True)
    except FileNotFoundError:
        console.print("[red]Error: 'nim' command not found. Please install Nim and ensure it is on your PATH.[/red]")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Nim GUI run failed with exit code {exc.returncode}.[/red]")
        raise typer.Exit(exc.returncode)


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


@app.command("play", help="Play Tribal Village using the Nim GUI or ANSI renderer")
def play(
    render: str = typer.Option(
        "gui",
        "--render",
        "-r",
        help="Render mode: gui (default) or ansi (text-only)",
    ),
    steps: int = typer.Option(128, "--steps", "-s", help="Steps to run when using ANSI render", min=1),
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Override max steps in the environment (ANSI mode only)",
        min=1,
    ),
    random_actions: bool = typer.Option(
        True,
        "--random-actions/--no-random-actions",
        help="Use random actions in ANSI mode (otherwise no-op)",
    ),
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


if __name__ == "__main__":
    app()


def main() -> None:
    # If invoked without subcommand, default to play
    app(prog_name="tribal-village")
