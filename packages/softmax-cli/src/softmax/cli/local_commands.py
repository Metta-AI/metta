#!/usr/bin/env -S uv run
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root
from softmax.cli.tools.local.kind import kind_app
from softmax.cli.utils import error, info

console = Console()

app = typer.Typer(
    help="Metta Local Development Commands",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = get_repo_root()


def _build_img(tag: str, dockerfile_path: Path, build_args: list[str] | None = None):
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path)]
    if build_args:
        cmd.extend(build_args)
    cmd.append(str(repo_root))
    subprocess.run(cmd, check=True)


def build_policy_evaluator_img_internal(
    tag: str = "metta-policy-evaluator-local:latest", build_args: list[str] | None = None
):
    _build_img(
        tag,
        repo_root / "devops" / "docker" / "Dockerfile.policy_evaluator",
        build_args or [],
    )


@app.command(name="build-policy-evaluator-img", context_settings={"allow_extra_args": True})
def build_policy_evaluator_img(
    ctx: typer.Context,
    tag: Annotated[str, typer.Option(help="Docker image tag")] = "metta-policy-evaluator-local:latest",
):
    build_args = ctx.args if ctx.args else []
    _build_img(
        tag,
        repo_root / "devops" / "docker" / "Dockerfile.policy_evaluator",
        build_args,
    )
    console.print(f"[green]Built policy evaluator image: {tag}[/green]")


@app.command(name="build-orchestrator-img")
def build_orchestrator_img():
    _build_img(
        "softmax-orchestrator:latest",
        repo_root / "packages" / "softmax-orchestrator" / "Dockerfile",
    )
    console.print("[green]Built orchestrator image: softmax-orchestrator:latest[/green]")


@app.command(name="stats-server", context_settings={"allow_extra_args": True})
def stats_server(ctx: typer.Context):
    cmd = [
        "uv",
        "run",
        "python",
        str(
            repo_root
            / "packages"
            / "softmax-orchestrator"
            / "src"
            / "softmax"
            / "orchestrator"
            / "server.py"
        ),
    ]
    if ctx.args:
        cmd.extend(ctx.args)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        error(f"Failed to launch Stats Server: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        info("\nStats Server shutdown")
        raise typer.Exit(0) from None


@app.command(name="observatory", context_settings={"allow_extra_args": True})
def observatory(ctx: typer.Context):
    cmd = [sys.executable, str(repo_root / "observatory" / "launch.py")]
    if ctx.args:
        cmd.extend(ctx.args)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        error(f"Failed to launch Observatory: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        info("\nObservatory shutdown")
        raise typer.Exit(0) from None


app.add_typer(kind_app, name="kind")


def main():
    app()


if __name__ == "__main__":
    main()
