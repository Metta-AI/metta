#!/usr/bin/env -S uv run
import os
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console

from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.constants import DEV_STATS_SERVER_URI, PROD_STATS_SERVER_URI
from metta.common.util.fs import get_repo_root
from metta.setup.tools.observatory.kind import kind_app
from metta.setup.utils import error, info

console = Console()

app = typer.Typer(
    help="Metta Local Development Commands",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = get_repo_root()


@app.command(
    name="backend",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    help=(
        "Manage local instance of Observatory Backend. "
        "Usage: metta observatory backend [build|up|down|restart|logs|enter]"
    ),
)
def observatory_backend(ctx: typer.Context):
    cmd = ["docker", "compose", "-f", str(get_repo_root() / "app_backend" / "docker-compose.dev.yml")]
    if ctx.args:
        cmd.extend(ctx.args)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        error(f"Failed to launch Stats Server: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        raise typer.Exit(0) from None


@app.command(name="frontend")
def frontend(
    backend: Annotated[str, typer.Option("--backend", "-b", help="Select backend: local or prod")] = "local",
):
    env = os.environ.copy()

    if backend == "local":
        env["VITE_API_URL"] = DEV_STATS_SERVER_URI
        info("Connecting to local backend at localhost:8000")
        info("Make sure backend is running: docker compose -f app_backend/docker-compose.dev.yml up")
    else:
        env["VITE_API_URL"] = PROD_STATS_SERVER_URI
        if token := get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token
        info("Connecting to prod backend")

    info("Starting Observatory frontend")
    info(f"API URL: {env.get('VITE_API_URL')}")

    try:
        subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=get_repo_root() / "observatory")
    except subprocess.CalledProcessError as e:
        error(f'Error running "pnpm run dev": {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nObservatory shutdown")
        sys.exit(0)


app.add_typer(kind_app, name="kind")


def main():
    app()


if __name__ == "__main__":
    main()
