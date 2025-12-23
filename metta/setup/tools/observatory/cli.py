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

HELP_TEXT = """
Observatory local development.

[bold]Setup:[/bold]
  metta observatory kind build    # One-time: create Kind cluster
  metta observatory postgres up -d
  metta observatory server        # Terminal 2
  metta observatory watcher       # Terminal 3

[bold]Submit test jobs:[/bold]
  uv run python app_backend/scripts/submit_test_jobs.py

[bold]Monitor:[/bold]
  kubectl get jobs -n jobs -w
  kubectl logs -n jobs -l app=episode-runner -f

[bold]Teardown:[/bold]
  metta observatory postgres down
  metta observatory kind clean
"""

app = typer.Typer(
    help=HELP_TEXT,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

repo_root = get_repo_root()

LOCAL_DB_URI = "postgres://postgres:password@127.0.0.1:5432/metta"
LOCAL_BACKEND_URL = "http://127.0.0.1:8000"
LOCAL_BACKEND_URL_FROM_KIND = "http://host.docker.internal:8000"
LOCAL_MACHINE_TOKEN = "local-dev-token"


@app.command(
    name="postgres",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    help="Manage postgres. Usage: metta observatory postgres [up|down|logs]",
)
def postgres(ctx: typer.Context):
    cmd = ["docker", "compose", "-f", str(repo_root / "app_backend" / "docker-compose.dev.yml")]
    if ctx.args:
        cmd.extend(ctx.args)
    else:
        cmd.append("up")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        error(f"Failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        raise typer.Exit(0) from None


@app.command(name="server", help="Run the backend server on host")
def server():
    env = os.environ.copy()
    env["STATS_DB_URI"] = LOCAL_DB_URI
    env["DEBUG_USER_EMAIL"] = "localdev@example.com"
    env["RUN_MIGRATIONS"] = "true"
    env["EPISODE_RUNNER_IMAGE"] = "metta-policy-evaluator-local:latest"
    env["BACKEND_URL"] = LOCAL_BACKEND_URL_FROM_KIND
    env["MACHINE_TOKEN"] = LOCAL_MACHINE_TOKEN

    info("Starting backend server...")
    info(f"  DB: {LOCAL_DB_URI}")
    info(f"  URL: {LOCAL_BACKEND_URL}")

    try:
        subprocess.run(
            ["uv", "run", "python", str(repo_root / "app_backend/src/metta/app_backend/server.py")],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error(f"Server failed: {e}")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        raise typer.Exit(0) from None


@app.command(name="watcher", help="Run the job watcher on host")
def watcher():
    env = os.environ.copy()
    env["BACKEND_URL"] = LOCAL_BACKEND_URL
    env["MACHINE_TOKEN"] = LOCAL_MACHINE_TOKEN

    info("Starting watcher...")
    info(f"  Backend: {LOCAL_BACKEND_URL}")

    try:
        subprocess.run(
            ["uv", "run", "python", "-m", "metta.app_backend.job_runner.watcher"],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error(f"Watcher failed: {e}")
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
