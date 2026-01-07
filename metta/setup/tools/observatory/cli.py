#!/usr/bin/env -S uv run
import functools
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.util.fs import get_repo_root
from metta.setup.tools.observatory.local_k8s import local_k8s_app
from metta.setup.tools.observatory.utils import LOCAL_METTA_POLICY_EVAL_IMG_NAME
from metta.setup.utils import error, info

console = Console()
repo_root = get_repo_root()

# Local dev configuration
LOCALHOST = "127.0.0.1"
POSTGRES_PORT = 5432
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "password"
POSTGRES_DB = "metta"
SERVER_PORT = 8000
PROCESS_COMPOSE_PORT = 8090

LOCAL_DB_URI = f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{LOCALHOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
LOCAL_BACKEND_URL = f"http://{LOCALHOST}:{SERVER_PORT}"
LOCAL_BACKEND_URL_FROM_K8S = f"http://host.docker.internal:{SERVER_PORT}"
LOCAL_MACHINE_TOKEN = "local-dev-token"
LOCAL_K8S_CONTEXT = "orbstack"
LOCAL_AWS_PROFILE = "softmax"
DEBUG_USER_EMAIL = "localdev@example.com"


def handle_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            error(f"Failed: {e}")
            raise typer.Exit(1) from e
        except KeyboardInterrupt:
            raise typer.Exit(0) from None

    return wrapper


HELP_TEXT = f"""
Observatory local development.

[bold]Prerequisites (one-time OrbStack setup):[/bold]
  OrbStack is installed via 'metta install' (profile=softmax).
  Enable Kubernetes: orb config set k8s.enable true
  Then restart OrbStack (or: orbctl stop && orbctl start)

[bold]Quick start:[/bold]
  metta observatory local-k8s setup  # One-time: build image and create jobs namespace
  metta observatory up               # Start all services (postgres, server, frontend, watcher, tournament)

[bold]Start specific services:[/bold]
  metta observatory up server frontend  # Only server and frontend

[bold]Individual services:[/bold]
  metta observatory postgres up -d   # Backgrounded postgres for api server
  metta observatory server           # API server
  metta observatory frontend         # Observatory frontend
  metta observatory watcher          # Watches k8s jobs and updates status via api server
  metta observatory tournament       # Tournament commissioner (creates matches, updates scores)

[bold]Upload policy:[/bold]
  uv run cogames submit -p class=scripted_baseline --server {LOCAL_BACKEND_URL} --skip-validation -n <your-policy-name>

[bold]Submit test jobs:[/bold]
  uv run python app_backend/scripts/submit_test_jobs.py --policy-uri metta://policy/<your-policy-name>

[bold]Monitor:[/bold]
  kubectl --context orbstack get pods -n jobs -w
  kubectl --context orbstack logs -n jobs -l app=episode-runner -f

[bold]Teardown:[/bold]
  metta observatory postgres down
  metta observatory local-k8s clean

[bold]Rebuild job runner image:[/bold]
  metta observatory local-k8s build-image
"""

app = typer.Typer(
    help=HELP_TEXT,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env


def _local_dev_env() -> dict[str, str]:
    env = _base_env()
    env["STATS_DB_URI"] = LOCAL_DB_URI
    env["DEBUG_USER_EMAIL"] = DEBUG_USER_EMAIL
    env["RUN_MIGRATIONS"] = "true"
    env["EPISODE_RUNNER_IMAGE"] = LOCAL_METTA_POLICY_EVAL_IMG_NAME
    env["MACHINE_TOKEN"] = LOCAL_MACHINE_TOKEN
    env["LOCAL_DEV"] = "true"
    env["LOCAL_DEV_K8S_CONTEXT"] = LOCAL_K8S_CONTEXT
    env["LOCAL_DEV_AWS_PROFILE"] = LOCAL_AWS_PROFILE

    aws_path = os.path.expanduser("~/.aws")
    source_mounts = [
        f"{aws_path}:/root/.aws",
        f"{repo_root}/metta:/workspace/metta/metta",
        f"{repo_root}/app_backend:/workspace/metta/app_backend",
        f"{repo_root}/common:/workspace/metta/common",
    ]
    env["LOCAL_DEV_MOUNTS"] = ",".join(source_mounts)
    return env


def _postgres_env() -> dict[str, str]:
    env = _base_env()
    env["POSTGRES_HOST"] = LOCALHOST
    env["POSTGRES_PORT"] = str(POSTGRES_PORT)
    env["POSTGRES_USER"] = POSTGRES_USER
    env["POSTGRES_PASSWORD"] = POSTGRES_PASSWORD
    env["POSTGRES_DB"] = POSTGRES_DB
    return env


@app.command(name="up", help="Start all observatory services (postgres, server, frontend, watcher)")
@handle_errors
def up(
    services: Annotated[list[str] | None, typer.Argument(help="Services to start (default: all)")] = None,
    tui: Annotated[bool, typer.Option("-t", "--tui", help="Enable TUI mode")] = False,
):
    compose_file = Path(__file__).parent / "process-compose.yaml"
    env = _postgres_env()
    env["SERVER_HOST"] = LOCALHOST
    env["SERVER_PORT"] = str(SERVER_PORT)
    cmd = ["process-compose", "-f", str(compose_file), "-p", str(PROCESS_COMPOSE_PORT)]
    if not tui:
        cmd.append("-t=false")
    if services:
        cmd.extend(services)
    info("Starting observatory services...")
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


@app.command(
    name="postgres",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    help="Manage postgres. Usage: metta observatory postgres [up|down|logs]",
)
@handle_errors
def postgres(ctx: typer.Context):
    cmd = ["docker", "compose", "-f", str(repo_root / "app_backend" / "docker-compose.dev.yml")]
    args = ctx.args if ctx.args else ["up"]
    if "up" in args and "-d" in args and "--wait" not in args:
        args = args + ["--wait"]
    cmd.extend(args)
    subprocess.run(cmd, env=_postgres_env(), check=True)


@app.command(name="server", help="Run the backend server on host")
@handle_errors
def server():
    env = _local_dev_env()
    env["HOST"] = "0.0.0.0"
    env["PORT"] = str(SERVER_PORT)
    env["STATS_SERVER_URI"] = LOCAL_BACKEND_URL_FROM_K8S

    info("Starting backend server...")
    subprocess.run(
        ["uv", "run", "python", str(repo_root / "app_backend/src/metta/app_backend/server.py")],
        env=env,
        check=True,
    )


@app.command(name="watcher", help="Run the job watcher on host")
@handle_errors
def watcher():
    env = _local_dev_env()
    env["STATS_SERVER_URI"] = LOCAL_BACKEND_URL

    info("Starting watcher...")
    subprocess.run(
        ["uv", "run", "python", "-m", "metta.app_backend.job_runner.watcher"],
        env=env,
        check=True,
    )


@app.command(name="frontend")
@handle_errors
def frontend(
    backend: Annotated[str, typer.Option("--backend", "-b", help="Select backend: local or prod")] = "local",
):
    env = _base_env()

    if backend == "local":
        env["VITE_API_URL"] = LOCAL_BACKEND_URL
        info(f"Connecting to local backend at {LOCAL_BACKEND_URL}")
    else:
        env["VITE_API_URL"] = PROD_STATS_SERVER_URI
        if token := get_machine_token(env["VITE_API_URL"]):
            env["VITE_AUTH_TOKEN"] = token
        info("Connecting to prod backend")

    info("Starting Observatory frontend")
    info(f"API URL: {env.get('VITE_API_URL')}")

    subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=repo_root / "observatory")


@app.command(name="tournament", help="Run the tournament commissioner")
@handle_errors
def tournament():
    env = _local_dev_env()
    subprocess.run(
        ["uv", "run", "python", str(repo_root / "app_backend/src/metta/app_backend/tournament/cli.py")],
        env=env,
        check=True,
    )


app.add_typer(local_k8s_app, name="local-k8s")


def main():
    app()


if __name__ == "__main__":
    main()
