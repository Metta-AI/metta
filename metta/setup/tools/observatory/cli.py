#!/usr/bin/env -S uv run
import functools
import os
import subprocess
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


LOCAL_DB_URI = "postgres://postgres:password@127.0.0.1:5432/metta"
LOCAL_BACKEND_URL = "http://127.0.0.1:8000"
LOCAL_BACKEND_URL_FROM_K8S = "http://host.docker.internal:8000"
LOCAL_MACHINE_TOKEN = "local-dev-token"


HELP_TEXT = f"""
Observatory local development.

[bold]Prerequisites (one-time OrbStack setup):[/bold]
  1. Install OrbStack: https://orbstack.dev/download
  2. Enable Kubernetes in OrbStack:
     orb config set k8s.enable true
  3. Restart OrbStack (or run: orbctl stop && orbctl start)
  4. Verify: kubectl config get-contexts | grep orbstack

[bold]Setup:[/bold]
  metta observatory postgres up -d   # Backgrounded postgres for api server
  metta observatory server           # API server
  metta observatory frontend         # Observatory frontend

[bold]Additional setup for jobs:[/bold]
  metta observatory local-k8s setup  # Build image and create namespaces
  metta observatory watcher          # Watches k8s jobs and updates status via api server

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

repo_root = get_repo_root()


@app.command(
    name="postgres",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    help="Manage postgres. Usage: metta observatory postgres [up|down|logs]",
)
@handle_errors
def postgres(ctx: typer.Context):
    cmd = ["docker", "compose", "-f", str(repo_root / "app_backend" / "docker-compose.dev.yml")]
    if ctx.args:
        cmd.extend(ctx.args)
    else:
        cmd.append("up")
    subprocess.run(cmd, check=True)


@app.command(name="server", help="Run the backend server on host")
@handle_errors
def server():
    env = os.environ.copy()
    env["STATS_DB_URI"] = LOCAL_DB_URI
    env["DEBUG_USER_EMAIL"] = "localdev@example.com"
    env["RUN_MIGRATIONS"] = "true"
    env["EPISODE_RUNNER_IMAGE"] = LOCAL_METTA_POLICY_EVAL_IMG_NAME
    env["STATS_SERVER_URI"] = LOCAL_BACKEND_URL_FROM_K8S
    env["MACHINE_TOKEN"] = LOCAL_MACHINE_TOKEN

    info("Starting backend server...")
    info(f"  DB: {LOCAL_DB_URI}")
    info(f"  URL: {LOCAL_BACKEND_URL}")
    info(f"  Episode runner image: {LOCAL_METTA_POLICY_EVAL_IMG_NAME}")

    subprocess.run(
        ["uv", "run", "python", str(repo_root / "app_backend/src/metta/app_backend/server.py")],
        env=env,
        check=True,
    )


@app.command(name="watcher", help="Run the job watcher on host")
@handle_errors
def watcher():
    env = os.environ.copy()
    env["STATS_SERVER_URI"] = LOCAL_BACKEND_URL
    env["MACHINE_TOKEN"] = LOCAL_MACHINE_TOKEN

    info("Starting watcher...")
    info(f"  Stats server: {LOCAL_BACKEND_URL}")

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
    env = os.environ.copy()

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

    subprocess.run(["pnpm", "run", "dev"], env=env, check=True, cwd=get_repo_root() / "observatory")


app.add_typer(local_k8s_app, name="local-k8s")


def main():
    app()


if __name__ == "__main__":
    main()
