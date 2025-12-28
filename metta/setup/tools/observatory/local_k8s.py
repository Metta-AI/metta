import subprocess
import sys
from typing import Annotated

import typer

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

repo_root = get_repo_root()

CONTEXT = "orbstack"
NAMESPACE = "jobs"
IMAGE = "metta-policy-evaluator-local:latest"


def _check_orbstack() -> None:
    result = subprocess.run(
        ["kubectl", "config", "get-contexts", "-o", "name"],
        capture_output=True,
        text=True,
    )
    if CONTEXT not in result.stdout.split():
        error("OrbStack Kubernetes is not available.")
        error("Please enable it: orb config set k8s.enable true")
        error("Then restart OrbStack.")
        sys.exit(1)


def _kubectl(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["kubectl", "--context", CONTEXT, *args], check=True)


def _build_image() -> None:
    old_id = subprocess.run(["docker", "images", "-q", IMAGE], capture_output=True, text=True).stdout.strip()

    info(f"Building {IMAGE}...")
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            IMAGE,
            "-f",
            "devops/docker/Dockerfile.policy_evaluator",
            "--platform",
            "linux/amd64",
            ".",
        ],
        check=True,
        cwd=repo_root,
    )

    if old_id:
        new_id = subprocess.run(["docker", "images", "-q", IMAGE], capture_output=True, text=True).stdout.strip()
        if new_id != old_id:
            info(f"Removing old image {old_id[:12]}...")
            subprocess.run(["docker", "rmi", old_id], capture_output=True)


local_k8s_app = typer.Typer(help="Manage local Kubernetes (OrbStack)", rich_markup_mode="rich", no_args_is_help=True)


@local_k8s_app.command(name="setup")
def cmd_setup():
    """Build image and create jobs namespace."""
    _check_orbstack()

    _build_image()
    success("Image built (OrbStack shares Docker images with k8s automatically)")

    result = subprocess.run(
        ["kubectl", "--context", CONTEXT, "create", "namespace", NAMESPACE],
        capture_output=True,
    )
    if result.returncode != 0 and b"already exists" not in result.stderr:
        error(f"Failed to create namespace {NAMESPACE}")
        raise typer.Exit(1)
    success(f"{NAMESPACE} namespace ready")


@local_k8s_app.command(name="clean")
def cmd_clean():
    """Delete jobs namespace."""
    _kubectl("delete", "namespace", NAMESPACE, "--ignore-not-found=true")
    success("Namespace deleted")


@local_k8s_app.command(name="get-pods")
def cmd_get_pods():
    """List job pods."""
    _kubectl("get", "pods", "-n", NAMESPACE)


@local_k8s_app.command(name="logs")
def cmd_logs(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    """Follow logs for job pods."""
    if pod_name:
        _kubectl("logs", pod_name, "-n", NAMESPACE, "--follow")
    else:
        _kubectl("logs", "-n", NAMESPACE, "-l", "app=episode-runner", "--follow")


@local_k8s_app.command(name="build-image")
def cmd_build_image():
    """Rebuild the job runner Docker image."""
    _build_image()
    success("Image built")


def main():
    local_k8s_app()


if __name__ == "__main__":
    main()
