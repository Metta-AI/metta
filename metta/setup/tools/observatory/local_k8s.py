import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

repo_root = get_repo_root()
console = Console()

LOCAL_K8S_CONTEXT = "orbstack"
LOCAL_K8S_JOBS_NAMESPACE = "jobs"
LOCAL_IMAGE_NAME = "metta-policy-evaluator-local:latest"


def _check_orbstack_k8s() -> bool:
    result = subprocess.run(
        ["kubectl", "config", "get-contexts", "-o", "name"],
        capture_output=True,
        text=True,
    )
    return LOCAL_K8S_CONTEXT in result.stdout.split()


def _get_local_image_id(image_name: str) -> str | None:
    result = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _build_image(force_build: bool = True):
    image_exists = subprocess.run(["docker", "image", "inspect", LOCAL_IMAGE_NAME], capture_output=True).returncode == 0
    old_image_id = _get_local_image_id(LOCAL_IMAGE_NAME) if image_exists else None

    if force_build or not image_exists:
        info(f"Building {LOCAL_IMAGE_NAME}...")
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                LOCAL_IMAGE_NAME,
                "-f",
                "devops/docker/Dockerfile.policy_evaluator",
                "--platform",
                "linux/amd64",
                ".",
            ],
            check=True,
            cwd=repo_root,
        )
        if old_image_id:
            new_image_id = _get_local_image_id(LOCAL_IMAGE_NAME)
            if new_image_id != old_image_id:
                info(f"Removing old image {old_image_id[:12]}...")
                subprocess.run(["docker", "rmi", old_image_id], capture_output=True)


class LocalK8s:
    jobs_namespace: str = LOCAL_K8S_JOBS_NAMESPACE
    context: str = LOCAL_K8S_CONTEXT

    def _check_orbstack(self) -> None:
        if not _check_orbstack_k8s():
            error("OrbStack Kubernetes is not available.")
            error("Please enable it: orb config set k8s.enable true")
            error("Then restart OrbStack.")
            sys.exit(1)

    def _use_context(self) -> None:
        subprocess.run(["kubectl", "config", "use-context", LOCAL_K8S_CONTEXT], check=True)

    def _ensure_namespace(self) -> None:
        result = subprocess.run(
            ["kubectl", "create", "namespace", self.jobs_namespace],
            capture_output=True,
        )
        if result.returncode != 0 and b"already exists" not in result.stderr:
            error(f"Failed to create namespace {self.jobs_namespace}")
            raise typer.Exit(1)

    def setup(self) -> None:
        self._check_orbstack()
        self._use_context()

        info("Building job runner image...")
        _build_image(force_build=True)
        success("Image built (OrbStack shares Docker images with k8s automatically)")

        self._ensure_namespace()
        success(f"{self.jobs_namespace} namespace ready")

    def clean(self) -> None:
        info("Cleaning up...")
        self._use_context()
        subprocess.run(
            ["kubectl", "delete", "namespace", self.jobs_namespace, "--ignore-not-found=true"],
            check=True,
        )
        success("Namespace deleted")

    def get_pods(self) -> None:
        self._use_context()
        subprocess.run(["kubectl", "get", "pods", "-n", self.jobs_namespace], check=True)

    def logs(self, pod_name: str | None = None) -> None:
        self._use_context()
        if pod_name:
            subprocess.run(["kubectl", "logs", pod_name, "-n", self.jobs_namespace, "--follow"], check=True)
        else:
            subprocess.run(
                ["kubectl", "logs", "-n", self.jobs_namespace, "-l", "app=episode-runner", "--follow"],
                check=True,
            )


local_k8s_app = typer.Typer(help="Manage local Kubernetes (OrbStack)", rich_markup_mode="rich", no_args_is_help=True)

local_k8s = LocalK8s()


@local_k8s_app.command(name="setup")
def cmd_setup():
    """Build image and create jobs namespace."""
    local_k8s.setup()
    console.print("[green]Local k8s setup complete[/green]")


@local_k8s_app.command(name="clean")
def cmd_clean():
    """Delete jobs namespace."""
    local_k8s.clean()
    console.print("[green]Cleanup complete[/green]")


@local_k8s_app.command(name="get-pods")
def cmd_get_pods():
    """List job pods."""
    local_k8s.get_pods()


@local_k8s_app.command(name="logs")
def cmd_logs(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    """Follow logs for job pods."""
    local_k8s.logs(pod_name)


@local_k8s_app.command(name="build-image")
def cmd_build_image():
    """Rebuild the job runner Docker image."""
    _build_image(force_build=True)
    console.print("[green]Image built[/green]")


def main():
    local_k8s_app()


if __name__ == "__main__":
    main()
