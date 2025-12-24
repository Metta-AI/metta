import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from metta.common.util.constants import DEV_STATS_SERVER_URI
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

repo_root = get_repo_root()
console = Console()

LOCAL_K8S_CONTEXT = "orbstack"
LOCAL_K8S_NAMESPACE = "orchestrator"
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
    namespace: str = LOCAL_K8S_NAMESPACE
    jobs_namespace: str = LOCAL_K8S_JOBS_NAMESPACE
    helm_release_name: str = "orchestrator"
    helm_chart_path: Path = repo_root / "devops/charts/orchestrator"
    environment_values_file: Path = repo_root / "devops/charts/orchestrator/environments/local.yaml"
    context: str = LOCAL_K8S_CONTEXT

    def _check_namespace_exists(self, namespace: str) -> bool:
        result = subprocess.run(
            ["kubectl", "get", "namespace", namespace], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0

    def _use_context(self) -> None:
        subprocess.run(["kubectl", "config", "use-context", LOCAL_K8S_CONTEXT], check=True)

    def _create_namespace(self, namespace: str) -> None:
        result = subprocess.run(["kubectl", "create", "namespace", namespace], capture_output=True)
        if result.returncode != 0 and b"already exists" not in result.stderr:
            error(f"Failed to create namespace {namespace}")
            raise Exception(f"Failed to create namespace {namespace}")

    def _create_secret(self, name: str, value: str) -> None:
        subprocess.run(
            ["kubectl", "delete", "secret", name, "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )
        subprocess.run(
            [
                "kubectl",
                "create",
                "secret",
                "generic",
                name,
                f"--from-literal={value}",
                "-n",
                self.namespace,
            ],
            check=True,
        )

    def _get_wandb_api_key(self) -> str | None:
        import wandb

        return wandb.Api().api_key

    def _maybe_load_secrets(self) -> None:
        wandb_api_key = self._get_wandb_api_key()
        if not wandb_api_key:
            error("No WANDB API key found. Please run 'wandb login' and try again.")
            sys.exit(1)

        from metta.app_backend.clients.base_client import get_machine_token

        machine_token = get_machine_token(DEV_STATS_SERVER_URI)

        info("Creating secrets...")
        self._create_secret("wandb-api-secret", f"api-key={wandb_api_key}")
        self._create_secret("machine-token-secret", f"token={machine_token}")

    def setup(self) -> None:
        if not _check_orbstack_k8s():
            error("OrbStack Kubernetes is not available.")
            error("Please enable it: orb config set k8s.enable true")
            error("Then restart OrbStack.")
            sys.exit(1)

        self._use_context()

        info("Building job runner image...")
        _build_image(force_build=True)
        success("Image built (OrbStack shares Docker images with k8s automatically)")

        if not self._check_namespace_exists(self.namespace):
            self._create_namespace(self.namespace)
        success(f"{self.namespace} namespace ready")

        if not self._check_namespace_exists(self.jobs_namespace):
            self._create_namespace(self.jobs_namespace)
        success(f"{self.jobs_namespace} namespace ready")

    def up(self) -> None:
        self._use_context()
        info("Creating namespace if needed...")

        self._maybe_load_secrets()

        info("Updating Helm dependencies...")
        subprocess.run(["helm", "dependency", "update", str(self.helm_chart_path)], check=True)
        success("Helm dependencies updated")

        result = subprocess.run(["helm", "list", "-n", self.namespace, "-q"], capture_output=True, text=True)
        cmd = "upgrade" if self.helm_release_name in result.stdout else "install"
        info(f"Running {cmd} for {self.helm_release_name}...")

        helm_cmd = [
            "helm",
            cmd,
            self.helm_release_name,
            str(self.helm_chart_path),
            "-n",
            self.namespace,
            "-f",
            str(self.environment_values_file),
        ]

        subprocess.run(helm_cmd, check=True)

        info("Orchestrator deployed via Helm")
        info("To view pods: metta observatory local-k8s get-pods")
        info("To view logs: metta observatory local-k8s logs <pod-name>")
        info("To stop: metta observatory local-k8s down")

    def down(self) -> None:
        info("Stopping...")
        self._use_context()

        subprocess.run(
            ["helm", "uninstall", self.helm_release_name, "-n", self.namespace, "--ignore-not-found"], check=True
        )

        subprocess.run(
            ["kubectl", "delete", "pods", "-l", "app=eval-worker", "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )

        success("Stopped")

    def clean(self) -> None:
        info("Cleaning up namespaces...")
        self._use_context()
        subprocess.run(["kubectl", "delete", "namespace", self.namespace, "--ignore-not-found=true"], check=True)
        subprocess.run(["kubectl", "delete", "namespace", self.jobs_namespace, "--ignore-not-found=true"], check=True)
        success("Namespaces deleted")

    def get_pods(self) -> None:
        self._use_context()
        subprocess.run(["kubectl", "get", "pods", "-n", self.namespace], check=True)
        subprocess.run(["kubectl", "get", "pods", "-n", self.jobs_namespace], check=True)

    def logs(self, pod_name: str | None = None) -> None:
        self._use_context()

        if pod_name:
            subprocess.run(["kubectl", "logs", pod_name, "-n", self.namespace, "--follow"], check=True)
        else:
            subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, "-l", "app.kubernetes.io/name=orchestrator", "--follow"],
                check=True,
            )

    def enter(self, pod_name: str | None = None) -> None:
        self._use_context()

        if not pod_name:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-l",
                    "app.kubernetes.io/name=orchestrator",
                    "-o",
                    "jsonpath={.items[0].metadata.name}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            pod_name = result.stdout.strip()

        subprocess.run(["kubectl", "exec", "-it", pod_name, "-n", self.namespace, "--", "/bin/bash"], check=True)


local_k8s_app = typer.Typer(help="Manage local Kubernetes (OrbStack)", rich_markup_mode="rich", no_args_is_help=True)

local_k8s = LocalK8s()


@local_k8s_app.command(name="setup")
def cmd_setup():
    """Set up local k8s: build image and create namespaces."""
    local_k8s.setup()
    console.print("[green]Local k8s setup complete[/green]")


@local_k8s_app.command(name="up")
def cmd_up():
    """Deploy orchestrator to local k8s."""
    local_k8s.up()
    console.print("[green]Orchestrator started[/green]")


@local_k8s_app.command(name="down")
def cmd_down():
    """Stop orchestrator."""
    local_k8s.down()
    console.print("[green]Orchestrator stopped[/green]")


@local_k8s_app.command(name="clean")
def cmd_clean():
    """Delete namespaces and clean up."""
    local_k8s.clean()
    console.print("[green]Cleanup complete[/green]")


@local_k8s_app.command(name="get-pods")
def cmd_get_pods():
    """List pods in local k8s."""
    local_k8s.get_pods()


@local_k8s_app.command(name="logs")
def cmd_logs(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    """Follow logs for orchestrator or specific pod."""
    local_k8s.logs(pod_name)


@local_k8s_app.command(name="enter")
def cmd_enter(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    """Enter a pod with an interactive shell."""
    local_k8s.enter(pod_name)


@local_k8s_app.command(name="build-image")
def cmd_build_image():
    """Build the job runner Docker image."""
    _build_image(force_build=True)
    console.print("[green]Image built[/green]")


def main():
    local_k8s_app()


if __name__ == "__main__":
    main()
