import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from metta.common.util.constants import DEV_STATS_SERVER_URI
from metta.common.util.fs import get_repo_root
from metta.setup.tools.observatory.utils import build_image, load_image_into_kind
from metta.setup.utils import error, info, success

repo_root = get_repo_root()
console = Console()


class Kind:
    cluster_name: str
    namespace: str
    helm_release_name: str
    helm_chart_path: Path
    environment_values_file: Path | None
    context: str

    def _check_namespace_exists(self, namespace: str) -> bool:
        result = subprocess.run(
            ["kubectl", "get", "namespace", namespace], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0

    def _use_appropriate_context(self) -> None:
        subprocess.run(["kubectl", "config", "use-context", self.context], check=True)

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

    def build(self) -> None:
        pass

    def _maybe_load_secrets(self) -> None:
        pass

    def up(self) -> None:
        """Start orchestrator in Kind cluster using Helm."""
        self._use_appropriate_context()
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
        ]

        if self.environment_values_file:
            helm_cmd.extend(["-f", str(self.environment_values_file)])

        subprocess.run(helm_cmd, check=True)

        info("Orchestrator deployed via Helm")

        info("To view pods: metta observatory kind get-pods")
        info("To view logs: metta observatory kind logs <pod-name>")
        info("To stop: metta observatory kind down")

    def down(self) -> None:
        """Stop orchestrator and worker pods."""
        info("Stopping...")
        self._use_appropriate_context()

        subprocess.run(
            ["helm", "uninstall", self.helm_release_name, "-n", self.namespace, "--ignore-not-found"], check=True
        )

        subprocess.run(
            ["kubectl", "delete", "pods", "-l", "app=eval-worker", "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )

        success("Stopped (cluster preserved for faster restarts)")

    def clean(self) -> None:
        """Delete the Kind cluster."""
        info("Deleting cluster...")
        self._use_appropriate_context()
        subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
        success("Cluster deleted")

    def get_pods(self) -> None:
        """Get list of pods in the cluster."""
        self._use_appropriate_context()
        subprocess.run(["kubectl", "get", "pods", "-n", self.namespace], check=True)

    def logs(self, pod_name: str | None = None) -> None:
        """Follow logs for orchestrator or specific pod."""
        self._use_appropriate_context()

        if pod_name:
            subprocess.run(["kubectl", "logs", pod_name, "-n", self.namespace, "--follow"], check=True)
        else:
            subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, "-l", "app.kubernetes.io/name=orchestrator", "--follow"],
                check=True,
            )

    def enter(self, pod_name: str | None = None) -> None:
        """Enter orchestrator or specific pod with an interactive shell."""
        self._use_appropriate_context()

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

    def _get_wandb_api_key(self) -> str | None:
        import wandb

        return wandb.Api().api_key


class KindLocal(Kind):
    cluster_name = "metta-local"
    namespace = "orchestrator"
    helm_release_name = "orchestrator"
    helm_chart_path = repo_root / "devops/charts/orchestrator"
    environment_values_file: Path | None = repo_root / "devops/charts/orchestrator/environments/kind.yaml"
    context = f"kind-{cluster_name}"

    def _create_namespace(self, namespace: str) -> None:
        result = subprocess.run(["kubectl", "create", "namespace", namespace], check=True)
        if result.returncode != 0:
            error(f"Failed to create namespace {namespace}")
            raise Exception(f"Failed to create namespace {namespace}")

    def _maybe_load_secrets(self) -> None:
        wandb_api_key = self._get_wandb_api_key()
        if not wandb_api_key:
            error("No WANDB API key found. Please run 'wandb login' and try again.")
            sys.exit(1)

        # Lazy import to avoid loading app_backend dependencies when using metta CLI for linting.
        # This keeps the metta-cli install mode lightweight and fast.
        from metta.app_backend.clients.base_client import get_machine_token  # pylint: disable=import-outside-toplevel

        machine_token = get_machine_token(DEV_STATS_SERVER_URI)

        info("Creating secrets...")
        self._create_secret("wandb-api-secret", f"api-key={wandb_api_key}")
        self._create_secret("machine-token-secret", f"token={machine_token}")

    def build(self) -> None:
        result = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        cluster_exists = self.cluster_name in result.stdout.split()
        cluster_created = False

        if not cluster_exists:
            info("Creating Kind cluster...")
            subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)
            cluster_created = True
        else:
            result = subprocess.run(["kubectl", "cluster-info", "--context", self.context], capture_output=True)
            if result.returncode != 0:
                info("Cluster exists but is not healthy. Recreating...")
                subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
                subprocess.run(
                    ["docker", "rm", "-f", f"{self.cluster_name}-control-plane"],
                    capture_output=True,
                )
                subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)
                cluster_created = True
        self._use_appropriate_context()

        build_image(force_build=False)
        load_image_into_kind(force_load=cluster_created)
        success("Kind cluster ready")

        if not self._check_namespace_exists(self.namespace):
            self._create_namespace(self.namespace)
        success(f"{self.namespace} ready")

        if not self._check_namespace_exists("jobs"):
            self._create_namespace("jobs")
        success("jobs namespace ready")


kind_app = typer.Typer(help="Manage Kind cluster", rich_markup_mode="rich", no_args_is_help=True)

kind_local = KindLocal()


@kind_app.command(name="build")
def cmd_build():
    kind_local.build()
    console.print("[green]Kind cluster created[/green]")


@kind_app.command(name="up")
def cmd_up():
    kind_local.up()
    console.print("[green]Orchestrator started[/green]")


@kind_app.command(name="down")
def cmd_down():
    kind_local.down()
    console.print("[green]Orchestrator stopped[/green]")


@kind_app.command(name="clean")
def cmd_clean():
    kind_local.clean()
    console.print("[green]Cluster deleted[/green]")


@kind_app.command(name="get-pods")
def cmd_get_pods():
    kind_local.get_pods()


@kind_app.command(name="logs")
def cmd_logs(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    kind_local.logs(pod_name)


@kind_app.command(name="enter")
def cmd_enter(pod_name: Annotated[str | None, typer.Argument(help="Pod name")] = None):
    kind_local.enter(pod_name)


def main():
    kind_app()


if __name__ == "__main__":
    main()
