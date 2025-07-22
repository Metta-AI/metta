import subprocess
import sys
from pathlib import Path
from typing import Callable

from metta.common.util.stats_client_cfg import get_machine_token
from metta.setup.utils import error, info, success


class Kind:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.cluster_name = "metta-local"
        self.namespace = "orchestrator"
        self.helm_release_name = "orchestrator"
        self.helm_chart_path = self.repo_root / "devops/charts/orchestrator"

    def _ensure_docker_img_built(self, img_name: str, load_fn: Callable[[], None]) -> None:
        result = subprocess.run(["docker", "image", "inspect", img_name], capture_output=True)
        if result.returncode != 0:
            info(f"Building {img_name} image...")
            load_fn()
        info(f"Loading {img_name} into Kind...")
        subprocess.run(["kind", "load", "docker-image", img_name, "--name", self.cluster_name], check=True)

    def _use_local_context(self) -> None:
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)

    def build(self) -> None:
        """Create Kind cluster and set up for Metta."""
        # Check if cluster exists
        result = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        cluster_exists = self.cluster_name in result.stdout.split()

        if not cluster_exists:
            info("Creating Kind cluster...")
            subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)
        else:
            # Verify cluster is healthy
            result = subprocess.run(
                ["kubectl", "cluster-info", "--context", f"kind-{self.cluster_name}"], capture_output=True
            )
            if result.returncode != 0:
                info("Cluster exists but is not healthy. Recreating...")
                subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
                subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)

        self._use_local_context()

        from metta.setup.local_commands import LocalCommands

        for img_name, load_fn in [
            ("metta-local:latest", lambda: LocalCommands(self.repo_root).build_docker_img()),
            ("metta-app-backend:latest", lambda: LocalCommands(self.repo_root).build_app_backend_img()),
        ]:
            self._ensure_docker_img_built(img_name, load_fn)

        # No need to create RBAC manually - Helm chart will handle it
        success("Kind cluster ready!")

    def up(self) -> None:
        """Start orchestrator in Kind cluster using Helm."""
        self._use_local_context()
        # Get credentials
        wandb_api_key = self._get_wandb_api_key()
        if not wandb_api_key:
            error("No WANDB API key found. Please run 'wandb login' and try again.")
            sys.exit(1)

        machine_token = get_machine_token("http://localhost:8000")

        # Create namespace if it doesn't exist
        info("Creating namespace if needed...")
        result = subprocess.run(
            ["kubectl", "get", "namespace", self.namespace], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            subprocess.run(["kubectl", "create", "namespace", self.namespace], check=True)

        # Create secrets (matching production secret names)
        info("Creating secrets...")
        # Delete existing secrets if they exist
        subprocess.run(
            ["kubectl", "delete", "secret", "wandb-api-secret", "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )
        subprocess.run(
            ["kubectl", "delete", "secret", "machine-token-secret", "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )

        # Create new secrets
        subprocess.run(
            [
                "kubectl",
                "create",
                "secret",
                "generic",
                "wandb-api-secret",
                f"--from-literal=api-key={wandb_api_key}",
                "-n",
                self.namespace,
            ],
            check=True,
        )

        subprocess.run(
            [
                "kubectl",
                "create",
                "secret",
                "generic",
                "machine-token-secret",
                f"--from-literal=token={machine_token}",
                "-n",
                self.namespace,
            ],
            check=True,
        )

        # Use existing kind.yaml values file
        kind_values_file = self.repo_root / "devops/charts/orchestrator/environments/kind.yaml"

        try:
            # Check if release already exists
            result = subprocess.run(["helm", "list", "-n", self.namespace, "-q"], capture_output=True, text=True)

            if self.helm_release_name in result.stdout:
                info("Upgrading existing Helm release...")
                subprocess.run(
                    [
                        "helm",
                        "upgrade",
                        self.helm_release_name,
                        str(self.helm_chart_path),
                        "-n",
                        self.namespace,
                        "-f",
                        str(kind_values_file),
                    ],
                    check=True,
                )
            else:
                info("Installing orchestrator via Helm...")
                subprocess.run(
                    [
                        "helm",
                        "install",
                        self.helm_release_name,
                        str(self.helm_chart_path),
                        "-n",
                        self.namespace,
                        "-f",
                        str(kind_values_file),
                    ],
                    check=True,
                )

            info("Orchestrator deployed via Helm")
            info("To view pods: metta local kind get-pods")
            info("To view logs: metta local kind logs <pod-name>")
            info("To stop: metta local kind down")

        except Exception as e:
            error(f"Failed to deploy orchestrator: {e}")
            raise

    def down(self) -> None:
        """Stop orchestrator and worker pods."""
        info("Stopping...")
        self._use_local_context()

        # Uninstall Helm release
        subprocess.run(
            ["helm", "uninstall", self.helm_release_name, "-n", self.namespace, "--ignore-not-found"], check=True
        )

        # Clean up any remaining worker pods
        subprocess.run(
            ["kubectl", "delete", "pods", "-l", "app=eval-worker", "-n", self.namespace, "--ignore-not-found=true"],
            check=True,
        )

        # Clean up secrets
        subprocess.run(
            [
                "kubectl",
                "delete",
                "secret",
                "wandb-api-secret",
                "machine-token-secret",
                "-n",
                self.namespace,
                "--ignore-not-found=true",
            ],
            check=True,
        )

        success("Stopped (cluster preserved for faster restarts)")

    def clean(self) -> None:
        """Delete the Kind cluster."""
        info("Deleting cluster...")
        self._use_local_context()
        subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
        success("Cluster deleted")

    def get_pods(self) -> None:
        """Get list of pods in the cluster."""
        self._use_local_context()
        subprocess.run(["kubectl", "get", "pods", "-n", self.namespace], check=True)

    def logs(self, pod_name: str | None = None) -> None:
        """Follow logs for orchestrator or specific pod."""
        self._use_local_context()

        if pod_name:
            subprocess.run(["kubectl", "logs", pod_name, "-n", self.namespace, "--follow"], check=True)
        else:
            # Default to orchestrator logs
            subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, "-l", "app.kubernetes.io/name=orchestrator", "--follow"],
                check=True,
            )

    def enter(self, pod_name: str | None = None) -> None:
        """Enter orchestrator or specific pod with an interactive shell."""
        self._use_local_context()

        if not pod_name:
            # Get orchestrator pod name
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
        """Get WANDB API key from .netrc file."""
        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            try:
                with open(netrc_path, "r") as f:
                    content = f.read()
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if "machine api.wandb.ai" in line:
                            # Look for login and password in subsequent lines
                            for j in range(i + 1, min(i + 3, len(lines))):
                                parts = lines[j].split()
                                if len(parts) >= 2 and parts[0] == "login":
                                    # Look for password
                                    for k in range(j, min(j + 2, len(lines))):
                                        parts2 = lines[k].split()
                                        if len(parts2) >= 2 and parts2[0] == "password":
                                            return parts2[1]
            except Exception:
                pass
        return None
