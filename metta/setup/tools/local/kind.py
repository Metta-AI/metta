import subprocess
import sys
from pathlib import Path
from typing import Callable

from devops.docker.push_image import push_image
from metta.common.util.constants import DEV_STATS_SERVER_URI, METTA_AWS_ACCOUNT_ID, METTA_AWS_REGION
from metta.common.util.fs import get_repo_root
from metta.common.util.stats_client_cfg import get_machine_token
from metta.setup.utils import error, info, success

repo_root = get_repo_root()


class Kind:
    cluster_name: str
    namespace: str
    helm_release_name: str
    helm_chart_path: Path
    environment_values_file: Path | None
    context: str

    def _check_namespace_exists(self) -> bool:
        result = subprocess.run(
            ["kubectl", "get", "namespace", self.namespace], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0

    def _ensure_docker_img_built(self, img_name: str, load_fn: Callable[[], None]) -> None:
        result = subprocess.run(["docker", "image", "inspect", img_name], capture_output=True)
        if result.returncode != 0:
            info(f"Building {img_name} image...")
            load_fn()
        info(f"Loading {img_name} into Kind...")
        subprocess.run(["kind", "load", "docker-image", img_name, "--name", self.cluster_name], check=True)

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

        # Build helm command with base values
        helm_cmd = [
            "helm",
            cmd,
            self.helm_release_name,
            str(self.helm_chart_path),
            "-n",
            self.namespace,
        ]

        # Add environment values file if present
        if self.environment_values_file:
            helm_cmd.extend(["-f", str(self.environment_values_file)])

        subprocess.run(helm_cmd, check=True)

        info("Orchestrator deployed via Helm")

        info("To view pods: metta local kind get-pods")
        info("To view logs: metta local kind logs <pod-name>")
        info("To stop: metta local kind down")

    def down(self) -> None:
        """Stop orchestrator and worker pods."""
        info("Stopping...")
        self._use_appropriate_context()

        # Uninstall Helm release
        subprocess.run(
            ["helm", "uninstall", self.helm_release_name, "-n", self.namespace, "--ignore-not-found"], check=True
        )

        # Clean up any remaining worker pods
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
            # Default to orchestrator logs
            subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, "-l", "app.kubernetes.io/name=orchestrator", "--follow"],
                check=True,
            )

    def enter(self, pod_name: str | None = None) -> None:
        """Enter orchestrator or specific pod with an interactive shell."""
        self._use_appropriate_context()

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
        import wandb

        return wandb.Api().api_key


class KindLocal(Kind):
    cluster_name = "metta-local"
    namespace = "orchestrator"
    helm_release_name = "orchestrator"
    helm_chart_path = repo_root / "devops/charts/orchestrator"
    environment_values_file: Path | None = repo_root / "devops/charts/orchestrator/environments/kind.yaml"
    context = f"kind-{cluster_name}"

    def _create_namespace(self) -> None:
        result = subprocess.run(["kubectl", "create", "namespace", self.namespace], check=True)
        if result.returncode != 0:
            error(f"Failed to create namespace {self.namespace}")
            raise Exception(f"Failed to create namespace {self.namespace}")

    def _maybe_load_secrets(self) -> None:
        wandb_api_key = self._get_wandb_api_key()
        if not wandb_api_key:
            error("No WANDB API key found. Please run 'wandb login' and try again.")
            sys.exit(1)

        machine_token = get_machine_token(DEV_STATS_SERVER_URI)

        info("Creating secrets...")
        self._create_secret("wandb-api-secret", f"api-key={wandb_api_key}")
        self._create_secret("machine-token-secret", f"token={machine_token}")

    def build(self) -> None:
        result = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        cluster_exists = self.cluster_name in result.stdout.split()

        if not cluster_exists:
            info("Creating Kind cluster...")
            subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)
        else:
            # Verify cluster is healthy
            result = subprocess.run(["kubectl", "cluster-info", "--context", self.context], capture_output=True)
            if result.returncode != 0:
                info("Cluster exists but is not healthy. Recreating...")
                subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
                subprocess.run(["kind", "create", "cluster", "--name", self.cluster_name], check=True)
        self._use_appropriate_context()

        from metta.setup.local_commands import LocalCommands

        self._ensure_docker_img_built("metta-policy-evaluator-local:latest", LocalCommands().build_policy_evaluator_img)
        success("Kind cluster ready")

        if not self._check_namespace_exists():
            self._create_namespace()
        success(f"{self.namespace} ready")


class EksProd(Kind):
    aws_account_id = METTA_AWS_ACCOUNT_ID
    aws_region = METTA_AWS_REGION
    cluster_name = f"arn:aws:eks:{aws_region}:{aws_account_id}:cluster/main"
    namespace = "orchestrator"
    context = cluster_name
    helm_release_name = "orchestrator"
    helm_chart_path = repo_root / "devops/charts/orchestrator"
    environment_values_file = None

    def build(self):
        info("Building AMD64 for EKS...")

        local_image_name = "metta-policy-evaluator-local:latest-amd64"
        from metta.setup.local_commands import LocalCommands

        LocalCommands().build_policy_evaluator_img(
            tag=local_image_name,
            build_args=["--platform", "linux/amd64"],
        )
        push_image(
            local_image_name=local_image_name,
            remote_image_name="metta-policy-evaluator:latest",
            region=self.aws_region,
            account_id=self.aws_account_id,
        )
