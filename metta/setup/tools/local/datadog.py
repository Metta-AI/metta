import os
import subprocess
import sys

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success

repo_root = get_repo_root()


class DatadogLocal:
    cluster_name = "metta-local"
    namespace = "datadog"
    helm_release_name = "datadog"
    helm_chart_path = repo_root / "devops/charts/datadog"
    environment_values_file = repo_root / "devops/charts/datadog/environments/kind.yaml"
    context = f"kind-{cluster_name}"

    def _use_appropriate_context(self) -> None:
        subprocess.run(["kubectl", "config", "use-context", self.context], check=True)

    def _check_namespace_exists(self) -> bool:
        result = subprocess.run(
            ["kubectl", "get", "namespace", self.namespace], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0

    def _create_namespace(self) -> None:
        result = subprocess.run(["kubectl", "create", "namespace", self.namespace])
        if result.returncode != 0:
            error(f"Failed to create namespace {self.namespace}")
            raise Exception(f"Failed to create namespace {self.namespace}")

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

    def _create_datadog_secret(self) -> None:
        datadog_api_key = os.environ.get("DD_API_KEY")
        if not datadog_api_key:
            error("DD_API_KEY environment variable not set. Please set it and try again.")
            sys.exit(1)

        info("Creating Datadog API key secret...")
        self._create_secret("datadog-secret", f"api-key={datadog_api_key}")

    def up(self) -> None:
        """Deploy Datadog to Kind cluster using Helm."""
        self._use_appropriate_context()

        # Check if Kind cluster exists
        result = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        if self.cluster_name not in result.stdout.split():
            error(f"Kind cluster '{self.cluster_name}' not found. Please run 'metta local kind build' first.")
            sys.exit(1)

        info("Creating namespace if needed...")
        if not self._check_namespace_exists():
            self._create_namespace()

        # Create Datadog secret
        self._create_datadog_secret()

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
            "-f",
            str(self.environment_values_file),
        ]

        subprocess.run(helm_cmd, check=True)
        success("Datadog deployed via Helm")

        info("To view Datadog pods: kubectl get pods -n datadog")
        info("To view logs: kubectl logs -n datadog -l app=datadog --follow")
        info("To stop: metta local datadog down")

    def down(self) -> None:
        """Stop Datadog deployment."""
        info("Stopping Datadog...")
        self._use_appropriate_context()

        # Uninstall Helm release
        subprocess.run(
            ["helm", "uninstall", self.helm_release_name, "-n", self.namespace, "--ignore-not-found"], check=True
        )

        success("Datadog stopped")

    def status(self) -> None:
        """Check status of Datadog deployment."""
        self._use_appropriate_context()

        info("Datadog pods:")
        subprocess.run(["kubectl", "get", "pods", "-n", self.namespace], check=True)

        info("\nDatadog services:")
        subprocess.run(["kubectl", "get", "services", "-n", self.namespace], check=True)

    def logs(self, follow: bool = True) -> None:
        """View Datadog agent logs."""
        self._use_appropriate_context()

        cmd = ["kubectl", "logs", "-n", self.namespace, "-l", "app=datadog"]
        if follow:
            cmd.append("--follow")

        subprocess.run(cmd, check=True)
