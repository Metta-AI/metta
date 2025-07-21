import os
import subprocess
from pathlib import Path

from metta.common.util.stats_client_cfg import get_machine_token
from metta.setup.utils import info, success


class Kind:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.cluster_name = "metta-local"

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

        # Set kubectl context
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)

        # Check if metta-local image exists
        result = subprocess.run(["docker", "image", "inspect", "metta-local:latest"], capture_output=True)
        if result.returncode != 0:
            info("Building metta-local image...")
            # Import here to avoid circular dependency
            from metta.setup.local_commands import LocalCommands

            local_commands = LocalCommands(self.repo_root)
            local_commands.build_docker_img(None)

        info("Loading metta-local:latest into Kind...")
        subprocess.run(["kind", "load", "docker-image", "metta-local:latest", "--name", self.cluster_name], check=True)

        # Create RBAC for pod management
        rbac_yaml = """apiVersion: v1
kind: ServiceAccount
metadata:
  name: default
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-manager
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "create", "delete", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: default-pod-manager
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: pod-manager
subjects:
- kind: ServiceAccount
  name: default
  namespace: default"""

        subprocess.run(["kubectl", "apply", "-f", "-"], input=rbac_yaml, text=True, check=True)
        success("Kind cluster ready!")

    def up(self) -> None:
        """Start orchestrator in Kind cluster."""
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)

        # Get WANDB API key
        wandb_api_key = self._get_wandb_api_key()
        docker_internal_host = "http://host.docker.internal:8000"
        backend_url = os.environ.get("BACKEND_URL", docker_internal_host)
        machine_token = get_machine_token(
            backend_url if backend_url != docker_internal_host else "http://localhost:8000"
        )

        # Run orchestrator pod
        cmd = [
            "kubectl",
            "run",
            "orchestrator",
            "--image=metta-local:latest",
            "--image-pull-policy=Never",
            "--env=CONTAINER_RUNTIME=k8s",
            "--env=KUBERNETES_NAMESPACE=default",
            "--env=DOCKER_IMAGE=metta-local:latest",
            f"--env=WANDB_API_KEY={wandb_api_key}",
            f"--env=MACHINE_TOKEN={machine_token}",
            f"--env=BACKEND_URL={backend_url}",
            "--restart=Never",
            "--command",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "metta.app_backend.eval_task_orchestrator",
        ]

        subprocess.run(cmd, check=True)

        info("Orchestrator running in Kind")
        info("")
        info(f"Using backend at: {backend_url}")
        info("")
        info("To view orchestrator logs: kubectl logs orchestrator -f")
        info("To view pods: kubectl get pods -w")
        info("To stop: metta local kind down")

    def down(self) -> None:
        """Stop orchestrator and worker pods."""
        info("Stopping...")
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)
        subprocess.run(["kubectl", "delete", "pod", "orchestrator", "--ignore-not-found=true"], check=True)
        subprocess.run(["kubectl", "delete", "pods", "-l", "app=eval-worker", "--ignore-not-found=true"], check=True)
        success("Stopped (cluster preserved for faster restarts)")

    def clean(self) -> None:
        """Delete the Kind cluster."""
        info("Deleting cluster...")
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)
        subprocess.run(["kind", "delete", "cluster", "--name", self.cluster_name], check=True)
        success("Cluster deleted")

    def get_pods(self) -> None:
        """Get list of pods in the cluster."""
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)
        subprocess.run(["kubectl", "get", "pods"], check=True)

    def logs(self, pod_name: str) -> None:
        """Follow logs for a specific pod."""
        subprocess.run(["kubectl", "config", "use-context", f"kind-{self.cluster_name}"], check=True)
        subprocess.run(["kubectl", "logs", pod_name, "--follow"], check=True)

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
