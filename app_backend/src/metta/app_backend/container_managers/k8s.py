import json
import logging
import os
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.worker_managers.worker import Worker
from metta.common.datadog.config import datadog_config


class K8sPodManager(AbstractContainerManager):
    def __init__(self, namespace: str | None = None, kubeconfig: str | None = None, wandb_api_key: str | None = None):
        self._logger = logging.getLogger(__name__)
        self._namespace = namespace or os.environ.get("KUBERNETES_NAMESPACE", "orchestrator")
        self._kubeconfig = kubeconfig or os.environ.get("KUBERNETES_KUBECONFIG", None)
        self._wandb_api_key = wandb_api_key or os.environ.get("WANDB_API_KEY", "")

    def _get_kubectl_cmd(self) -> list[str]:
        cmd = ["kubectl"]
        if self._kubeconfig:
            cmd.extend(["--kubeconfig", self._kubeconfig])
        cmd.extend(["--namespace", self._namespace])
        return cmd

    def _get_pod_manifest(
        self,
        backend_url: str,
        docker_image: str,
        machine_token: str,
    ) -> dict:
        pod_name = self._format_container_name()
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "labels": {
                    "app": "eval-worker",
                    "created-by": "eval-task-orchestrator",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "serviceAccountName": os.getenv("KUBERNETES_SERVICE_ACCOUNT", f"orchestrator-{self._namespace}"),
                "containers": [
                    {
                        "name": "eval-worker",
                        "image": docker_image,
                        "imagePullPolicy": os.getenv("IMAGE_PULL_POLICY", "IfNotPresent"),
                        "command": ["uv", "run", "python", "-m", "metta.app_backend.eval_task_worker"],
                        "env": [
                            {"name": "BACKEND_URL", "value": backend_url},
                            {"name": "WORKER_ASSIGNEE", "value": pod_name},
                            {"name": "WANDB_API_KEY", "value": self._wandb_api_key},
                            {"name": "MACHINE_TOKEN", "value": machine_token},
                            *[{"name": k, "value": str(v)} for k, v in datadog_config.to_env_dict().items()],
                            {"name": "DD_SERVICE", "value": "eval-worker"},
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "3",
                                "memory": "1Gi",
                            },
                        },
                    }
                ],
                "tolerations": [
                    {
                        "key": "workload-type",
                        "operator": "Equal",
                        "value": "eval-worker",
                        "effect": "NoSchedule",
                    }
                ],
                "nodeSelector": {
                    "workload-type": "eval-worker",
                },
            },
        }

    def start_worker_container(
        self,
        backend_url: str,
        docker_image: str,
        machine_token: str,
    ) -> str:
        # Create pod via kubectl
        pod_manifest = self._get_pod_manifest(backend_url, docker_image, machine_token)
        pod_name = pod_manifest["metadata"]["name"]
        cmd = self._get_kubectl_cmd() + ["create", "-f", "-"]
        manifest_str = json.dumps(pod_manifest)

        self._logger.info("Starting worker pod")

        try:
            subprocess.run(cmd, input=manifest_str, capture_output=True, text=True, check=True)
            self._logger.info(f"Started worker pod {pod_name}")
            return pod_name
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker pod: {e.stderr}")
            raise

    def cleanup_container(self, pod_name: str) -> None:
        """Delete a Kubernetes pod."""
        try:
            # Delete the pod
            delete_cmd = self._get_kubectl_cmd() + ["delete", "pod", pod_name, "--force", "--grace-period=0"]
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                self._logger.info(f"Cleaned up pod {pod_name}")
            else:
                self._logger.warning(f"Failed to delete pod {pod_name}: {result.stderr}")
        except subprocess.CalledProcessError as e:
            self._logger.warning(f"Failed to cleanup pod {pod_name}: {e}")
        except Exception as e:
            self._logger.warning(f"Unexpected error cleaning up pod {pod_name}: {e}")

    async def discover_alive_workers(self) -> list[Worker]:
        # Get pods with eval-worker label
        cmd = self._get_kubectl_cmd() + ["get", "pods", "-l", "app=eval-worker", "-o", "json"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        pods_data = json.loads(result.stdout)
        workers: list[Worker] = []

        for pod in pods_data.get("items", []):
            metadata = pod.get("metadata", {})
            status = pod.get("status", {})

            # Check if pod is not being deleted
            phase = status.get("phase", "Unknown")
            deletion_timestamp = metadata.get("deletionTimestamp")

            if not deletion_timestamp:
                pod_name = metadata.get("name", "")
                if pod_name.startswith("eval-worker-"):
                    workers.append(Worker(name=pod_name, status=phase))

        if workers:
            self._logger.info(f"Discovered {len(workers)} alive workers in Kubernetes")
        return workers
