import json
import logging
import os
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo


class K8sPodManager(AbstractContainerManager):
    def __init__(self, namespace: str | None = None, kubeconfig: str | None = None, wandb_api_key: str | None = None):
        self._logger = logging.getLogger(__name__)
        self._namespace = namespace or os.environ.get("KUBERNETES_NAMESPACE", "observatory")
        self._kubeconfig = kubeconfig or os.environ.get("KUBERNETES_KUBECONFIG", None)
        self._wandb_api_key = wandb_api_key or os.environ.get("WANDB_API_KEY", None)

    def _get_kubectl_cmd(self) -> list[str]:
        cmd = ["kubectl"]
        if self._kubeconfig:
            cmd.extend(["--kubeconfig", self._kubeconfig])
        cmd.extend(["--namespace", self._namespace])
        return cmd

    def _get_pod_manifest(self, git_hash: str, backend_url: str, docker_image: str) -> dict:
        pod_name = self._format_container_name(git_hash)
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "labels": {
                    "app": "eval-worker",
                    "git-hash": git_hash,
                    "created-by": "eval-task-orchestrator",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "eval-worker",
                        "image": docker_image,
                        "imagePullPolicy": "Never",
                        "command": ["uv", "run", "python", "-m", "metta.app_backend.eval_task_worker"],
                        "env": [
                            {"name": "BACKEND_URL", "value": backend_url},
                            {"name": "GIT_HASH", "value": git_hash},
                            {"name": "WORKER_ASSIGNEE", "value": pod_name},
                            {"name": "WANDB_API_KEY", "value": self._wandb_api_key},
                        ],
                    }
                ],
            },
        }

    def start_worker_container(self, git_hash: str, backend_url: str, docker_image: str) -> WorkerInfo:
        # Create pod via kubectl
        pod_manifest = self._get_pod_manifest(git_hash, backend_url, docker_image)
        pod_name = pod_manifest["metadata"]["name"]
        cmd = self._get_kubectl_cmd() + ["create", "-f", "-"]

        self._logger.info(f"Starting worker pod for git hash {git_hash}")

        try:
            subprocess.run(cmd, input=json.dumps(pod_manifest), capture_output=True, text=True, check=True)

            # Get pod UID as container_id
            get_uid_cmd = self._get_kubectl_cmd() + ["get", "pod", pod_name, "-o", "jsonpath={.metadata.uid}"]
            uid_result = subprocess.run(get_uid_cmd, capture_output=True, text=True, check=True)
            pod_uid = uid_result.stdout.strip()

            self._logger.info(f"Started worker pod {pod_name} (UID: {pod_uid[:12]})")
            return WorkerInfo(
                git_hash=git_hash,
                container_id=pod_uid,
                container_name=pod_name,
                alive=True,
                task=None,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker pod: {e.stderr}")
            raise

    def cleanup_container(self, container_id: str) -> None:
        """Delete a Kubernetes pod."""
        try:
            # First try to get pod name by UID
            get_name_cmd = self._get_kubectl_cmd() + [
                "get",
                "pods",
                "-o",
                f"jsonpath={{.items[?(@.metadata.uid=='{container_id}')].metadata.name}}",
            ]
            name_result = subprocess.run(get_name_cmd, capture_output=True, text=True, check=False)

            pod_name = container_id
            if name_result.returncode == 0 and (stripped_name := name_result.stdout.strip()):
                pod_name = stripped_name

            # Delete the pod
            delete_cmd = self._get_kubectl_cmd() + ["delete", "pod", pod_name, "--force", "--grace-period=0"]
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                self._logger.info(f"Cleaned up pod {pod_name} (requested as {container_id})")
            else:
                self._logger.warning(f"Failed to delete pod {pod_name}: {result.stderr}")
        except subprocess.CalledProcessError as e:
            self._logger.warning(f"Failed to cleanup pod {container_id}: {e}")
        except Exception as e:
            self._logger.warning(f"Unexpected error cleaning up pod {container_id}: {e}")

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        # Get pods with eval-worker label
        cmd = self._get_kubectl_cmd() + ["get", "pods", "-l", "app=eval-worker", "-o", "json"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        pods_data = json.loads(result.stdout)
        workers = []

        for pod in pods_data.get("items", []):
            metadata = pod.get("metadata", {})
            status = pod.get("status", {})

            # Check if pod is running and not being deleted
            phase = status.get("phase", "")
            deletion_timestamp = metadata.get("deletionTimestamp")

            if phase in ["Running", "Pending"] and not deletion_timestamp:
                pod_name = metadata.get("name", "")
                pod_uid = metadata.get("uid", "")
                git_hash = metadata.get("labels", {}).get("git-hash", "")

                if pod_name.startswith("eval-worker-") and git_hash:
                    workers.append(
                        WorkerInfo(
                            git_hash=git_hash,
                            container_id=pod_uid,
                            container_name=pod_name,
                            alive=True,
                            task=None,
                        )
                    )

        if workers:
            self._logger.info(f"Discovered {len(workers)} alive workers in Kubernetes")
        return workers
