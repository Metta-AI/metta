import json
import logging
import os
import socket
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo

logger = logging.getLogger(__name__)


class EksPodManager(AbstractContainerManager):
    def __init__(self, namespace: str = "observatory", kubeconfig: str | None = None):
        self._logger = logger
        self._namespace = namespace
        self._kubeconfig = kubeconfig

    def _get_kubectl_cmd(self) -> list[str]:
        """Get base kubectl command with optional kubeconfig."""
        cmd = ["kubectl"]
        if self._kubeconfig:
            cmd.extend(["--kubeconfig", self._kubeconfig])
        cmd.extend(["--namespace", self._namespace])
        return cmd

    def start_worker_container(self, git_hash: str, backend_url: str, docker_image: str) -> WorkerInfo:
        suffix = self.generate_container_suffix()
        pod_name = f"eval-worker-{git_hash}-{suffix}"
        worker_assignee = f"worker-{git_hash[:8]}-{socket.gethostname()}-{os.getpid()}"

        # Create pod manifest
        pod_manifest = {
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
                        "command": ["uv", "run", "python", "-m", "metta.app_backend.eval_task_worker"],
                        "env": [
                            {"name": "BACKEND_URL", "value": backend_url},
                            {"name": "GIT_HASH", "value": git_hash},
                            {"name": "WORKER_ASSIGNEE", "value": worker_assignee},
                        ],
                    }
                ],
            },
        }

        # Create pod via kubectl
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
        """Delete a Kubernetes pod.

        Note: container_id can be either the pod UID or pod name.
        """
        try:
            # First try to get pod name by UID
            get_name_cmd = self._get_kubectl_cmd() + [
                "get",
                "pods",
                "-o",
                f"jsonpath={{.items[?(@.metadata.uid=='{container_id}')].metadata.name}}",
            ]
            name_result = subprocess.run(get_name_cmd, capture_output=True, text=True, check=False)

            pod_name = name_result.stdout.strip() if name_result.returncode == 0 else container_id

            # Delete the pod
            delete_cmd = self._get_kubectl_cmd() + ["delete", "pod", pod_name, "--force", "--grace-period=0"]
            subprocess.run(delete_cmd, capture_output=True, check=False)

        except Exception as e:
            self._logger.warning(f"Failed to cleanup pod {container_id}: {e}")
        else:
            self._logger.info(f"Cleaned up pod {container_id}")

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        try:
            # Get pods with eval-worker label
            cmd = self._get_kubectl_cmd() + ["get", "pods", "-l", "app=eval-worker", "-o", "json"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            pods_data = json.loads(result.stdout)
            workers = []

            for pod in pods_data.get("items", []):
                metadata = pod.get("metadata", {})
                status = pod.get("status", {})

                # Check if pod is running
                phase = status.get("phase", "")
                if phase in ["Running", "Pending"]:
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
                self._logger.info(f"Discovered {len(workers)} alive workers in EKS")
            return workers

        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to discover workers in EKS: {e}")
            return []
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse kubectl output: {e}")
            return []
