import logging
import os
import socket
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo


class DockerContainerManager(AbstractContainerManager):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def start_worker_container(self, git_hash: str, backend_url: str, docker_image: str) -> WorkerInfo:
        suffix = self.generate_container_suffix()
        container_name = f"eval-worker-{git_hash}-{suffix}"
        worker_assignee = f"worker-{git_hash[:8]}-{socket.gethostname()}-{os.getpid()}"

        env_vars = {
            "BACKEND_URL": backend_url,
            "GIT_HASH": git_hash,
            "WORKER_ASSIGNEE": worker_assignee,
        }

        cmd = [
            "docker",
            "run",
            # "--rm",  # Remove container when it exits
            "-d",  # Run in detached mode
            "--name",
            container_name,
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Worker will set up its own versioned checkout
        cmd.extend([docker_image, "uv", "run", "python", "-m", "metta.app_backend.eval_task_worker"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            self._logger.info(f"Started worker container {container_name} ({container_id[:12]})")
            return WorkerInfo(
                git_hash=git_hash,
                container_id=container_id,
                container_name=container_name,
                alive=True,
                task=None,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker container: {e.stderr}")
            raise

    def cleanup_container(self, container_id: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                check=False,
            )
        except Exception as e:
            self._logger.warning(f"Failed to cleanup container {container_id}: {e}")
        else:
            self._logger.info(f"Cleaned up container {container_id}")

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=eval-worker-",
                    "--format",
                    "{{.ID}}\t{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            workers = []
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        container_id, container_name = parts
                        # Extract git hash from name (format: eval-worker-{git_hash}-{suffix})
                        if container_name.startswith("eval-worker-"):
                            name_parts = container_name.split("-", 3)
                            if len(name_parts) >= 3:
                                git_hash = name_parts[2]
                                workers.append(
                                    WorkerInfo(
                                        git_hash=git_hash,
                                        container_id=container_id,
                                        container_name=container_name,
                                        alive=True,
                                        task=None,
                                    )
                                )

            if workers:
                self._logger.info(f"Discovered {len(workers)} alive workers")
            return workers
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to discover workers: {e}")
            return []
