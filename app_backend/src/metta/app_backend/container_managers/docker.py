import logging
import os
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.models import WorkerInfo


class DockerContainerManager(AbstractContainerManager):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def start_worker_container(
        self,
        git_hash: str,
        backend_url: str,
        docker_image: str,
        machine_token: str,
    ) -> WorkerInfo:
        container_name = self._format_container_name(git_hash)
        env_vars = {
            "BACKEND_URL": backend_url,
            "GIT_HASH": git_hash,
            "WORKER_ASSIGNEE": container_name,
            "MACHINE_TOKEN": machine_token,
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        }

        cmd = [
            "docker",
            "run",
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
                    if container_name.startswith("eval-worker-"):
                        try:
                            git_hash, _ = self._parse_container_name(container_name)
                            workers.append(
                                WorkerInfo(
                                    git_hash=git_hash,
                                    container_id=container_id,
                                    container_name=container_name,
                                )
                            )
                        except ValueError:
                            self._logger.warning(f"Skipping container with invalid name: {container_name}")
        return workers
