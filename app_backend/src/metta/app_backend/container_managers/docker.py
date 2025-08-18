import logging
import os
import subprocess

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.worker_managers.worker import Worker
from metta.common.datadog.config import datadog_config


class DockerContainerManager(AbstractContainerManager):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def start_worker_container(
        self,
        backend_url: str,
        docker_image: str,
        machine_token: str,
        dd_env_vars: dict[str, str] | None = None,
    ) -> str:
        container_name = self._format_container_name()
        env_vars = {
            "BACKEND_URL": backend_url,
            "WORKER_ASSIGNEE": container_name,
            "MACHINE_TOKEN": machine_token,
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            **datadog_config.to_env_dict(),
            "DD_SERVICE": "eval-worker",
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
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._logger.info(f"Started worker container {container_name}")
            return container_name
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker container: {e.stderr}")
            raise

    def cleanup_container(self, name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                check=False,
            )
        except Exception as e:
            self._logger.warning(f"Failed to cleanup container {name}: {e}")
        else:
            self._logger.info(f"Cleaned up container {name}")

    async def discover_alive_workers(self) -> list[Worker]:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=eval-worker-",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        workers = []
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) == 3:
                    container_name = parts[1]
                    status_info = parts[2]
                    if container_name.startswith("eval-worker-"):
                        try:
                            # Map Docker status to simplified status
                            # Docker status is like "Up 5 minutes" or "Exited (0) 2 minutes ago"
                            if status_info.startswith("Up"):
                                status = "Running"
                            elif status_info.startswith("Exited"):
                                status = "Failed"
                            else:
                                status = "Unknown"

                            workers.append(Worker(name=container_name, status=status))
                        except ValueError:
                            self._logger.warning(f"Skipping container with invalid name: {container_name}")
        return workers
