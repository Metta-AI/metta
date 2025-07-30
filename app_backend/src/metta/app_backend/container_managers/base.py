import random
import string
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from metta.app_backend.container_managers.models import WorkerInfo


class AbstractContainerManager(ABC):
    _container_prefix = "eval-worker-"

    def _format_container_name(self, git_hash: str) -> str:
        return f"{self._container_prefix}{git_hash}-{self._generate_container_suffix()}"

    def _parse_container_name(self, container_name: str) -> tuple[str, str]:
        git_hash, suffix = container_name.replace(self._container_prefix, "").split("-", 1)
        return git_hash, suffix

    def _generate_container_suffix(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}-{random_suffix}"

    @abstractmethod
    def start_worker_container(
        self,
        git_hash: str,
        backend_url: str,
        docker_image: str,
        machine_token: str,
    ) -> WorkerInfo:
        """Start a worker container for a specific git hash.

        Args:
            git_hash: The git hash for the worker
            backend_url: The backend URL for the worker to connect to
            docker_image: The Docker image to use

        Returns:
            WorkerInfo object with container details
        """
        pass

    @abstractmethod
    def cleanup_container(self, container_id: str) -> None:
        """Remove/cleanup a container.

        Args:
            container_id: The container ID to cleanup
        """
        pass

    @abstractmethod
    async def discover_alive_workers(self) -> list[WorkerInfo]:
        """Discover all alive workers.

        Returns:
            List of WorkerInfo objects for alive workers
        """
        pass
