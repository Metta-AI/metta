import random
import string
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from metta.app_backend.worker_managers.worker import Worker


class AbstractContainerManager(ABC):
    _container_prefix = "eval-worker-"

    def _format_container_name(self) -> str:
        return f"{self._container_prefix}-{self._generate_container_suffix()}"

    def _generate_container_suffix(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}-{random_suffix}"

    @abstractmethod
    def start_worker_container(
        self,
        backend_url: str,
        docker_image: str,
        machine_token: str,
        num_cpus_request: int = 3,
        memory_request: int = 12,
    ) -> str:
        """Start a worker container

        Args:
            backend_url: The backend URL for the worker to connect to
            docker_image: The Docker image to use
            machine_token: Authentication token for the worker
            num_cpus_request: Number of CPUs to request (3, 7, 11, or 15)
            memory_request: Memory to request (in GB)

        Returns:
            WorkerInfo object with container details
        """
        pass

    @abstractmethod
    def cleanup_container(self, name: str) -> None:
        """Remove/cleanup a container.

        Args:
            container_id: The container ID to cleanup
        """
        pass

    @abstractmethod
    async def discover_alive_workers(self) -> list[Worker]:
        """Discover all alive workers.

        Returns:
            List of Worker objects for alive workers
        """
        pass
