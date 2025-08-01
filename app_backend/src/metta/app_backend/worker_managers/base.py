import random
import string
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from metta.app_backend.container_managers.models import WorkerInfo


class AbstractWorkerManager(ABC):
    """Abstract base class for managing eval task workers."""

    _worker_prefix = "eval-worker-"

    def _format_worker_name(self) -> str:
        return f"{self._worker_prefix}-{self._generate_worker_suffix()}"

    def _generate_worker_suffix(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}-{random_suffix}"

    @abstractmethod
    def start_worker(self) -> WorkerInfo:
        """Start a worker

        Args:
            client: The EvalTaskClient for the worker to use
            machine_token: Authentication token for the worker

        Returns:
            WorkerInfo object with worker details
        """
        pass

    @abstractmethod
    def cleanup_worker(self, worker_id: str) -> None:
        """Remove/cleanup a worker.

        Args:
            worker_id: The worker ID to cleanup
        """
        pass

    @abstractmethod
    async def discover_alive_workers(self) -> list[WorkerInfo]:
        """Discover all alive workers.

        Returns:
            List of WorkerInfo objects for alive workers
        """
        pass
