from abc import ABC, abstractmethod

from metta.app_backend.container_managers.models import WorkerInfo


class AbstractWorkerManager(ABC):
    """Abstract base class for managing eval task workers."""

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
