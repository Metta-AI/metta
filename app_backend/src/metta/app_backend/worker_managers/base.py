from abc import ABC, abstractmethod

from metta.app_backend.worker_managers.worker import Worker


class AbstractWorkerManager(ABC):
    """Abstract base class for managing eval task workers."""

    @abstractmethod
    def start_worker(self, num_cpus_request: int = 3, memory_request: int = 12) -> str:
        """Start a worker

        Args:
            num_cpus_request: Number of CPUs to request (3, 7, 11, or 15)
            memory_request: Memory to request (in GB)

        Returns:
            WorkerInfo object with worker details
        """
        pass

    @abstractmethod
    def cleanup_worker(self, worker_name: str) -> None:
        """Remove/cleanup a worker.

        Args:
            worker_name: The worker name to cleanup
        """
        pass

    @abstractmethod
    async def discover_alive_workers(self) -> list[Worker]:
        """Discover all alive workers.

        Returns:
            List of Worker objects for alive workers
        """
        pass
