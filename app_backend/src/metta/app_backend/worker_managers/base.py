import abc

import metta.app_backend.worker_managers.worker


class AbstractWorkerManager(abc.ABC):
    """Abstract base class for managing eval task workers."""

    @abc.abstractmethod
    def start_worker(self) -> str:
        """Start a worker

        Args:
            client: The EvalTaskClient for the worker to use
            machine_token: Authentication token for the worker

        Returns:
            WorkerInfo object with worker details
        """
        pass

    @abc.abstractmethod
    def cleanup_worker(self, worker_name: str) -> None:
        """Remove/cleanup a worker.

        Args:
            worker_name: The worker name to cleanup
        """
        pass

    @abc.abstractmethod
    async def discover_alive_workers(self) -> list[metta.app_backend.worker_managers.worker.Worker]:
        """Discover all alive workers.

        Returns:
            List of Worker objects for alive workers
        """
        pass
