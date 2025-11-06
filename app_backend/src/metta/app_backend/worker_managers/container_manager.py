import metta.app_backend.container_managers.base
import metta.app_backend.worker_managers.base
import metta.app_backend.worker_managers.worker


class ContainerWorkerManager(metta.app_backend.worker_managers.base.AbstractWorkerManager):
    """Adapter that wraps AbstractContainerManager to implement AbstractWorkerManager interface."""

    def __init__(
        self,
        container_manager: metta.app_backend.container_managers.base.AbstractContainerManager,
        backend_url: str,
        docker_image: str,
        machine_token: str,
    ):
        self._container_manager = container_manager
        self._backend_url = backend_url
        self._docker_image = docker_image
        self._machine_token = machine_token

    def start_worker(self) -> str:
        """Start a worker using the underlying container manager."""
        return self._container_manager.start_worker_container(
            backend_url=self._backend_url,
            docker_image=self._docker_image,
            machine_token=self._machine_token,
        )

    def cleanup_worker(self, worker_id: str) -> None:
        """Cleanup a worker using the underlying container manager."""
        self._container_manager.cleanup_container(worker_id)

    async def discover_alive_workers(self) -> list[metta.app_backend.worker_managers.worker.Worker]:
        """Discover alive workers using the underlying container manager."""
        return await self._container_manager.discover_alive_workers()
