import asyncio
import logging
import random
import string
import threading
from datetime import datetime, timezone
from typing import Callable

from metta.app_backend.eval_task_worker import EvalTaskWorker
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.worker import Worker


class ThreadWorkerManager(AbstractWorkerManager):
    """Manages EvalTaskWorker instances running on separate threads."""

    _worker_prefix = "eval-worker-"

    def _format_worker_name(self) -> str:
        return f"{self._worker_prefix}-{self._generate_worker_suffix()}"

    def _generate_worker_suffix(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}-{random_suffix}"

    def __init__(
        self,
        create_worker: Callable[[str], EvalTaskWorker],
        logger: logging.Logger | None = None,
    ):
        self._create_worker = create_worker
        self._logger = logger or logging.getLogger(__name__)
        self._workers: dict[str, dict] = {}  # worker_name -> {thread, worker, stop_event}
        self._lock = threading.Lock()

    def start_worker(self) -> str:
        """Start a worker on a new thread."""
        worker_name = self._format_worker_name()

        with self._lock:
            if worker_name in self._workers:
                raise ValueError(f"Worker {worker_name} already exists")

            # Create worker instance with the generated worker name
            worker = self._create_worker(worker_name)

            # Create stop event for graceful shutdown
            stop_event = threading.Event()

            # Create and start thread
            thread = threading.Thread(
                target=self._run_worker, args=(worker, stop_event), name=f"worker-{worker_name}", daemon=True
            )

            self._workers[worker_name] = {
                "thread": thread,
                "worker": worker,
                "stop_event": stop_event,
            }

            thread.start()
            self._logger.info(f"Started worker {worker_name} on thread {thread.name}")

            return worker_name

    def _run_worker(self, worker: EvalTaskWorker, stop_event: threading.Event) -> None:
        """Run worker in a thread with event loop."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_with_stop_check():
                """Run worker but check stop event periodically."""
                try:
                    # Create task for worker.run()
                    worker_task = asyncio.create_task(worker.run())

                    # Check stop event every second
                    while not stop_event.is_set():
                        if worker_task.done():
                            break
                        await asyncio.sleep(1)

                    # Cancel worker if stop requested
                    if not worker_task.done():
                        worker_task.cancel()
                        try:
                            await worker_task
                        except asyncio.CancelledError:
                            self._logger.info(f"Worker {worker._assignee} cancelled")

                except Exception as e:
                    self._logger.error(f"Error in worker {worker._assignee}: {e}", exc_info=True)
                finally:
                    # Clean up worker resources
                    if hasattr(worker, "__aexit__"):
                        await worker.__aexit__(None, None, None)

            loop.run_until_complete(run_with_stop_check())

        except Exception as e:
            self._logger.error(f"Thread error for worker {worker._assignee}: {e}", exc_info=True)
        finally:
            loop.close()

    def cleanup_worker(self, worker_name: str) -> None:
        """Stop and cleanup a worker."""
        with self._lock:
            if worker_name not in self._workers:
                self._logger.warning(f"Worker {worker_name} not found for cleanup")
                return

            worker_info = self._workers[worker_name]
            stop_event = worker_info["stop_event"]
            thread = worker_info["thread"]

            # Signal worker to stop
            stop_event.set()

            # Wait for thread to finish (with timeout)
            thread.join(timeout=10.0)

            if thread.is_alive():
                self._logger.warning(f"Worker thread {worker_name} did not stop gracefully")
            else:
                self._logger.info(f"Worker {worker_name} stopped successfully")

            # Remove from tracking
            del self._workers[worker_name]

    async def discover_alive_workers(self) -> list[Worker]:
        """Discover all alive workers."""
        alive_workers = []

        with self._lock:
            # Create a copy to avoid modification during iteration
            workers_copy = dict(self._workers)

        for worker_name, worker_info in workers_copy.items():
            thread = worker_info["thread"]

            if thread.is_alive():
                # For thread workers, we consider them "Running" if thread is alive
                alive_workers.append(Worker(name=worker_name, status="Running"))
            else:
                # Clean up dead workers
                self._logger.info(f"Removing dead worker {worker_name}")
                with self._lock:
                    if worker_name in self._workers:
                        del self._workers[worker_name]

        return alive_workers

    def shutdown_all(self) -> None:
        """Shutdown all workers gracefully."""
        self._logger.info("Shutting down all workers")

        with self._lock:
            worker_names = list(self._workers.keys())

        for worker_name in worker_names:
            try:
                self.cleanup_worker(worker_name)
            except Exception as e:
                self._logger.error(f"Error cleaning up worker {worker_name}: {e}")

        self._logger.info("All workers shutdown complete")
