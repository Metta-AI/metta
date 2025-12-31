import asyncio
import os

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.eval_task_orchestrator import EvalTaskOrchestrator, init_logging
from metta.app_backend.eval_task_worker import EvalTaskWorker, SimTaskExecutor
from metta.app_backend.worker_managers.thread_manager import ThreadWorkerManager
from metta.common.datadog.config import datadog_config
from metta.common.datadog.tracing import init_tracing
from metta.common.util.constants import DEV_STATS_SERVER_URI


async def main() -> None:
    init_logging()
    datadog_config.DD_TRACE_ENABLED = False
    init_tracing()

    backend_url = os.environ.get("BACKEND_URL", DEV_STATS_SERVER_URI)
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    task_timeout_minutes = float(os.environ.get("TASK_TIMEOUT_MINUTES", "60"))

    task_client = EvalTaskClient(backend_url)

    def create_worker(name: str) -> EvalTaskWorker:
        return EvalTaskWorker(task_client, SimTaskExecutor(backend_url), name, poll_interval)

    worker_manager = ThreadWorkerManager(create_worker)
    orchestrator = EvalTaskOrchestrator(
        task_client=task_client,
        worker_manager=worker_manager,
        poll_interval=poll_interval,
        task_timeout_minutes=task_timeout_minutes,
    )

    try:
        await orchestrator.run()
    finally:
        orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
