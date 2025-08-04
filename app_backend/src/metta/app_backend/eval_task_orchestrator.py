#!/usr/bin/env -S uv run
"""
Orchestrates containers to process eval tasks, one container per git hash.

This script:
1. Maintains one worker container per unique git hash
2. Pulls tasks from the backend queue and routes them to the appropriate worker
3. Dynamically creates workers for new git hashes
4. Monitors container status and reports results
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from ddtrace.trace import tracer
from pydantic import BaseModel

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.container_managers.factory import create_container_manager
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.container_manager import ContainerWorkerManager
from metta.app_backend.worker_managers.worker import Worker
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.collections import group_by
from metta.common.util.constants import DEV_STATS_SERVER_URI
from metta.common.util.logging_helpers import init_logging


class WorkerInfo(BaseModel):
    worker: Worker
    git_hashes: list[str] = []
    assigned_task: TaskResponse | None = None


class EvalTaskOrchestrator:
    def __init__(
        self,
        task_client: EvalTaskClient,
        worker_manager: AbstractWorkerManager,
        poll_interval: float = 5.0,
        worker_idle_timeout: float = 1200.0,
        max_workers: int = 5,
        logger: logging.Logger | None = None,
    ):
        self._task_client = task_client
        self._worker_manager = worker_manager
        self._poll_interval = poll_interval
        self._worker_idle_timeout = worker_idle_timeout
        self._max_workers = max_workers
        self._logger = logger or logging.getLogger(__name__)

    @trace("orchestrator.claim_task")
    async def _attempt_claim_task(self, task: TaskResponse, worker: WorkerInfo) -> bool:
        claim_request = TaskClaimRequest(tasks=[task.id], assignee=worker.worker.name)
        claimed_ids = await self._task_client.claim_tasks(claim_request)
        if task.id in claimed_ids.claimed:
            self._logger.info(f"Assigned task {task.id} to worker {worker.worker.name}")
            return True
        else:
            self._logger.debug("Failed to claim task; someone else must have it")
            return False

    async def _get_available_workers(self, claimed_tasks: list[TaskResponse]) -> dict[str, WorkerInfo]:
        alive_workers = await self._worker_manager.discover_alive_workers()

        worker_names = [w.name for w in alive_workers]
        git_hashes_by_assignee = await self._task_client.get_git_hashes_for_workers(worker_names)
        alive_workers_by_name: dict[str, WorkerInfo] = {w.name: WorkerInfo(worker=w) for w in alive_workers}

        for task in claimed_tasks:
            if task.assignee and (worker := alive_workers_by_name.get(task.assignee)):
                worker.assigned_task = task

        for assignee, git_hashes in git_hashes_by_assignee.git_hashes.items():
            if assignee in alive_workers_by_name:
                alive_workers_by_name[assignee].git_hashes = git_hashes

        return alive_workers_by_name

    async def _kill_dead_workers_and_tasks(
        self, claimed_tasks: list[TaskResponse], alive_workers_by_name: dict[str, WorkerInfo]
    ) -> None:
        for task in claimed_tasks:
            if task.assignee and task.assignee not in alive_workers_by_name:
                reason = "worker_dead"
            elif task.assigned_at and task.assigned_at < datetime.now(timezone.utc) - timedelta(minutes=10):
                reason = "worker_timeout"
            else:
                continue

            if task.retries < 3:
                status = "unprocessed"
            else:
                status = "error"

            self._logger.info(f"Unclaiming task {task.id} because {reason}. Setting status to {status}")
            await self._task_client.update_task_status(
                TaskUpdateRequest(
                    updates={
                        task.id: TaskStatusUpdate(
                            status=status, clear_assignee=True, attributes={f"unassign_reason_{task.retries}": reason}
                        )
                    }
                )
            )

            if task.assignee and (worker := alive_workers_by_name.get(task.assignee)):
                self._logger.info(f"Killing worker {task.assignee} because it has been working too long")
                self._worker_manager.cleanup_worker(worker.worker.name)
                del alive_workers_by_name[worker.worker.name]

    async def _kill_idle_workers(self, alive_workers_by_name: dict[str, WorkerInfo]) -> None:
        for worker in alive_workers_by_name.values():
            if worker.assigned_task:
                continue

            latest_task = await self._task_client.get_latest_assigned_task_for_worker(worker.worker.name)
            last_task_assigned_at = datetime.min
            if latest_task and latest_task.assigned_at:
                last_task_assigned_at = latest_task.assigned_at.replace(tzinfo=timezone.utc)
            idle_duration = (datetime.now(timezone.utc) - last_task_assigned_at).total_seconds()

            if idle_duration > self._worker_idle_timeout:
                self._logger.info(f"Killing idle worker {worker.worker.name}")
                self._worker_manager.cleanup_worker(worker.worker.name)
                del alive_workers_by_name[worker.worker.name]

    async def _assign_task_to_worker(
        self, worker: WorkerInfo, available_tasks_by_git_hash: dict[str | None, list[TaskResponse]]
    ) -> None:
        # Assign a task to a worker, prioritizing its existing git hashes
        for git_hash in worker.git_hashes:
            if git_hash in available_tasks_by_git_hash:
                tasks = available_tasks_by_git_hash[git_hash]
                if tasks:
                    task = tasks.pop(0)
                    await self._attempt_claim_task(task, worker)
                    return
                else:
                    available_tasks_by_git_hash.pop(git_hash)

        # If no tasks are available for the worker's git hashes, assign a task from the remaining tasks
        for _, tasks in available_tasks_by_git_hash.items():
            if tasks:
                task = tasks.pop(0)
                await self._attempt_claim_task(task, worker)
                return

    async def _assign_tasks_to_workers(self, alive_workers_by_name: dict[str, WorkerInfo]) -> None:
        available_tasks = await self._task_client.get_available_tasks()
        available_tasks_by_git_hash: dict[str | None, list[TaskResponse]] = group_by(
            available_tasks.tasks, key_fn=lambda t: t.git_hash
        )
        for worker in alive_workers_by_name.values():
            # Only assign tasks to workers that are running and don't have an assigned task
            if not worker.assigned_task and worker.worker.status == "Running":
                await self._assign_task_to_worker(worker, available_tasks_by_git_hash)

    async def _start_new_workers(self, alive_workers_by_name: dict[str, WorkerInfo]) -> None:
        # Todo: start workers on demand.  For now just start fixed number of workers
        for _ in range(self._max_workers - len(alive_workers_by_name)):
            # Reuse the orchestrator's client for workers to maintain test setup
            # Note: worker manager implementations should store necessary tokens/config
            self._worker_manager.start_worker()

    @trace("orchestrator.run_cycle")
    async def run_cycle(self) -> None:
        claimed_tasks = await self._task_client.get_claimed_tasks()
        alive_workers_by_name = await self._get_available_workers(claimed_tasks.tasks)

        await self._kill_dead_workers_and_tasks(claimed_tasks.tasks, alive_workers_by_name)

        await self._assign_tasks_to_workers(alive_workers_by_name)

        await self._kill_idle_workers(alive_workers_by_name)

        await self._start_new_workers(alive_workers_by_name)

    async def run(self) -> None:
        self._logger.info(f"Backend URL: {getattr(self._task_client, '_base_url', 'unknown')}")
        self._logger.info(f"Worker idle timeout: {self._worker_idle_timeout}s")

        with tracer.trace("orchestrator.startup"):
            self._logger.info("Orchestrator startup trace")

        while True:
            start_time = datetime.now(timezone.utc)
            try:
                await self.run_cycle()
            except Exception as e:
                self._logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            sleep_time = max(0, self._poll_interval - elapsed_time)
            await asyncio.sleep(sleep_time)


async def main() -> None:
    init_logging()
    init_tracing()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    backend_url = os.environ.get("BACKEND_URL", DEV_STATS_SERVER_URI)
    docker_image = os.environ.get("DOCKER_IMAGE", "metta-policy-evaluator-local:latest")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    worker_idle_timeout = float(os.environ.get("WORKER_IDLE_TIMEOUT", "1200"))
    max_workers = int(os.environ.get("MAX_WORKERS", "5"))
    machine_token = os.environ["MACHINE_TOKEN"]

    task_client = EvalTaskClient(backend_url)
    container_manager = create_container_manager()
    worker_manager = ContainerWorkerManager(
        container_manager=container_manager,
        backend_url=backend_url,
        docker_image=docker_image,
        machine_token=machine_token,
        logger=logger,
    )
    orchestrator = EvalTaskOrchestrator(
        task_client=task_client,
        worker_manager=worker_manager,
        poll_interval=poll_interval,
        worker_idle_timeout=worker_idle_timeout,
        max_workers=max_workers,
        logger=logger,
    )

    try:
        await orchestrator.run()
    finally:
        await orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
