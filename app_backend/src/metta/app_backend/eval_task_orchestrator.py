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
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.factory import create_container_manager
from metta.app_backend.container_managers.models import WorkerInfo
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.util.collections import group_by
from metta.common.util.constants import DEV_STATS_SERVER_URI
from metta.common.util.logging_helpers import init_logging


class EvalTaskOrchestrator:
    def __init__(
        self,
        backend_url: str,
        machine_token: str,
        docker_image: str = "metta-policy-evaluator-local:latest",
        poll_interval: float = 5.0,
        worker_idle_timeout: float = 1200.0,
        max_workers_per_git_hash: int = 5,
        container_manager: AbstractContainerManager | None = None,
        logger: logging.Logger | None = None,
    ):
        self._backend_url = backend_url
        self._docker_image = docker_image
        self._poll_interval = poll_interval
        self._worker_idle_timeout = worker_idle_timeout
        self._max_workers_per_git_hash = max_workers_per_git_hash
        self._machine_token = machine_token
        self._logger = logger or logging.getLogger(__name__)
        self._task_client = EvalTaskClient(backend_url)
        self._container_manager = container_manager or create_container_manager()

    async def _attempt_claim_task(self, task: TaskResponse, worker: WorkerInfo) -> bool:
        claim_request = TaskClaimRequest(tasks=[task.id], assignee=worker.container_name)
        claimed_ids = await self._task_client.claim_tasks(claim_request)
        if task.id in claimed_ids.claimed:
            self._logger.info(f"Assigned task {task.id} to worker {worker.container_name}")
            return True
        else:
            self._logger.debug("Failed to claim task; someone else must have it")
            return False

    async def run_cycle(self) -> None:
        alive_workers_by_name: dict[str, WorkerInfo] = {
            w.container_name: w for w in await self._container_manager.discover_alive_workers()
        }
        worker_assignments: defaultdict[str, list[TaskResponse]] = defaultdict(list)
        claimed_tasks = await self._task_client.get_claimed_tasks()

        for task in claimed_tasks.tasks:
            # Unclaim all assigned tasks that are overdue and kill their workers
            assigned_at = task.assigned_at.replace(tzinfo=timezone.utc) if task.assigned_at else None
            if assigned_at and (assigned_at < datetime.now(timezone.utc) - timedelta(minutes=10)):
                self._logger.info(f"Killing task {task.id} because it has been running for more than 10 minutes")
                if task.retries < 3:
                    await self._task_client.update_task_status(
                        TaskUpdateRequest(
                            updates={
                                task.id: TaskStatusUpdate(
                                    status="unprocessed",
                                    clear_assignee=True,
                                    attributes={f"unassign_reason_{task.retries}": "worker_timeout"},
                                )
                            },
                        )
                    )
                else:
                    await self._task_client.update_task_status(
                        TaskUpdateRequest(
                            updates={
                                task.id: TaskStatusUpdate(status="error", attributes={"reason": "max_retries_exceeded"})
                            },
                        )
                    )
                    self._logger.info(f"Not retrying task {task.id} because it has exceeded max retries")
                if task.assignee and alive_workers_by_name.get(task.assignee):
                    self._logger.info(f"Killing worker {task.assignee} because it has been working too long")
                    worker = alive_workers_by_name[task.assignee]
                    self._container_manager.cleanup_container(worker.container_id)
                    del alive_workers_by_name[worker.container_name]
            elif task.assignee and (assigned_worker := alive_workers_by_name.get(task.assignee)):
                self._logger.debug(f"Worker {task.assignee} is still working on task {task.id}")
                worker_assignments[assigned_worker.container_name].append(task)

        # Assign tasks to existing workers and spawn new workers
        available_tasks = await self._task_client.get_available_tasks()
        alive_workers_by_git_hash = group_by(list(alive_workers_by_name.values()), key_fn=lambda w: w.git_hash)
        for task in available_tasks.tasks:
            if not task.git_hash:
                await self._task_client.update_task_status(
                    TaskUpdateRequest(
                        updates={task.id: TaskStatusUpdate(status="error", attributes={"reason": "no_git_hash"})},
                    )
                )
                continue
            # (a) Ensure we have available workers for this git hash
            existing_workers = alive_workers_by_git_hash[task.git_hash]
            available_workers = [w for w in existing_workers if not len(worker_assignments[w.container_name])]

            # If no available workers, try to spawn a new one
            if not available_workers and len(existing_workers) < self._max_workers_per_git_hash:
                self._logger.info(
                    f"All {len(existing_workers)} workers for git hash {task.git_hash} are busy, spawning new worker"
                )
                new_worker = self._container_manager.start_worker_container(
                    git_hash=task.git_hash,
                    backend_url=self._backend_url,
                    docker_image=self._docker_image,
                    machine_token=self._machine_token,
                )
                alive_workers_by_name[new_worker.container_name] = new_worker
                alive_workers_by_git_hash[task.git_hash].append(new_worker)
                available_workers = [new_worker]

            # (b) If still no available workers, we're at capacity
            if not available_workers:
                self._logger.info(
                    f"Workers for git hash {task.git_hash} are all busy "
                    f"({len(existing_workers)}/{self._max_workers_per_git_hash} max), "
                    f"skipping assigning {task.id}"
                )
                continue

            # (c) Assign task to first available worker
            for worker in available_workers:
                if await self._attempt_claim_task(task, worker):
                    worker_assignments[worker.container_name].append(task)
                    break

        # Clean up idle workers
        alive_unassigned_workers = [
            w for w in alive_workers_by_name.values() if w.container_name not in worker_assignments
        ]
        for worker in alive_unassigned_workers:
            try:
                latest_task = await self._task_client.get_latest_assigned_task_for_worker(worker.container_name)
                last_task_assigned_at = (
                    latest_task.assigned_at.replace(tzinfo=timezone.utc)
                    if latest_task and latest_task.assigned_at
                    else datetime.min
                )
                last_assigned_age = (datetime.now(timezone.utc) - last_task_assigned_at).total_seconds()
                if last_assigned_age > self._worker_idle_timeout:
                    self._logger.info(f"Cleaning up {worker.container_name} - no tasks for {last_assigned_age:.0f}s")
                    self._container_manager.cleanup_container(worker.container_id)
                    del alive_workers_by_name[worker.container_name]
            except Exception:
                self._logger.error(f"Failed to check idle status for worker {worker.container_name}", exc_info=True)

    async def run(self) -> None:
        self._logger.info(f"Backend URL: {self._backend_url}")
        self._logger.info(f"Worker idle timeout: {self._worker_idle_timeout}s")
        self._logger.info(f"Max workers per git hash: {self._max_workers_per_git_hash}")

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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    backend_url = os.environ.get("BACKEND_URL", DEV_STATS_SERVER_URI)
    docker_image = os.environ.get("DOCKER_IMAGE", "metta-policy-evaluator-local:latest")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    worker_idle_timeout = float(os.environ.get("WORKER_IDLE_TIMEOUT", "1200"))
    max_workers_per_git_hash = int(os.environ.get("MAX_WORKERS_PER_GIT_HASH", "5"))
    machine_token = os.environ["MACHINE_TOKEN"]

    orchestrator = EvalTaskOrchestrator(
        backend_url=backend_url,
        machine_token=machine_token,
        docker_image=docker_image,
        poll_interval=poll_interval,
        worker_idle_timeout=worker_idle_timeout,
        max_workers_per_git_hash=max_workers_per_git_hash,
        logger=logger,
    )

    try:
        await orchestrator.run()
    finally:
        await orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
