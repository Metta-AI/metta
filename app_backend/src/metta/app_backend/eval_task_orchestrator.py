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
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from ddtrace.trace import tracer
from pydantic import BaseModel

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.container_managers.factory import create_container_manager
from metta.app_backend.routes.eval_task_routes import (
    EvalTaskResponse,
    TaskClaimRequest,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.container_manager import ContainerWorkerManager
from metta.app_backend.worker_managers.worker import Worker
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.collections import group_by
from metta.common.util.constants import DEV_STATS_SERVER_URI
from metta.common.util.log_config import init_logging

logger = logging.getLogger(__name__)


class WorkerInfo(BaseModel):
    worker: Worker
    git_hashes: list[str] = []
    assigned_task: EvalTaskResponse | None = None


class AbstractWorkerScaler(ABC):
    @abstractmethod
    async def get_desired_workers(self, num_workers: int) -> int:
        pass


class FixedScaler(AbstractWorkerScaler):
    def __init__(self, num_workers: int):
        self._num_workers = num_workers

    async def get_desired_workers(self, num_workers: int) -> int:
        return self._num_workers


class AutoScaler(AbstractWorkerScaler):
    CREATED_IN_LAST_DAY_FILTER = "created_at > NOW() - INTERVAL '1 day'"
    DONE_FILTER = "status = 'done'"
    UNPROCESSED_FILTER = "status = 'unprocessed' OR status = 'running'"

    def __init__(self, task_client: EvalTaskClient, default_task_runtime_seconds: float):
        self._task_client = task_client
        self._default_task_runtime_seconds = default_task_runtime_seconds

    async def _compute_desired_workers(self, avg_task_runtime: float) -> int:
        num_tasks_per_day = (await self._task_client.count_tasks(self.CREATED_IN_LAST_DAY_FILTER)).count
        total_work_time_seconds = num_tasks_per_day * avg_task_runtime
        single_worker_work_time_seconds = 60 * 60 * 24
        return math.ceil(total_work_time_seconds / single_worker_work_time_seconds * 1.2)  # 20% buffer

    async def _get_avg_task_runtime(self) -> float:
        avg_task_runtime = self._default_task_runtime_seconds
        num_done_tasks_last_day = (
            await self._task_client.count_tasks(f"{self.DONE_FILTER} AND {self.CREATED_IN_LAST_DAY_FILTER}")
        ).count
        if num_done_tasks_last_day > 20:
            avg_runtime_last_day = (
                await self._task_client.get_avg_runtime(f"{self.DONE_FILTER} AND {self.CREATED_IN_LAST_DAY_FILTER}")
            ).avg_runtime
            if avg_runtime_last_day is not None:
                avg_task_runtime = avg_runtime_last_day
        return avg_task_runtime

    async def get_desired_workers(self, num_workers: int) -> int:
        num_active_tasks = (await self._task_client.count_tasks(self.UNPROCESSED_FILTER)).count

        avg_task_runtime = await self._get_avg_task_runtime()
        num_desired_workers = await self._compute_desired_workers(avg_task_runtime)

        return max(num_desired_workers, math.ceil(num_active_tasks * avg_task_runtime / 3600))


class EvalTaskOrchestrator:
    def __init__(
        self,
        task_client: EvalTaskClient,
        worker_manager: AbstractWorkerManager,
        worker_scaler: AbstractWorkerScaler,
        poll_interval: float = 5.0,
        worker_idle_timeout_minutes: float = 60.0,
    ):
        self._task_client = task_client
        self._worker_manager = worker_manager
        self._worker_scaler = worker_scaler
        self._poll_interval = poll_interval
        self._worker_idle_timeout_minutes = worker_idle_timeout_minutes

    @trace("orchestrator.claim_task")
    async def _attempt_claim_task(self, task: EvalTaskResponse, worker: WorkerInfo) -> bool:
        claim_request = TaskClaimRequest(tasks=[task.id], assignee=worker.worker.name)
        claimed_ids = await self._task_client.claim_tasks(claim_request)
        if task.id in claimed_ids.claimed:
            logger.info(f"Assigned task {task.id} to worker {worker.worker.name}")
            return True
        else:
            logger.debug("Failed to claim task; someone else must have it")
            return False

    async def _get_available_workers(self, claimed_tasks: list[EvalTaskResponse]) -> dict[str, WorkerInfo]:
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
        self, claimed_tasks: list[EvalTaskResponse], alive_workers_by_name: dict[str, WorkerInfo]
    ) -> None:
        try:
            for task in claimed_tasks:
                if task.assignee and task.assignee not in alive_workers_by_name:
                    reason = "worker_dead"
                elif task.assigned_at and task.assigned_at.replace(tzinfo=timezone.utc) < (
                    datetime.now(timezone.utc) - timedelta(minutes=self._worker_idle_timeout_minutes)
                ):
                    reason = "worker_timeout"
                else:
                    continue

                status = "error"

                logger.info(f"Releasing claim on task {task.id} because {reason}. Setting status to {status}")
                await self._task_client.update_task_status(
                    TaskUpdateRequest(
                        updates={
                            task.id: TaskStatusUpdate(
                                status=status,
                                clear_assignee=True,
                                attributes={f"unassign_reason_{task.retries}": reason},
                            )
                        }
                    )
                )

                if task.assignee and (worker := alive_workers_by_name.get(task.assignee)):
                    logger.info(f"Killing worker {task.assignee} because it has been working too long")
                    self._worker_manager.cleanup_worker(worker.worker.name)
                    del alive_workers_by_name[worker.worker.name]
        except Exception as e:
            logger.error(f"Error killing dead workers and tasks: {e}", exc_info=True)

    async def _assign_task_to_worker(
        self, worker: WorkerInfo, available_tasks_by_git_hash: dict[str | None, list[EvalTaskResponse]]
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
        available_tasks_by_git_hash: dict[str | None, list[EvalTaskResponse]] = group_by(
            available_tasks.tasks, key_fn=lambda t: t.git_hash
        )
        for worker in alive_workers_by_name.values():
            # Only assign tasks to workers that are running and don't have an assigned task
            if not worker.assigned_task and worker.worker.status == "Running":
                await self._assign_task_to_worker(worker, available_tasks_by_git_hash)

    async def _scale_workers(self, alive_workers_by_name: dict[str, WorkerInfo]) -> None:
        desired_workers = await self._worker_scaler.get_desired_workers(len(alive_workers_by_name))
        if desired_workers > len(alive_workers_by_name):
            logger.info(f"Launching {desired_workers - len(alive_workers_by_name)} extra workers")
            for _ in range(desired_workers - len(alive_workers_by_name)):
                self._worker_manager.start_worker()
        elif desired_workers < len(alive_workers_by_name):
            # If we have too many workers, kill some idle workers
            idle_workers = [
                w for w in alive_workers_by_name.values() if not w.assigned_task and w.worker.status == "Running"
            ]
            if idle_workers:
                num_workers_to_kill = len(alive_workers_by_name) - desired_workers
                logger.info(f"Killing {num_workers_to_kill} idle workers")
                for worker in idle_workers[:num_workers_to_kill]:
                    self._worker_manager.cleanup_worker(worker.worker.name)
                    del alive_workers_by_name[worker.worker.name]

    @trace("orchestrator.run_cycle")
    async def run_cycle(self) -> None:
        claimed_tasks = await self._task_client.get_claimed_tasks()
        alive_workers_by_name = await self._get_available_workers(claimed_tasks.tasks)

        await self._kill_dead_workers_and_tasks(claimed_tasks.tasks, alive_workers_by_name)

        await self._assign_tasks_to_workers(alive_workers_by_name)

        await self._scale_workers(alive_workers_by_name)

    async def run(self) -> None:
        logger.info(f"Backend URL: {getattr(self._task_client, '_base_url', 'unknown')}")
        logger.info(f"Worker idle timeout: {self._worker_idle_timeout_minutes} minutes")

        with tracer.trace("orchestrator.startup"):
            logger.info("Orchestrator startup trace")

        while True:
            start_time = datetime.now(timezone.utc)
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            sleep_time = max(0, self._poll_interval - elapsed_time)
            await asyncio.sleep(sleep_time)


async def main() -> None:
    init_logging()
    init_tracing()

    backend_url = os.environ.get("BACKEND_URL", DEV_STATS_SERVER_URI)
    docker_image = os.environ.get("DOCKER_IMAGE", "metta-policy-evaluator-local:latest")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    worker_idle_timeout_minutes = float(os.environ.get("WORKER_IDLE_TIMEOUT", "60"))
    machine_token = os.environ["MACHINE_TOKEN"]

    task_client = EvalTaskClient(backend_url)
    container_manager = create_container_manager()
    worker_manager = ContainerWorkerManager(
        container_manager=container_manager,
        backend_url=backend_url,
        docker_image=docker_image,
        machine_token=machine_token,
    )
    worker_scaler = AutoScaler(task_client, 1200.0)
    orchestrator = EvalTaskOrchestrator(
        task_client=task_client,
        worker_manager=worker_manager,
        worker_scaler=worker_scaler,
        poll_interval=poll_interval,
        worker_idle_timeout_minutes=worker_idle_timeout_minutes,
    )

    try:
        await orchestrator.run()
    finally:
        await orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
