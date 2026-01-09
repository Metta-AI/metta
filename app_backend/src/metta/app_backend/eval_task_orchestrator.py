#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402
"""
Orchestrates containers to process eval tasks, one worker per task.

This script:
1. Creates one worker container per task
2. Assigns the task to the worker immediately upon creation
3. Kills workers when their task is complete or timed out
4. Monitors container status and reports results
"""

from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from ddtrace.trace import tracer

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.container_managers.factory import create_container_manager
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskFinishRequest,
)
from metta.app_backend.worker_managers.base import AbstractWorkerManager
from metta.app_backend.worker_managers.container_manager import ContainerWorkerManager
from metta.app_backend.worker_managers.worker import Worker
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.constants import DEV_STATS_SERVER_URI

logger = logging.getLogger(__name__)


class EvalTaskOrchestrator:
    def __init__(
        self,
        task_client: EvalTaskClient,
        worker_manager: AbstractWorkerManager,
        poll_interval: float = 5.0,
        task_timeout_minutes: float = 60.0,
    ):
        self._task_client = task_client
        self._worker_manager = worker_manager
        self._poll_interval = poll_interval
        self._task_timeout_minutes = task_timeout_minutes

    def _is_task_timed_out(self, task: EvalTaskRow) -> bool:
        if not task.assigned_at:
            return False
        assigned_at_utc = task.assigned_at.replace(tzinfo=timezone.utc)
        timeout_threshold = datetime.now(timezone.utc) - timedelta(minutes=self._task_timeout_minutes)
        return assigned_at_utc < timeout_threshold

    def _fail_task(self, task: EvalTaskRow, reason: str) -> None:
        logger.info(f"Failing task {task.id}: {reason}")
        self._task_client.finish_task(
            task.id,
            TaskFinishRequest(
                task_id=task.id,
                status="system_error",
                status_details={"unassign_reason": reason},
            ),
        )

    def _cleanup_stale_tasks(self, claimed_tasks: list[EvalTaskRow], alive_worker_names: set[str]) -> None:
        """Fail tasks that are timed out or whose worker is dead."""
        for task in claimed_tasks:
            worker_is_dead = task.assignee and task.assignee not in alive_worker_names

            if self._is_task_timed_out(task):
                self._fail_task(task, "task_timeout")
                if task.assignee and task.assignee in alive_worker_names:
                    logger.info(f"Killing worker {task.assignee} due to task timeout")
                    self._worker_manager.cleanup_worker(task.assignee)
                    alive_worker_names.discard(task.assignee)
            elif worker_is_dead:
                self._fail_task(task, "worker_dead")

    def _cleanup_idle_workers(self, alive_workers: list[Worker], assigned_worker_names: set[str]) -> None:
        """Kill workers that have no assigned task."""
        for worker in alive_workers:
            if worker.name not in assigned_worker_names:
                logger.info(f"Killing idle worker {worker.name}")
                self._worker_manager.cleanup_worker(worker.name)

    def _compute_num_cpus(self, parallelism: int) -> int:
        """Compute the number of CPUs to request for a given parallelism. Our nodes have 4, 8, 12, or 16 CPUs.
        Kubernetes nodes have "allocatable" resources that are less than total capacity because kubelet,
        os-level daemons (sytemd), etc end up consuming some of the total capacity.
        So if we want to allocate on a machine with 4 vpcpus in total, we need to request an allocation of fewer
        than 4. Thus we request 3, 7, 11, or 15.
        """
        if parallelism <= 4:
            return 3
        elif parallelism <= 8:
            return 7
        elif parallelism <= 12:
            return 11
        else:
            return 15

    def _compute_memory_request(self, parallelism: int) -> int:
        """Compute the memory to request for a given parallelism. We request 3GB per parallel process."""
        parallelism = max(1, parallelism)
        parallelism = min(16, parallelism)
        return parallelism * 3

    def _spawn_workers_for_tasks(self) -> None:
        """Create one worker per unassigned task and claim the task for that worker."""
        available_tasks = self._task_client.get_available_tasks()

        for task in available_tasks.tasks:
            parallelism = task.attributes.get("parallelism", 1)
            num_cpus = self._compute_num_cpus(parallelism)
            memory_request = self._compute_memory_request(parallelism)

            worker_name = self._worker_manager.start_worker(num_cpus_request=num_cpus, memory_request=memory_request)
            logger.info(f"Started worker {worker_name} for task {task.id}")

            claim_request = TaskClaimRequest(tasks=[task.id], assignee=worker_name)
            claimed_ids = self._task_client.claim_tasks(claim_request)
            if task.id in claimed_ids.claimed:
                logger.info(f"Assigned task {task.id} to worker {worker_name}")
            else:
                logger.warning(f"Failed to claim task {task.id} for worker {worker_name}; killing worker")
                self._worker_manager.cleanup_worker(worker_name)

    @trace("orchestrator.run_cycle")
    async def run_cycle(self) -> None:
        claimed_tasks = self._task_client.get_claimed_tasks().tasks
        alive_workers = await self._worker_manager.discover_alive_workers()
        alive_worker_names = {w.name for w in alive_workers}

        # Fail stale tasks (timed out or worker dead)
        self._cleanup_stale_tasks(claimed_tasks, alive_worker_names)

        # Kill workers that have no assigned task
        assigned_worker_names = {t.assignee for t in claimed_tasks if t.assignee}
        self._cleanup_idle_workers(alive_workers, assigned_worker_names)

        # Spawn one worker per available task
        self._spawn_workers_for_tasks()

    async def run(self) -> None:
        logger.info(f"Backend URL: {getattr(self._task_client, '_base_url', 'unknown')}")
        logger.info(f"Task timeout: {self._task_timeout_minutes} minutes")

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


def init_logging():
    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()  # Remove any existing handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main() -> None:
    init_logging()
    init_tracing()

    backend_url = os.environ.get("BACKEND_URL", DEV_STATS_SERVER_URI)
    docker_image = os.environ.get("DOCKER_IMAGE", "metta-policy-evaluator-local:latest")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    task_timeout_minutes = float(os.environ.get("TASK_TIMEOUT_MINUTES", "90"))
    machine_token = os.environ["MACHINE_TOKEN"]

    task_client = EvalTaskClient(backend_url)
    container_manager = create_container_manager()
    worker_manager = ContainerWorkerManager(
        container_manager=container_manager,
        backend_url=backend_url,
        docker_image=docker_image,
        machine_token=machine_token,
    )
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
