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
import os
import textwrap
from datetime import datetime, timezone

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.factory import create_container_manager
from metta.app_backend.container_managers.models import WorkerInfo
from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.util.collections import group_by
from metta.common.util.script_decorators import setup_mettagrid_logger


class EvalTaskOrchestrator:
    def __init__(
        self,
        backend_url: str,
        docker_image: str = "metta/eval-worker:latest",
        poll_interval: float = 5.0,
        worker_idle_timeout: float = 600.0,
        container_manager: AbstractContainerManager | None = None,
    ):
        self._backend_url = backend_url
        self._docker_image = docker_image
        self._poll_interval = poll_interval
        self._worker_idle_timeout = worker_idle_timeout
        self._logger = setup_mettagrid_logger("eval_worker_orchestrator")
        self._task_client = EvalTaskClient(backend_url)

        # Use provided container manager or create one based on environment
        self._container_manager = container_manager or create_container_manager()

    def start_worker_container(self, git_hash: str) -> WorkerInfo:
        """Start a worker container for a specific git hash."""
        return self._container_manager.start_worker_container(
            git_hash=git_hash, backend_url=self._backend_url, docker_image=self._docker_image
        )

    def cleanup_container(self, container_id: str) -> None:
        """Remove a container."""
        self._container_manager.cleanup_container(container_id)

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        """Discover all alive workers."""
        return await self._container_manager.discover_alive_workers()

    async def update_worker_assignments(self, workers: dict[str, WorkerInfo]) -> None:
        """
        Assign only the earliest task for each git hash.
        If a worker is assigned a task but is not alive, mark it as not alive.
        Else, update the task for the worker.
        """
        claimed_tasks = await self._task_client.get_claimed_tasks()
        for git_hash, tasks in group_by(claimed_tasks.tasks, key_fn=lambda t: t.git_hash).items():
            if not git_hash:
                continue
            earliest_task = min(tasks, key=lambda t: t.created_at)
            if git_hash not in workers:
                # Worker is believed to exist but is not alive
                workers[git_hash] = WorkerInfo(
                    git_hash=git_hash, container_id="", container_name="", alive=False, task=earliest_task
                )
            else:
                # Worker is alive - update the task
                workers[git_hash].task = earliest_task

    async def _attempt_claim_task(self, task: TaskResponse, worker: WorkerInfo) -> bool:
        claim_request = TaskClaimRequest(tasks=[task.id], assignee=worker.container_name)
        claimed_ids = await self._task_client.claim_tasks(claim_request)
        if task.id in claimed_ids.claimed:
            worker.task = task
            self._logger.info(f"Assigned task {task.id} to worker {worker.container_name}")
            return True
        else:
            self._logger.debug("Failed to claim task; someone else must have it")
            return False

    async def process_unassigned_tasks(self, workers: dict[str, WorkerInfo]) -> None:
        response = await self._task_client.get_available_tasks(limit=50)
        claimed_but_dead_worker_tasks = [w.task for w in workers.values() if w.task and not w.alive]

        assignable = sorted(response.tasks + claimed_but_dead_worker_tasks, key=lambda t: t.created_at)
        if not assignable:
            self._logger.info("No assignable tasks found")
            return
        self._logger.info(f"Assigning {len(assignable)} tasks...")
        for task in assignable:
            if not task.git_hash:
                continue
            existing_worker = workers.get(task.git_hash)

            # Case 1: There is an alive worker that has no assignment. Claim it
            if existing_worker and existing_worker.alive and not existing_worker.task:
                await self._attempt_claim_task(task, existing_worker)
            # Case 2: There is an alive worker that has an assignment. Skip
            elif existing_worker and existing_worker.alive and existing_worker.task:
                self._logger.debug(
                    f"Task {task.id} for git hash {task.git_hash} can't be assigned - "
                    f"worker {existing_worker.container_name} is busy with task {existing_worker.task.id}"
                )
            # Case 3: There is an assigned worker but it isn't alive, and the task has been retried too many times
            elif existing_worker and task.workers_spawned > 3:
                assert not existing_worker.alive
                assert task.assignee
                # Too many errors - mark as error
                update_request = TaskUpdateRequest(
                    assignee=task.assignee,
                    statuses={
                        task.id: TaskStatusUpdate(status="error", details={"reason": "max_error_count_exceeded"})
                    },
                )
                await self._task_client.update_task_status(update_request)
            # Case 4: There is no assigned worker, or the assigned worker is dead and it should be revived
            else:
                new_worker = self.start_worker_container(task.git_hash)
                if await self._attempt_claim_task(task, new_worker):
                    workers[task.git_hash] = new_worker
                else:
                    # new_worker will be cleaned up next cycle
                    pass

    async def cleanup_idle_workers(self, workers: dict[str, WorkerInfo]) -> None:
        for git_hash, worker in workers.items():
            if worker.alive and not worker.task:
                try:
                    # Get latest assigned task for this git hash
                    latest_task = await self._task_client.get_latest_assigned_task_for_git_hash(git_hash)

                    if latest_task and latest_task.assigned_at:
                        # Check if worker has been idle too long
                        task_age = (
                            datetime.now(timezone.utc) - latest_task.assigned_at.replace(tzinfo=timezone.utc)
                        ).total_seconds()
                        if task_age > self._worker_idle_timeout:
                            self._logger.info(
                                f"Cleaning up idle worker {worker.container_name} - no tasks for {task_age:.0f}s"
                            )
                            self.cleanup_container(worker.container_id)
                            worker.alive = False
                except Exception as e:
                    self._logger.error(f"Failed to check idle status for worker {worker.container_name}: {e}")

    async def remove_duplicate_workers(self, state: list[WorkerInfo]) -> dict[str, WorkerInfo]:
        grouped = group_by(state, key_fn=lambda w: w.git_hash)
        to_remove = []
        remaining_workers = {}
        for git_hash, workers in grouped.items():
            if len(workers) == 1:
                remaining_workers[git_hash] = workers[0]
                continue
            # Keep the newest worker
            sorted_workers = sorted(workers, key=lambda w: w.container_name, reverse=True)
            remaining_workers[git_hash] = sorted_workers[0]
            to_remove.extend(sorted_workers[1:])
        for worker in to_remove:
            self.cleanup_container(worker.container_id)
        if to_remove:
            self._logger.info(f"Removed {len(to_remove)} duplicate workers")
        return remaining_workers

    async def run_cycle(self) -> None:
        # Discover alive workers
        alive_workers: list[WorkerInfo] = await self.discover_alive_workers()

        # Remove duplicate workers
        workers = await self.remove_duplicate_workers(alive_workers)

        # Update assignments from backend
        await self.update_worker_assignments(workers)

        self._logger.info(f"Current state: {list(workers.values())}")

        # Process unassigned tasks and handle dead workers
        await self.process_unassigned_tasks(workers)

        # Clean up idle workers
        await self.cleanup_idle_workers(workers)

        # Log status
        self.report_status(workers)

    def report_status(self, workers: dict[str, WorkerInfo]) -> None:
        grouped = group_by(list(workers.values()), key_fn=lambda w: (w.alive, w.task is not None))
        self._logger.debug(
            textwrap.dedent(
                f"""Workers:
                total: {len(workers)}
                alive: {len(grouped[(True, False)])}
                assigned: {len(grouped[(True, True)])}
                idle: {len(grouped[(False, True)])}
                dead: {len(grouped[(False, False)])}"""
            )
        )

    async def run(self) -> None:
        self._logger.info("Starting stateless eval task orchestrator")
        self._logger.info(f"Backend URL: {self._backend_url}")
        self._logger.info(f"Worker idle timeout: {self._worker_idle_timeout}s")

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
    setup_mettagrid_logger("eval_worker_orchestrator")

    # Suppress httpx INFO logs
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    docker_image = os.environ.get("DOCKER_IMAGE", "metta-local:latest")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "5"))
    worker_idle_timeout = float(os.environ.get("WORKER_IDLE_TIMEOUT", "600"))

    # Create the appropriate container manager based on environment
    container_manager = create_container_manager()

    orchestrator = EvalTaskOrchestrator(
        backend_url=backend_url,
        docker_image=docker_image,
        poll_interval=poll_interval,
        worker_idle_timeout=worker_idle_timeout,
        container_manager=container_manager,
    )

    try:
        await orchestrator.run()
    finally:
        await orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
