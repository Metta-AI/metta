#!/usr/bin/env -S uv run
"""
Orchestrates Docker containers to process eval tasks, one container per git hash.

This script:
1. Maintains one worker container per unique git hash
2. Pulls tasks from the backend queue and routes them to the appropriate worker
3. Dynamically creates workers for new git hashes
4. Monitors container status and reports results
"""

import asyncio
import os
import random
import socket
import string
import subprocess
import textwrap
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.util.collections import group_by
from metta.common.util.script_decorators import setup_mettagrid_logger


class WorkerInfo(BaseModel):
    """Complete worker state information."""

    git_hash: str
    container_id: str
    container_name: str
    alive: bool
    task: Optional[TaskResponse] = None

    def __str__(self) -> str:
        return (
            f"WorkerInfo(hash={self.git_hash[:8]}, id={self.container_id[:3]}, "
            f"name={self.container_name[:10]}, alive={self.alive}, task={str(self.task.id)[:8] if self.task else None})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class EvalTaskOrchestrator:
    def __init__(
        self,
        backend_url: str,
        docker_image: str = "metta/eval-worker:latest",
        poll_interval: float = 5.0,
        worker_idle_timeout: float = 600.0,
    ):
        self._backend_url = backend_url
        self._docker_image = docker_image
        self._poll_interval = poll_interval
        self._worker_idle_timeout = worker_idle_timeout
        self._logger = setup_mettagrid_logger("eval_worker_orchestrator")
        self._task_client = EvalTaskClient(backend_url)

    def generate_container_suffix(self) -> str:
        """Generate a unique suffix for container names."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{timestamp}-{random_suffix}"

    def start_worker_container(self, git_hash: str) -> WorkerInfo:
        """Start a Docker container for a specific git hash.

        Returns:
            (container_id, container_name)
        """
        suffix = self.generate_container_suffix()
        container_name = f"eval-worker-{git_hash}-{suffix}"
        worker_assignee = f"worker-{git_hash[:8]}-{socket.gethostname()}-{os.getpid()}"

        env_vars = {
            "BACKEND_URL": self._backend_url,
            "GIT_HASH": git_hash,
            "WORKER_ASSIGNEE": worker_assignee,
        }

        cmd = [
            "docker",
            "run",
            # "--rm",  # Remove container when it exits
            "-d",  # Run in detached mode
            "--name",
            container_name,
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([self._docker_image, "uv", "run", "python", "-m", "metta.app_backend.eval_task_worker"])

        self._logger.info(f"Starting worker container for git hash {git_hash}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            self._logger.info(f"Started worker container {container_name} ({container_id[:12]})")
            return WorkerInfo(
                git_hash=git_hash,
                container_id=container_id,
                container_name=container_name,
                alive=True,
                task=None,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker container: {e.stderr}")
            raise

    def cleanup_container(self, container_id: str) -> None:
        """Remove a container."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                check=False,
            )
        except Exception as e:
            self._logger.warning(f"Failed to cleanup container {container_id}: {e}")
        else:
            self._logger.info(f"Cleaned up container {container_id}")

    async def discover_alive_workers(self) -> list[WorkerInfo]:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=eval-worker-",
                    "--format",
                    "{{.ID}}\t{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            workers = []
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        container_id, container_name = parts
                        # Extract git hash from name (format: eval-worker-{git_hash}-{suffix})
                        if container_name.startswith("eval-worker-"):
                            name_parts = container_name.split("-", 3)
                            if len(name_parts) >= 3:
                                git_hash = name_parts[2]
                                workers.append(
                                    WorkerInfo(
                                        git_hash=git_hash,
                                        container_id=container_id,
                                        container_name=container_name,
                                        alive=True,
                                        task=None,
                                    )
                                )

            if workers:
                self._logger.info(f"Discovered {len(workers)} alive workers")
            return workers
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to discover workers: {e}")
            return []

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

    orchestrator = EvalTaskOrchestrator(
        backend_url=backend_url,
        docker_image=docker_image,
        poll_interval=poll_interval,
        worker_idle_timeout=worker_idle_timeout,
    )

    try:
        await orchestrator.run()
    finally:
        await orchestrator._task_client.close()


if __name__ == "__main__":
    asyncio.run(main())
