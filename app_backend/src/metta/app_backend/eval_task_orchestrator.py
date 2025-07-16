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
import socket
import subprocess
import uuid
from datetime import datetime

from pydantic import BaseModel

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
)
from metta.common.util.collections import group_by
from metta.common.util.script_decorators import setup_mettagrid_logger


class AssignedTask(BaseModel):
    task_id: uuid.UUID
    start_time: datetime


class GitHashWorker(BaseModel):
    git_hash: str
    container_id: str
    last_activity: datetime
    assigned_task: AssignedTask | None = None


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
        self._workers_by_git_hash: dict[str, GitHashWorker] = {}
        self._assignee = f"orchestrator-{socket.gethostname()}-{os.getpid()}"
        self._logger = setup_mettagrid_logger("eval_worker_orchestrator")
        self._task_client = EvalTaskClient(backend_url)

    def start_worker_container(self, git_hash: str) -> str:
        """Start a Docker container for a specific git hash."""
        worker_assignee = f"worker-{git_hash[:8]}-{socket.gethostname()}-{os.getpid()}"

        env_vars = {
            "BACKEND_URL": self._backend_url,
            "GIT_HASH": git_hash,
            "WORKER_ASSIGNEE": worker_assignee,
            "ORCHESTRATOR_ASSIGNEE": self._assignee,
        }

        cmd = [
            "docker",
            "run",
            "--rm",  # Remove container when it exits
            "-d",  # Run in detached mode
            "--name",
            f"eval-worker-{git_hash[:8]}",
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Mount the train_dir for policy access. This is relevant locally
        # When running in Docker, TRAIN_DIR_HOST_PATH should be set to the host path
        train_dir_path = os.environ.get("TRAIN_DIR_HOST_PATH", os.path.abspath("./train_dir"))
        cmd.extend(["-v", f"{train_dir_path}:/workspace/train_dir"])

        cmd.extend([self._docker_image, "uv", "run", "tools/eval_task_runner.py"])

        self._logger.info(f"Starting worker container for git hash {git_hash}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            self._logger.info(f"Started worker container {container_id[:12]} for git hash {git_hash}")
            return container_id
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to start worker container: {e.stderr}")
            raise

    def check_container_status(self, container_id: str) -> tuple[bool, int, str | None]:
        """Check if a container has finished and get its exit code.

        Returns:
            (finished, exit_code, error_message)
        """
        try:
            # Check if container is still running
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip() == "true":
                return False, 0, None

            # Container has stopped, get exit code
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            exit_code = int(result.stdout.strip())

            # Get logs if failed
            error_message = None
            if exit_code != 0:
                result = subprocess.run(
                    ["docker", "logs", "--tail", "50", container_id], capture_output=True, text=True
                )
                error_message = f"Container exited with code {exit_code}:\n{result.stderr}"

            return True, exit_code, error_message

        except subprocess.CalledProcessError as e:
            # Container might have been removed
            return True, -1, f"Failed to inspect container: {e}"

    def cleanup_container(self, container_id: str) -> None:
        """Remove a container."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                check=False,  # Don't fail if container doesn't exist
            )
        except Exception as e:
            self._logger.warning(f"Failed to cleanup container {container_id}: {e}")

    async def get_or_create_worker(self, git_hash: str) -> GitHashWorker:
        """Get existing worker for git hash or create a new one."""
        if git_hash in self._workers_by_git_hash:
            worker = self._workers_by_git_hash[git_hash]
            # Check if container is still running
            is_running = self.check_container_health(worker.container_id)
            if is_running:
                return worker
            else:
                # Container died, remove it
                self._logger.warning(f"Worker container for git hash {git_hash} died, creating new one")
                del self._workers_by_git_hash[git_hash]

        # Create new worker
        container_id = self.start_worker_container(git_hash)
        worker = GitHashWorker(git_hash=git_hash, container_id=container_id, last_activity=datetime.now())
        self._workers_by_git_hash[git_hash] = worker
        self._logger.info(f"Created new worker for git hash {git_hash}")
        return worker

    def check_container_health(self, container_id: str) -> bool:
        """Check if a container is healthy and running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError:
            return False

    async def check_task_completion(self, task_id: uuid.UUID) -> bool:
        """Check if a task has been completed."""
        try:
            task = await self._task_client.get_task_by_id(str(task_id))
        except Exception as e:
            self._logger.warning(f"Failed to check task completion for {task_id}: {e}")
            # If we can't check the task, assume it's still running
            return False
        else:
            return task.status in ["done", "error", "canceled"]

    async def run(self) -> None:
        self._logger.info("Starting per-git-hash worker orchestrator")
        self._logger.info(f"Backend URL: {self._backend_url}")
        self._logger.info(f"Orchestrator assignee: {self._assignee}")
        self._logger.info(f"Worker idle timeout: {self._worker_idle_timeout}s")

        while True:
            start_time = datetime.now()
            try:
                response = await self._task_client.get_available_tasks(limit=10)
                tasks_by_git_hash = group_by(response.tasks, lambda task: task.attributes["git_hash"])

                # 1. Claim tasks for workers that don't have a task
                for git_hash, git_tasks in tasks_by_git_hash.items():
                    worker = await self.get_or_create_worker(git_hash)
                    if worker.assigned_task is None:
                        task = git_tasks[0]
                        claim_request = TaskClaimRequest(tasks=[task.id], assignee=self._assignee)
                        claimed_ids = await self._task_client.claim_tasks(claim_request)
                        if task.id in claimed_ids.claimed:
                            worker.assigned_task = AssignedTask(task_id=task.id, start_time=datetime.now())
                            worker.last_activity = datetime.now()
                            self._logger.info(f"Assigned task {task.id} to worker for git hash {git_hash}")
                        else:
                            self._logger.debug(f"Task {task.id} not claimed by worker for git hash {git_hash}")
                    else:
                        self._logger.debug(
                            f"Worker for git hash {git_hash} already has a task. Not scheduling {git_tasks}"
                        )

                # 2. Check for completed tasks
                for git_hash, worker in list(self._workers_by_git_hash.items()):
                    if worker.assigned_task:
                        task_completed = await self.check_task_completion(worker.assigned_task.task_id)
                        if task_completed:
                            duration = (datetime.now() - worker.assigned_task.start_time).total_seconds()
                            self._logger.info(f"Task {worker.assigned_task.task_id} completed in {duration:.1f}s")
                            worker.assigned_task = None
                            worker.last_activity = datetime.now()

                    # Check for idle timeout
                    if worker.assigned_task is None:
                        idle_time = (datetime.now() - worker.last_activity).total_seconds()
                        if idle_time > self._worker_idle_timeout:
                            self._logger.info(f"Worker for git hash {git_hash} idle for {idle_time:.0f}s, removing")
                            self.cleanup_container(worker.container_id)
                            del self._workers_by_git_hash[git_hash]

                # Log status
                active_count = sum(1 for w in self._workers_by_git_hash.values() if w.assigned_task)
                if self._workers_by_git_hash:
                    self._logger.debug(
                        f"Workers: {len(self._workers_by_git_hash)} total, "
                        f"{active_count} active, "
                        f"{len(self._workers_by_git_hash) - active_count} idle"
                    )

            except Exception as e:
                self._logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, self._poll_interval - elapsed_time)
            await asyncio.sleep(sleep_time)


async def main() -> None:
    setup_mettagrid_logger("eval_worker_orchestrator")

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    docker_image = "metta/eval-worker:latest"
    poll_interval = 5.0
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
        # Cleanup all workers on exit
        for worker in orchestrator._workers_by_git_hash.values():
            orchestrator.cleanup_container(worker.container_id)


if __name__ == "__main__":
    asyncio.run(main())
