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
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
from omegaconf import DictConfig, OmegaConf

from metta.common.util.script_decorators import get_metta_logger, metta_script


@dataclass
class GitHashWorker:
    """Information about a worker container for a specific git hash."""

    git_hash: str
    container_id: str
    current_task_id: Optional[str] = None
    task_start_time: Optional[float] = None
    last_activity: float = 0


class WorkerPool:
    """Manages Docker containers for processing eval tasks, one per git hash."""

    def __init__(
        self,
        backend_url: str,
        docker_image: str = "metta/eval-worker:latest",
        poll_interval: float = 5.0,
        worker_idle_timeout: float = 600.0,  # 10 minutes
    ):
        self.backend_url = backend_url
        self.docker_image = docker_image
        self.poll_interval = poll_interval
        self.worker_idle_timeout = worker_idle_timeout
        self.workers_by_git_hash: Dict[str, GitHashWorker] = {}
        self.assignee = f"orchestrator-{socket.gethostname()}-{os.getpid()}"
        self.logger = get_metta_logger()
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_available_tasks(self, limit: int = 1) -> list:
        """Get available tasks from the backend."""
        try:
            response = await self.client.get(f"{self.backend_url}/tasks/available", params={"limit": limit})
            response.raise_for_status()
            return response.json()["tasks"]
        except Exception as e:
            self.logger.error(f"Failed to get available tasks: {e}")
            return []

    async def claim_tasks(self, task_ids: list[str]) -> list[str]:
        """Claim tasks from the backend."""
        try:
            response = await self.client.post(
                f"{self.backend_url}/tasks/claim", json={"eval_task_ids": task_ids, "assignee": self.assignee}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to claim tasks: {e}")
            return []

    async def update_task_status(self, task_id: str, status: str, error_reason: Optional[str] = None) -> None:
        """Update task status in the backend."""
        try:
            if error_reason:
                status_update = {"status": status, "error_reason": error_reason}
            else:
                status_update = status

            response = await self.client.post(
                f"{self.backend_url}/tasks/claimed/update",
                json={"assignee": self.assignee, "statuses": {task_id: status_update}},
            )
            response.raise_for_status()
            self.logger.info(f"Updated task {task_id} status to: {status}")
        except Exception as e:
            self.logger.error(f"Failed to update task status: {e}")

    def start_worker_container(self, git_hash: str) -> str:
        """Start a Docker container for a specific git hash."""
        worker_assignee = f"worker-{git_hash[:8]}-{socket.gethostname()}-{os.getpid()}"

        env_vars = {
            "BACKEND_URL": self.backend_url,
            "GIT_HASH": git_hash,
            "WORKER_ASSIGNEE": worker_assignee,
            "ORCHESTRATOR_ASSIGNEE": self.assignee,
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

        # Mount the train_dir for policy access
        cmd.extend(["-v", f"{os.path.abspath('./train_dir')}:/workspace/train_dir"])

        # Add the image and command - run a persistent worker
        cmd.extend([self.docker_image, "python", "tools/eval_task_runner.py", "--worker-mode"])

        self.logger.info(f"Starting worker container for git hash {git_hash}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            self.logger.info(f"Started worker container {container_id[:12]} for git hash {git_hash}")
            return container_id
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start worker container: {e.stderr}")
            raise

    def check_container_status(self, container_id: str) -> tuple[bool, int, Optional[str]]:
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
            self.logger.warning(f"Failed to cleanup container {container_id}: {e}")

    async def get_or_create_worker(self, git_hash: str) -> GitHashWorker:
        """Get existing worker for git hash or create a new one."""
        if git_hash in self.workers_by_git_hash:
            worker = self.workers_by_git_hash[git_hash]
            # Check if container is still running
            is_running = self.check_container_health(worker.container_id)
            if is_running:
                return worker
            else:
                # Container died, remove it
                self.logger.warning(f"Worker container for git hash {git_hash} died, creating new one")
                del self.workers_by_git_hash[git_hash]

        # Create new worker
        container_id = self.start_worker_container(git_hash)
        worker = GitHashWorker(git_hash=git_hash, container_id=container_id, last_activity=time.time())
        self.workers_by_git_hash[git_hash] = worker
        self.logger.info(f"Created new worker for git hash {git_hash}")
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

    async def notify_worker_of_task(self, worker: GitHashWorker, task: dict) -> None:
        """Notify a worker container about a new task."""
        # The worker will poll for tasks assigned to it
        # So we don't need to actively notify it
        pass

    async def check_task_completion(self, task_id: str) -> bool:
        """Check if a task has been completed."""
        try:
            # Query the backend for task status
            response = await self.client.get(f"{self.backend_url}/tasks/status/{task_id}")
            if response.status_code == 200:
                task_data = response.json()
                return task_data.get("status") in ["done", "error", "canceled"]
            else:
                # If we can't get status, assume it's still running
                return False
        except Exception:
            # On error, assume task is still running
            return False

    async def run(self) -> None:
        """Main orchestrator loop."""
        self.logger.info("Starting per-git-hash worker orchestrator")
        self.logger.info(f"Backend URL: {self.backend_url}")
        self.logger.info(f"Orchestrator assignee: {self.assignee}")
        self.logger.info(f"Worker idle timeout: {self.worker_idle_timeout}s")

        while True:
            try:
                # Get available tasks (up to 10 to see what git hashes we need)
                tasks = await self.get_available_tasks(limit=10)

                # Group tasks by git hash
                tasks_by_git_hash = {}
                for task in tasks:
                    git_hash = task["attributes"]["git_hash"]
                    if git_hash not in tasks_by_git_hash:
                        tasks_by_git_hash[git_hash] = []
                    tasks_by_git_hash[git_hash].append(task)

                # Process each git hash
                for git_hash, git_tasks in tasks_by_git_hash.items():
                    # Get or create worker for this git hash
                    worker = await self.get_or_create_worker(git_hash)

                    # Check if worker is busy
                    if worker.current_task_id is None:
                        # Worker is available, assign first task
                        task = git_tasks[0]

                        # Claim the task
                        claimed = await self.claim_tasks([task["id"]])
                        if claimed:
                            # Assign to worker
                            worker.current_task_id = task["id"]
                            worker.task_start_time = time.time()
                            worker.last_activity = time.time()

                            # Notify worker about the task (we'll implement this via a file or API)
                            await self.notify_worker_of_task(worker, task)

                            self.logger.info(f"Assigned task {task['id']} to worker for git hash {git_hash}")

                # Monitor existing workers
                for git_hash, worker in list(self.workers_by_git_hash.items()):
                    # Check if worker has a task
                    if worker.current_task_id:
                        # Check task status via backend
                        task_completed = await self.check_task_completion(worker.current_task_id)

                        if task_completed:
                            duration = time.time() - worker.task_start_time
                            self.logger.info(f"Task {worker.current_task_id} completed in {duration:.1f}s")
                            worker.current_task_id = None
                            worker.task_start_time = None
                            worker.last_activity = time.time()

                    # Check for idle timeout
                    if worker.current_task_id is None:
                        idle_time = time.time() - worker.last_activity
                        if idle_time > self.worker_idle_timeout:
                            self.logger.info(f"Worker for git hash {git_hash} idle for {idle_time:.0f}s, removing")
                            self.cleanup_container(worker.container_id)
                            del self.workers_by_git_hash[git_hash]

                # Log status
                active_count = sum(1 for w in self.workers_by_git_hash.values() if w.current_task_id)
                if self.workers_by_git_hash:
                    self.logger.debug(
                        f"Workers: {len(self.workers_by_git_hash)} total, "
                        f"{active_count} active, "
                        f"{len(self.workers_by_git_hash) - active_count} idle"
                    )

            except Exception as e:
                self.logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)


@metta_script
async def main(cfg: DictConfig) -> None:
    """Main entry point."""
    backend_url = cfg.get("backend_url", os.environ.get("BACKEND_URL", "http://localhost:8000"))
    docker_image = cfg.get("docker_image", "metta/eval-worker:latest")
    poll_interval = cfg.get("poll_interval", 5.0)
    worker_idle_timeout = cfg.get("worker_idle_timeout", float(os.environ.get("WORKER_IDLE_TIMEOUT", "600")))

    pool = WorkerPool(
        backend_url=backend_url,
        docker_image=docker_image,
        poll_interval=poll_interval,
        worker_idle_timeout=worker_idle_timeout,
    )

    try:
        await pool.run()
    finally:
        await pool.client.aclose()
        # Cleanup all workers on exit
        for worker in pool.workers_by_git_hash.values():
            pool.cleanup_container(worker.container_id)


if __name__ == "__main__":
    # Simple config for standalone execution
    config = OmegaConf.create({})
    asyncio.run(main(config))
