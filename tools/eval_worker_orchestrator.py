#!/usr/bin/env -S uv run
"""
Orchestrates a pool of Docker containers to process eval tasks.

This script:
1. Maintains a pool of worker containers
2. Pulls tasks from the backend queue one at a time
3. Dispatches each task to an available container
4. Monitors container status and reports results
"""

import asyncio
import json
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
class ContainerTask:
    """Information about a task running in a container."""

    task_id: str
    container_id: str
    start_time: float


class WorkerPool:
    """Manages a pool of Docker containers for processing eval tasks."""

    def __init__(
        self,
        backend_url: str,
        pool_size: int = 5,
        docker_image: str = "metta/eval-worker:latest",
        poll_interval: float = 5.0,
    ):
        self.backend_url = backend_url
        self.pool_size = pool_size
        self.docker_image = docker_image
        self.poll_interval = poll_interval
        self.active_containers: Dict[str, ContainerTask] = {}
        self.assignee = f"worker-{socket.gethostname()}-{os.getpid()}"
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

    def start_container(self, task: dict) -> str:
        """Start a Docker container for a task."""
        env_vars = {
            "BACKEND_URL": self.backend_url,
            "EVAL_TASK_ID": task["id"],
            "POLICY_ID": task["policy_id"],
            "SIM_SUITE": task["sim_suite"],
            "GIT_HASH": task["attributes"]["git_hash"],
            "ASSIGNEE": self.assignee,
            "ENV_OVERRIDES": json.dumps(task["attributes"].get("env_overrides", {})),
        }

        cmd = [
            "docker",
            "run",
            "--rm",  # Remove container when it exits
            "-d",  # Run in detached mode
            "--name",
            f"eval-task-{task['id'][:8]}",
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Mount the train_dir for policy access
        cmd.extend(["-v", f"{os.path.abspath('./train_dir')}:/workspace/train_dir"])

        # Add the image and command
        cmd.extend([self.docker_image, "python tools/eval_task_runner.py"])

        self.logger.info(f"Starting container for task {task['id']}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            self.logger.info(f"Started container {container_id[:12]} for task {task['id']}")
            return container_id
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container: {e.stderr}")
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

    async def run(self) -> None:
        """Main orchestrator loop."""
        self.logger.info(f"Starting worker pool with {self.pool_size} slots")
        self.logger.info(f"Backend URL: {self.backend_url}")
        self.logger.info(f"Assignee: {self.assignee}")

        while True:
            try:
                # Check for available container slots
                if len(self.active_containers) < self.pool_size:
                    # Pull one task from queue
                    tasks = await self.get_available_tasks(limit=1)
                    if tasks:
                        task = tasks[0]
                        # Claim it
                        claimed = await self.claim_tasks([task["id"]])
                        if claimed:
                            # Dispatch to new container
                            try:
                                container_id = self.start_container(task)
                                self.active_containers[container_id] = ContainerTask(
                                    task_id=task["id"], container_id=container_id, start_time=time.time()
                                )
                            except Exception as e:
                                # Failed to start container, update task status
                                await self.update_task_status(task["id"], "error", f"Failed to start container: {e}")

                # Monitor running containers
                for container_id, task_info in list(self.active_containers.items()):
                    finished, exit_code, error_message = self.check_container_status(container_id)

                    if finished:
                        if exit_code == 0:
                            # Success - status should already be updated by the container
                            self.logger.info(f"Task {task_info.task_id} completed successfully")
                        else:
                            # Failure - update status if not already done
                            self.logger.error(f"Task {task_info.task_id} failed: {error_message}")
                            if exit_code == -1:
                                # Container inspection failed, update status
                                await self.update_task_status(
                                    task_info.task_id, "error", error_message or "Container failed"
                                )

                        # Cleanup
                        self.cleanup_container(container_id)
                        del self.active_containers[container_id]

                        duration = time.time() - task_info.start_time
                        self.logger.info(f"Task {task_info.task_id} took {duration:.1f} seconds")

                # Log pool status periodically
                if len(self.active_containers) > 0:
                    self.logger.debug(f"Active containers: {len(self.active_containers)}/{self.pool_size}")

            except Exception as e:
                self.logger.error(f"Error in orchestrator loop: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)


@metta_script
async def main(cfg: DictConfig) -> None:
    """Main entry point."""
    backend_url = cfg.get("backend_url", os.environ.get("BACKEND_URL", "http://localhost:8000"))
    pool_size = cfg.get("pool_size", int(os.environ.get("WORKER_POOL_SIZE", "5")))
    docker_image = cfg.get("docker_image", "metta/eval-worker:latest")
    poll_interval = cfg.get("poll_interval", 5.0)

    pool = WorkerPool(
        backend_url=backend_url, pool_size=pool_size, docker_image=docker_image, poll_interval=poll_interval
    )

    try:
        await pool.run()
    finally:
        await pool.client.aclose()


if __name__ == "__main__":
    # Simple config for standalone execution
    config = OmegaConf.create({})
    asyncio.run(main(config))
