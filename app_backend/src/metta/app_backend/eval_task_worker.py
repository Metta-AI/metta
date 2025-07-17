#!/usr/bin/env -S uv run
"""
Runs eval tasks inside a Docker container.

- Checks out the specified git hash once at startup
- Polls the backend for tasks assigned to this worker
- Processes tasks one at a time
- Reports success/failure back
"""

import asyncio
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    TaskUpdateRequest,
)

logger = logging.getLogger(__name__)


class EvalTaskWorker:
    def __init__(self, backend_url: str, git_hash: str, assignee: str):
        self._backend_url = backend_url
        self._git_hash = git_hash
        self._assignee = assignee
        self._client = EvalTaskClient(backend_url)
        self._logger = logger
        self._poll_interval = 5.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close()

    def _setup_versioned_checkout(self) -> None:
        """Set up the versioned checkout for running sim.py."""
        self._versioned_path = f"/tmp/metta-versioned/{self._git_hash}"

        # Check if already exists
        if os.path.exists(self._versioned_path):
            self._logger.info(f"Versioned checkout already exists at {self._versioned_path}")
            return

        self._logger.info(f"Setting up versioned checkout at {self._versioned_path}")

        # Create parent directory
        os.makedirs(os.path.dirname(self._versioned_path), exist_ok=True)

        # Clone repository directly at the specific commit
        # Using --depth 1 for faster clone of just the needed commit
        result = subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                self._git_hash,
                "--depth",
                "1",
                "https://github.com/Metta-AI/metta.git",
                self._versioned_path,
            ],
            capture_output=True,
            text=True,
        )

        # If cloning by hash failed (not a branch/tag), do a full clone and checkout
        if result.returncode != 0:
            self._logger.info(f"Direct clone of hash failed, trying full clone: {result.stderr}")

            # Clone the repository
            result = subprocess.run(
                ["git", "clone", "https://github.com/Metta-AI/metta.git", self._versioned_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")

            # Checkout the specific commit
            result = subprocess.run(
                ["git", "checkout", self._git_hash],
                cwd=self._versioned_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to checkout git hash {self._git_hash}: {result.stderr}")

        self._logger.info(f"Successfully set up versioned checkout at {self._versioned_path}")

    async def _run_sim_task(
        self,
        task_id: uuid.UUID,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        detailed_task = await self._client.get_task_by_id(str(task_id))
        policy_name = detailed_task.policy_name
        if not policy_name:
            raise RuntimeError(f"Policy name not found for task {task_id}")
        cmd = [
            "uv",
            "--project",
            self._versioned_path,
            "run",
            "tools/sim.py",
            f"policy_uri=wandb://run/{policy_name}",
            f"sim={sim_suite}",
            f"eval_task_id={str(task_id)}",
        ]

        for key, value in env_overrides.items():
            cmd.append(f"env_overrides.{key}={value}")

        self._logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"sim.py failed with exit code {result.returncode}:\n{result.stderr}")

        self._logger.info(f"Simulation completed successfully: {result.stdout}")

    async def _update_task_status(
        self,
        task_id: uuid.UUID,
        status: TaskStatus,
        error_reason: str | None = None,
    ) -> None:
        try:
            update_request = TaskUpdateRequest(
                assignee=self._assignee,
                statuses={
                    task_id: TaskStatusUpdate(
                        status=status, details={"error_reason": error_reason} if error_reason else None
                    )
                },
            )
            await self._client.update_task_status(update_request)
            self._logger.info(f"Updated task {task_id} status to: {status}")
        except Exception as e:
            self._logger.error(f"Failed to update task status: {e}")
            raise

    async def run(self) -> None:
        self._logger.info(f"Starting eval worker for git hash {self._git_hash}")
        self._logger.info(f"Backend URL: {self._backend_url}")
        self._logger.info(f"Assignee: {self._assignee}")

        # Set up versioned checkout for sim.py
        try:
            self._setup_versioned_checkout()
        except Exception as e:
            self._logger.error(f"Failed to set up versioned checkout: {e}")
            sys.exit(1)

        self._logger.info(f"Worker running from main branch, sim.py will use git hash {self._git_hash}")

        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = await self._client.get_claimed_tasks(assignee=self._assignee)

                if claimed_tasks.tasks:
                    task: TaskResponse = min(claimed_tasks.tasks, key=lambda x: x.assigned_at or datetime.min)
                    self._logger.info(f"Processing task {task.id}")

                    try:
                        await self._run_sim_task(
                            task.id,
                            task.sim_suite,
                            task.attributes.get("env_overrides", {}),
                        )

                        await self._update_task_status(task.id, "done")

                    except Exception as e:
                        self._logger.error(f"Task failed: {e}", exc_info=True)
                        await self._update_task_status(task.id, "error", str(e))

                elapsed_time = (datetime.now() - loop_start_time).total_seconds()
                await asyncio.sleep(max(0, self._poll_interval - elapsed_time))

            except KeyboardInterrupt:
                self._logger.info("Worker interrupted")
                break
            except Exception as e:
                self._logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(10)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    backend_url = os.environ["BACKEND_URL"]
    git_hash = os.environ["GIT_HASH"]
    assignee = os.environ["WORKER_ASSIGNEE"]

    logger.info(f"Starting worker with backend_url={backend_url}, git_hash={git_hash}, assignee={assignee}")

    async with EvalTaskWorker(backend_url, git_hash, assignee) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
