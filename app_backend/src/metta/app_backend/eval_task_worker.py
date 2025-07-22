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
import uuid
from datetime import datetime

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.util.collections import remove_none_values
from metta.common.util.logging_helpers import init_logging


class EvalTaskWorker:
    def __init__(self, backend_url: str, git_hash: str, assignee: str, logger: logging.Logger | None = None):
        self._backend_url = backend_url
        self._git_hash = git_hash
        self._assignee = assignee
        self._client = EvalTaskClient(backend_url)
        self._logger = logger or logging.getLogger(__name__)
        self._poll_interval = 5.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close()

    def _setup_versioned_checkout(self) -> None:
        self._versioned_path = f"/tmp/metta-versioned/{self._git_hash}"
        if os.path.exists(self._versioned_path):
            self._logger.info(f"Versioned checkout already exists at {self._versioned_path}")
            return

        self._logger.info(f"Setting up versioned checkout at {self._versioned_path}")

        os.makedirs(os.path.dirname(self._versioned_path), exist_ok=True)

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
        task: TaskResponse,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        policy_name = task.policy_name
        if not policy_name:
            raise RuntimeError(f"Policy name not found for task {task.id}")
        cmd = [
            "uv",
            "--project",
            self._versioned_path,
            "run",
            "tools/sim.py",
            f"policy_uri=wandb://run/{policy_name}",
            f"sim={sim_suite}",
            f"eval_task_id={str(task.id)}",
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
        await self._client.update_task_status(
            TaskUpdateRequest(
                require_assignee=self._assignee,
                updates={
                    task_id: TaskStatusUpdate(
                        status=status,
                        attributes=remove_none_values({f"error_reason_{self._assignee}": error_reason}),
                    )
                },
            )
        )
        self._logger.info(
            f"Updated task {task_id} status to: {status}" + "\n" + f"Error reason: {error_reason}"
            if error_reason
            else ""
        )

    async def run(self) -> None:
        self._logger.info(f"Starting eval worker for git hash {self._git_hash}")
        self._logger.info(f"Backend URL: {self._backend_url}")
        self._logger.info(f"Worker id: {self._assignee}")

        self._setup_versioned_checkout()

        self._logger.info(f"Worker running from main branch, sim.py will use git hash {self._git_hash}")

        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = await self._client.get_claimed_tasks(assignee=self._assignee)

                if claimed_tasks.tasks:
                    task: TaskResponse = min(claimed_tasks.tasks, key=lambda x: x.assigned_at or datetime.min)
                    self._logger.info(f"Processing task {task.id}")
                    try:
                        await self._run_sim_task(task, task.sim_suite, task.attributes.get("env_overrides", {}))
                        self._logger.info(f"Task {task.id} completed successfully")
                        await self._update_task_status(task.id, "done")
                        self._logger.info(f"Task {task.id} updated to done")
                    except Exception as e:
                        self._logger.error(f"Task failed: {e}", exc_info=True)
                        await self._update_task_status(task.id, "error", str(e))
                else:
                    self._logger.debug("No tasks claimed")

                elapsed_time = (datetime.now() - loop_start_time).total_seconds()
                await asyncio.sleep(max(0, self._poll_interval - elapsed_time))

            except KeyboardInterrupt:
                self._logger.info("Worker interrupted")
                break
            except Exception as e:
                self._logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(10)


async def main() -> None:
    init_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    backend_url = os.environ["BACKEND_URL"]
    git_hash = os.environ["GIT_HASH"]
    assignee = os.environ["WORKER_ASSIGNEE"]

    async with EvalTaskWorker(backend_url, git_hash, assignee, logger) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
