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
import traceback
import uuid
from datetime import datetime

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskStatus,
    TaskStatusUpdate,
    TaskUpdateRequest,
)

logger = logging.getLogger(__name__)


class EvalWorker:
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

    def _checkout_git_hash(self) -> None:
        self._logger.info(f"Checking out git hash: {self._git_hash}")
        result = subprocess.run(
            ["git", "checkout", self._git_hash],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout git hash {self._git_hash}: {result.stderr}")
        self._logger.info(f"Successfully checked out git hash: {self._git_hash}")

    def _run_sim_task(
        self,
        policy_id: str,
        sim_suite: str,
        eval_task_id: str,
        env_overrides: dict,
    ) -> None:
        if policy_id.startswith("wandb://") or policy_id.startswith("file://"):
            policy_uri = policy_id
        else:
            policy_uri = f"wandb://metta-research/metta/{policy_id}:latest"

        cmd = [
            "uv",
            "run",
            "tools/sim.py",
            f"policy_uri={policy_uri}",
            f"sim={sim_suite}",
            f"eval_task_id={eval_task_id}",
            f"run=eval_task_{eval_task_id[:8]}",
        ]

        for key, value in env_overrides.items():
            cmd.append(f"env_overrides.{key}={value}")

        self._logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"sim.py failed with exit code {result.returncode}:\n{result.stderr}")

        self._logger.info("Simulation completed successfully")

    async def _get_claimed_tasks(self) -> list:
        try:
            response = await self._client.get_claimed_tasks(assignee=self._assignee)
            return response.tasks
        except Exception as e:
            self._logger.warning(f"Failed to get claimed tasks: {e}")
            return []

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

        try:
            self._checkout_git_hash()
        except Exception as e:
            self._logger.error(f"Failed to checkout git hash: {e}")
            sys.exit(1)

        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = await self._get_claimed_tasks()

                if claimed_tasks:
                    task = claimed_tasks[0]
                    self._logger.info(f"Processing task {task.id}")

                    try:
                        self._run_sim_task(
                            str(task.policy_id),
                            task.sim_suite,
                            str(task.id),
                            task.attributes.get("env_overrides", {}),
                        )

                        await self._update_task_status(task.id, "done")

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                        self._logger.error(f"Task failed: {error_msg}")

                        await self._update_task_status(task.id, "error", error_msg)

                elapsed_time = (datetime.now() - loop_start_time).total_seconds()
                await asyncio.sleep(max(0, self._poll_interval - elapsed_time))

            except KeyboardInterrupt:
                self._logger.info("Worker interrupted")
                break
            except Exception as e:
                self._logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(10)


async def main() -> None:
    backend_url = os.environ["BACKEND_URL"]
    git_hash = os.environ["GIT_HASH"]
    assignee = os.environ["WORKER_ASSIGNEE"]

    async with EvalWorker(backend_url, git_hash, assignee) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
