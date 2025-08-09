#!/usr/bin/env -S uv run
"""
Runs eval tasks inside a Docker container.

- Checks out the specified git hash once at startup
- Polls the backend for tasks assigned to this worker
- Processes tasks one at a time
- Reports success/failure back
"""

import asyncio
import json
import logging
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from devops.observatory_login import CLIAuthenticator
from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.collections import remove_none_values
from metta.common.util.git import METTA_API_REPO_URL
from metta.common.util.logging_helpers import init_logging


class AbstractTaskExecutor(ABC):
    @abstractmethod
    async def execute_task(self, task: TaskResponse) -> None:
        pass


class SimTaskExecutor(AbstractTaskExecutor):
    def __init__(self, backend_url: str, machine_token: str, logger: logging.Logger):
        self._backend_url = backend_url
        CLIAuthenticator(backend_url).save_token(machine_token)
        self._logger = logger
        self._logger.info(f"Backend URL: {self._backend_url}")

    def _run_cmd_from_versioned_checkout(
        self,
        cmd: list[str],
        error_msg: str,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command from the versioned checkout with a clean environment."""
        env = os.environ.copy()
        for key in ["PYTHONPATH", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"]:
            env.pop(key, None)

        result = subprocess.run(
            cmd,
            cwd=self._versioned_path,
            capture_output=capture_output,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"{error_msg}: {result.stderr}")

        return result

    @trace("worker.setup_checkout")
    def _setup_versioned_checkout(self, git_hash: str) -> None:
        self._versioned_path = f"/tmp/metta-versioned/{git_hash}"
        if os.path.exists(self._versioned_path):
            self._logger.info(f"Versioned checkout already exists at {self._versioned_path}")
            return

        self._logger.info(f"Setting up versioned checkout at {self._versioned_path}")

        os.makedirs(os.path.dirname(self._versioned_path), exist_ok=True)

        result = subprocess.run(
            ["git", "clone", METTA_API_REPO_URL, self._versioned_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        # Checkout the specific commit
        result = subprocess.run(
            ["git", "checkout", git_hash],
            cwd=self._versioned_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout git hash {git_hash}: {result.stderr}")

        # Install dependencies in the versioned checkout
        self._logger.info("Installing dependencies in versioned checkout...")
        self._run_cmd_from_versioned_checkout(
            ["uv", "run", "metta", "configure", "--profile=softmax-docker"],
            "Failed to configure dependencies",
        )
        self._run_cmd_from_versioned_checkout(
            ["uv", "run", "metta", "install"],
            "Failed to install dependencies",
        )

        self._logger.info(f"Successfully set up versioned checkout at {self._versioned_path}")

    @trace("worker.run_sim_task")
    async def _run_sim_task(
        self,
        task: TaskResponse,
        sim_suite: str,
        sim_suite_config: dict | None,
    ) -> None:
        if not task.git_hash:
            raise RuntimeError(f"Git hash not found for task {task.id}")

        self._setup_versioned_checkout(task.git_hash)

        policy_name = task.policy_name
        if not policy_name:
            raise RuntimeError(f"Policy name not found for task {task.id}")

        sim_suite_config_str = json.dumps(sim_suite_config) if sim_suite_config else None

        cmd = [
            "uv",
            "run",
            "tools/sim.py",
            f"policy_uri=wandb://run/{policy_name}",
            f"sim={sim_suite}",
            f"eval_task_id={str(task.id)}",
            f"stats_server_uri={self._backend_url}",
            "device=cpu",
            "vectorization=serial",
            "push_metrics_to_wandb=true",
        ]
        if sim_suite_config_str:
            cmd.append(f"sim_suite_config={sim_suite_config_str}")

        self._logger.info(f"Running command: {' '.join(cmd)}")

        result = self._run_cmd_from_versioned_checkout(
            cmd,
            "sim.py failed with exit code",
        )

        self._logger.info(f"Simulation completed successfully: {result.stdout}")

    async def execute_task(self, task: TaskResponse) -> None:
        await self._run_sim_task(task, task.sim_suite, task.sim_suite_config)


class EvalTaskWorker:
    def __init__(
        self,
        client: EvalTaskClient,
        task_executor: AbstractTaskExecutor,
        assignee: str,
        poll_interval: float = 5.0,
        logger: logging.Logger | None = None,
    ):
        self._client = client
        self._task_executor = task_executor
        self._assignee = assignee

        self._logger = logger or logging.getLogger(__name__)
        self._poll_interval = poll_interval

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.close()

    @trace("worker.update_status")
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
        self._logger.info("Starting eval worker")
        self._logger.info(f"Worker id: {self._assignee}")

        self._logger.info("Worker running from main branch, sim.py will use git hash")

        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = await self._client.get_claimed_tasks(assignee=self._assignee)

                if claimed_tasks.tasks:
                    task: TaskResponse = min(claimed_tasks.tasks, key=lambda x: x.assigned_at or datetime.min)
                    self._logger.info(f"Processing task {task.id}")
                    try:
                        await self._task_executor.execute_task(task)
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
    init_tracing()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    backend_url = os.environ["BACKEND_URL"]
    assignee = os.environ["WORKER_ASSIGNEE"]
    machine_token = os.environ["MACHINE_TOKEN"]

    client = EvalTaskClient(backend_url)
    task_executor = SimTaskExecutor(backend_url, machine_token, logger)
    async with EvalTaskWorker(client, task_executor, assignee, logger=logger) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
