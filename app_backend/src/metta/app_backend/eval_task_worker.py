#!/usr/bin/env -S uv run
"""
Runs eval tasks inside a Docker container.

- Checks out the specified git hash once at startup
- Polls the backend for tasks assigned to this worker
- Processes tasks one at a time
- Reports success/failure back
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import boto3

from devops.observatory_login import CLIAuthenticator
from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import (
    EvalTaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    TaskUpdateRequest,
)
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import SOFTMAX_S3_BASE, SOFTMAX_S3_BUCKET
from metta.common.util.git_repo import REPO_URL
from metta.rl.checkpoint_manager import CheckpointManager
from metta.tools.remote_job import JobResult

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    success: bool
    log_path: str | None = None
    warnings: list[str] | None = None
    error: str | None = None


class AbstractTaskExecutor(ABC):
    @abstractmethod
    async def execute_task(self, task: EvalTaskResponse) -> TaskResult:
        pass


class SimTaskExecutor(AbstractTaskExecutor):
    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url

    def _run_cmd_from_versioned_checkout(
        self,
        cmd: list[str],
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command from the versioned checkout with a clean environment."""
        env = os.environ.copy()
        for key in ["PYTHONPATH", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"]:
            env.pop(key, None)

        if capture_output:
            # Redirect stderr to stdout to get chronologically interspersed output (like 2>&1)
            result = subprocess.run(
                cmd,
                cwd=self._versioned_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        else:
            result = subprocess.run(
                cmd,
                cwd=self._versioned_path,
                text=True,
                env=env,
            )

        return result

    def _log_path(self, job_id: str) -> str:
        return f"jobs/{job_id}/output.log"

    def _upload_logs_to_s3(self, job_id: str, process: subprocess.CompletedProcess) -> None:
        logger.info(f"Uploading logs to S3: {job_id}")
        logger.info(f"Combined output: {process.stdout}")

        # stdout contains both stderr and stdout interspersed (due to stderr=STDOUT redirect)
        log_content = process.stdout or ""

        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=SOFTMAX_S3_BUCKET,
            Key=self._log_path(job_id),
            Body=log_content,
            ContentType="text/plain",
        )

    @trace("worker.setup_checkout")
    def _setup_versioned_checkout(self, git_hash: str) -> None:
        try:
            self._versioned_path = f"/tmp/metta-versioned/{git_hash}"
            if os.path.exists(self._versioned_path):
                logger.info(f"Versioned checkout already exists at {self._versioned_path}")
                return

            logger.info(f"Setting up versioned checkout at {self._versioned_path}")

            os.makedirs(os.path.dirname(self._versioned_path), exist_ok=True)

            result = subprocess.run(
                ["git", "clone", REPO_URL, self._versioned_path],
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
            logger.info("Installing dependencies in versioned checkout...")
            self._run_cmd_from_versioned_checkout(
                ["uv", "run", "metta", "configure", "--profile=softmax-docker"],
                capture_output=True,
            )
            self._run_cmd_from_versioned_checkout(
                ["uv", "run", "metta", "install"],
            )

            logger.info(f"Successfully set up versioned checkout at {self._versioned_path}")
        except Exception as e:
            logger.error(f"Failed to set up versioned checkout: {e}", exc_info=True)
            if os.path.exists(self._versioned_path):
                shutil.rmtree(self._versioned_path)
            raise

    @trace("worker.execute_task")
    async def execute_task(
        self,
        task: EvalTaskResponse,
    ) -> TaskResult:
        if not task.git_hash:
            raise RuntimeError(f"Git hash not found for task {task.id}")

        self._setup_versioned_checkout(task.git_hash)

        # Convert simulations list to a base64-encoded JSON string to avoid parsing issues
        simulations = task.attributes.get("simulations", [])
        simulations_json = json.dumps(simulations)
        simulations_base64 = base64.b64encode(simulations_json.encode()).decode()

        # write simulations_json_base64 to a file
        file_path = f"simulations_json_base64_{task.id}.json"
        with open(file_path, "w") as f:
            f.write(simulations_base64)

        normalized = CheckpointManager.normalize_uri(task.policy_uri)

        job_result_file_path = f"job_result_file_path_{task.id}.json"

        cmd = [
            "uv",
            "run",
            "tools/run.py",
            "experiments.evals.run.eval",
            f"policy_uri={normalized}",
            f"simulations_json_base64_path={os.path.abspath(file_path)}",
            f"eval_task_id={str(task.id)}",
            f"stats_server_uri={self._backend_url}",
            f"job_result_file_path={os.path.abspath(job_result_file_path)}",
            "push_metrics_to_wandb=true",
        ]
        # exclude simulation_json_base64 from logging, since it's too large and undescriptive
        logged_cmd = [arg for arg in cmd if not arg.startswith("simulations_json_base64")]
        logger.info(f"Running command: {' '.join(logged_cmd)}")

        result = self._run_cmd_from_versioned_checkout(cmd)

        self._upload_logs_to_s3(str(task.id), result)

        logger.info(f"Simulation completed with return code {result.returncode}")

        log_path = f"{SOFTMAX_S3_BASE}/{self._log_path(str(task.id))}"

        if result.returncode == 0:
            if os.path.exists(job_result_file_path):
                with open(job_result_file_path, "r") as f:
                    output = json.load(f)
                result = JobResult.model_validate(output)
                return TaskResult(
                    success=result.result == "success",
                    warnings=result.warnings,
                    error=result.error,
                    log_path=log_path,
                )
            else:
                return TaskResult(
                    success=False,
                    log_path=log_path,
                    error="Job result file not found",
                )
        else:
            return TaskResult(
                success=False,
                log_path=log_path,
                error="Job failed with return code " + str(result.returncode),
            )


class EvalTaskWorker:
    def __init__(
        self, client: EvalTaskClient, task_executor: AbstractTaskExecutor, assignee: str, poll_interval: float = 5.0
    ):
        self._client = client
        self._task_executor = task_executor
        self._assignee = assignee
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
        log_path: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        await self._client.update_task_status(
            TaskUpdateRequest(
                require_assignee=self._assignee,
                updates={
                    task_id: TaskStatusUpdate(
                        status=status,
                        attributes=remove_none_values(
                            {
                                f"error_reason_{self._assignee}": error_reason,
                                "output_log_path": log_path,
                                "warnings": warnings,
                            }
                        ),
                    )
                },
            )
        )
        logger.info(
            f"Updated task {task_id} status to: {status}" + "\n" + f"Error reason: {error_reason}"
            if error_reason
            else ""
        )

    async def run(self) -> None:
        logger.info("Starting eval worker")
        logger.info(f"Worker id: {self._assignee}")
        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = await self._client.get_claimed_tasks(assignee=self._assignee)

                if claimed_tasks.tasks:
                    task: EvalTaskResponse = min(claimed_tasks.tasks, key=lambda x: x.assigned_at or datetime.min)
                    logger.info(f"Processing task {task.id}")
                    try:
                        task_result = await self._task_executor.execute_task(task)
                        status = "done" if task_result.success else "error"

                        logger.info(f"Task {task.id} completed with status {status}")
                        warnings = None
                        if task_result.warnings is not None and len(task_result.warnings) > 0:
                            warnings = task_result.warnings

                        await self._update_task_status(
                            task.id,
                            status,
                            error_reason=task_result.error,
                            log_path=task_result.log_path,
                            warnings=warnings,
                        )
                        logger.info(f"Task {task.id} updated to {status}")
                    except Exception as e:
                        logger.error(f"Task failed: {e}", exc_info=True)
                        await self._update_task_status(
                            task.id,
                            "error",
                            str(e),
                        )
                else:
                    logger.debug("No tasks claimed")

                elapsed_time = (datetime.now() - loop_start_time).total_seconds()
                await asyncio.sleep(max(0, self._poll_interval - elapsed_time))

            except KeyboardInterrupt:
                logger.info("Worker interrupted")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(10)


def init_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main() -> None:
    init_logging()
    init_tracing()

    backend_url = os.environ["BACKEND_URL"]
    assignee = os.environ["WORKER_ASSIGNEE"]
    machine_token = os.environ["MACHINE_TOKEN"]
    CLIAuthenticator().save_token(machine_token, backend_url)
    client = EvalTaskClient(backend_url)
    task_executor = SimTaskExecutor(backend_url)
    logger.info(
        "Running with: "
        + "\n".join(
            f"{k}={v}"
            for k, v in {
                "backend_url": backend_url,
                "assignee": assignee,
            }.items()
        )
    )
    async with EvalTaskWorker(client, task_executor, assignee) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
