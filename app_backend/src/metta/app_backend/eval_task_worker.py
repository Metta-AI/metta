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
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import boto3
from ddtrace.trace import tracer

from gitta import get_latest_commit
from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.metta_repo import EvalTaskRow, FinishedTaskStatus
from metta.app_backend.routes.eval_task_routes import TaskFinishRequest
from metta.common.auth.auth_config_reader_writer import observatory_auth_config
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.tool.tool import ToolResult
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import SOFTMAX_S3_BASE, SOFTMAX_S3_BUCKET
from metta.common.util.file import local_copy
from metta.common.util.git_repo import REPO_SLUG, REPO_URL
from metta.common.util.log_config import init_suppress_warnings

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    success: bool
    log_path: str | None = None
    warnings: list[str] | None = None
    error: str | None = None


class AbstractTaskExecutor(ABC):
    @abstractmethod
    async def execute_task(self, task: EvalTaskRow) -> TaskResult:
        pass


class SimTaskExecutor(AbstractTaskExecutor):
    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url
        self._temp_dir = tempfile.mkdtemp()

    def _run_cmd_from_versioned_checkout(
        self,
        cmd: list[str],
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command from the versioned checkout with a clean environment."""
        env = os.environ.copy()
        for key in ["PYTHONPATH", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"]:
            env.pop(key, None)
        env["DISABLE_RICH_LOGGING"] = "1"

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
            self._versioned_path = f"{self._temp_dir}/metta-versioned/{git_hash}"
            checkout_success_file = f"{self._versioned_path}/checkout_success"

            if os.path.exists(self._versioned_path) and os.path.exists(checkout_success_file):
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

            with open(checkout_success_file, "w") as f:
                f.write("Success")

            logger.info(f"Successfully set up versioned checkout at {self._versioned_path}")
        except Exception as e:
            logger.error(f"Failed to set up versioned checkout: {e}", exc_info=True)
            if os.path.exists(self._versioned_path):
                shutil.rmtree(self._versioned_path)
            raise

    @trace("worker.execute_task")
    async def execute_task(
        self,
        task: EvalTaskRow,
    ) -> TaskResult:
        git_hash = task.git_hash or await get_latest_commit(REPO_SLUG, branch="main")

        self._setup_versioned_checkout(git_hash)

        cmd = task.command.split(" ")

        job_result_file_path = f"job_result_file_path_{task.id}.json"
        cmd.append(f"result_file_path={os.path.abspath(job_result_file_path)}")
        cmd.append(f"eval_task_id='{task.id}'")

        # Download data file if data_uri is provided, using local_copy context manager
        if task.data_uri:
            try:
                with local_copy(task.data_uri) as local_path:
                    # Add task_data_path flag with the local file path
                    cmd.append(f"task_data_path={os.path.abspath(local_path)}")
                    logger.info(f"Added task_data_path {os.path.abspath(local_path)} to command")
                    logger.info(f"Running command: {' '.join(cmd)}")

                    result = self._run_cmd_from_versioned_checkout(cmd)

                    self._upload_logs_to_s3(str(task.id), result)

                    logger.info(f"Simulation completed with return code {result.returncode}")
                    # local_copy context manager will automatically clean up the file
            except Exception as e:
                logger.error(f"Failed to download or process data file: {e}", exc_info=True)
                return TaskResult(
                    success=False,
                    error=f"Failed to download data file from {task.data_uri}: {str(e)}",
                )
        else:
            # No data file to download, run command directly
            logger.info(f"Running command: {' '.join(cmd)}")

            result = self._run_cmd_from_versioned_checkout(cmd)

            self._upload_logs_to_s3(str(task.id), result)

            logger.info(f"Simulation completed with return code {result.returncode}")

        log_path = f"{SOFTMAX_S3_BASE}/{self._log_path(str(task.id))}"

        if result.returncode == 0:
            if os.path.exists(job_result_file_path):
                with open(job_result_file_path, "r") as f:
                    output = json.load(f)
                result = ToolResult.model_validate(output)
                os.remove(job_result_file_path)
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
                error="Job failed with return code " + str(result.returncode) + " and output: " + result.stdout,
                warnings=[result.stderr],
            )


class EvalTaskWorker:
    def __init__(
        self, client: EvalTaskClient, task_executor: AbstractTaskExecutor, assignee: str, poll_interval: float = 5.0
    ):
        self._client = client
        self._task_executor = task_executor
        self._assignee = assignee
        self._poll_interval = poll_interval

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    @trace("worker.update_status")
    async def _update_task_status(
        self,
        task_id: int,
        status: FinishedTaskStatus,
        error_reason: str | None = None,
        log_path: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._client.finish_task(
            task_id,
            TaskFinishRequest(
                task_id=task_id,
                status=status,
                log_path=log_path,
                status_details=remove_none_values(
                    {
                        "error_reason": error_reason,
                        "warnings": warnings,
                    }
                ),
            ),
        )
        logger.info(
            f"Updated task {task_id} status to: {status}" + "\n" + f"Error reason: {error_reason}"
            if error_reason
            else ""
        )

    @trace("worker.attempt_task")
    async def attempt_task(self, task: EvalTaskRow) -> None:
        logger.info(f"Processing task {task.id}")
        span = tracer.current_span()
        if span:
            span.set_tags(
                {
                    f"task.{key}": value
                    for key, value in task.model_dump(mode="json").items()
                    if key in ["id", "policy_id", "policy_uri", "policy_name", "git_hash", "user_id", "assignee"]
                }
            )
            span.resource = str(task.id)

        try:
            task_result = await self._task_executor.execute_task(task)
            status: FinishedTaskStatus = "done" if task_result.success else "error"

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
                error_reason=str(e),
            )

    async def run(self) -> None:
        logger.info("Starting eval worker")
        logger.info(f"Worker id: {self._assignee}")
        while True:
            loop_start_time = datetime.now()
            try:
                claimed_tasks = self._client.get_claimed_tasks(assignee=self._assignee)

                if claimed_tasks.tasks:
                    task: EvalTaskRow = min(claimed_tasks.tasks, key=lambda x: x.assigned_at or datetime.min)
                    await self.attempt_task(task)
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
    init_suppress_warnings()
    init_tracing()

    backend_url = os.environ["BACKEND_URL"]
    assignee = os.environ["WORKER_ASSIGNEE"]
    machine_token = os.environ["MACHINE_TOKEN"]
    observatory_auth_config.save_token(machine_token, backend_url)
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
    with EvalTaskWorker(client, task_executor, assignee) as worker:
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
