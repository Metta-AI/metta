"""Job runner for local and remote execution.

Provides a unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with support for both async and sync execution.

Example usage:
    # Local job (sync)
    job = LocalJob(name="test", cmd=["pytest", "tests/"], timeout_s=900)
    result = job.wait(stream_output=True)
    print(f"Exit code: {result.exit_code}")

    # Remote job (sync)
    job = RemoteJob(
        name="train",
        task_yaml="sky_tasks/arena_train.yaml",
        timeout_s=3600,
    )
    result = job.wait(stream_output=True)
    print(f"Job ID: {result.job_id}, Exit code: {result.exit_code}")

    # Remote job (async - poll for completion)
    job = RemoteJob(name="train", task_yaml="sky_tasks/arena_train.yaml")
    job.submit()
    while not job.is_complete():
        time.sleep(10)
    result = job.get_result()

    # Attach to existing remote job
    job = RemoteJob(name="train", task_yaml="sky_tasks/arena_train.yaml", job_id=12345)
    result = job.wait(stream_output=True)
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import sky

from metta.common.util.fs import get_repo_root


@dataclass
class JobResult:
    """Result of a completed job execution."""

    name: str
    exit_code: int
    logs_path: str
    job_id: Optional[str] = None
    duration_s: Optional[float] = None

    def get_logs(self) -> str:
        """Read logs from file."""
        if self.logs_path and Path(self.logs_path).exists():
            return Path(self.logs_path).read_text(errors="ignore")
        return ""

    @property
    def success(self) -> bool:
        """Whether job completed successfully."""
        return self.exit_code == 0


class Job(ABC):
    """Abstract base class for job execution."""

    def __init__(self, name: str, log_dir: str, timeout_s: int = 3600):
        self.name = name
        self.log_dir = Path(log_dir)
        self.timeout_s = timeout_s
        self._submitted = False
        self._result: Optional[JobResult] = None

    @abstractmethod
    def submit(self) -> None:
        """Submit job for execution (async - returns immediately)."""
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if job has completed."""
        pass

    @abstractmethod
    def get_logs(self) -> str:
        """Get current logs (may be partial if job still running)."""
        pass

    def get_result(self) -> Optional[JobResult]:
        """Get result if job is complete, None otherwise."""
        if self.is_complete() and not self._result:
            self._result = self._fetch_result()
        return self._result

    @abstractmethod
    def _fetch_result(self) -> JobResult:
        """Internal method to fetch result when job completes."""
        pass

    def wait(self, stream_output: bool = False, poll_interval_s: float = 0.5) -> JobResult:
        """Wait for job to complete (sync).

        Args:
            stream_output: Stream logs to console as they arrive
            poll_interval_s: Seconds between status checks (default: 0.5s)

        Returns:
            JobResult when complete

        Raises:
            KeyboardInterrupt: If interrupted by user (after cleanup)
        """
        if not self._submitted:
            self.submit()

        start_time = time.time()
        printed_bytes = 0

        try:
            while not self.is_complete():
                # Check timeout
                if (time.time() - start_time) > self.timeout_s:
                    # Job timed out - attempt to cancel and mark timed out
                    try:
                        self.cancel()
                    except Exception:
                        pass
                    self._result = JobResult(
                        name=self.name,
                        exit_code=124,
                        logs_path=str(self._get_log_path()),
                        duration_s=time.time() - start_time,
                    )
                    return self._result

                # Stream output if requested
                if stream_output:
                    logs = self.get_logs()
                    if logs and len(logs) > printed_bytes:
                        new_content = logs[printed_bytes:]
                        print(new_content, end="", flush=True)
                        printed_bytes = len(logs)

                time.sleep(poll_interval_s)

            # Job complete - get result
            result = self.get_result()
            assert result is not None
            return result

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted! Cleaning up job '{self.name}'...")
            self._handle_interrupt()
            raise

    @abstractmethod
    def _handle_interrupt(self) -> None:
        """Handle job interruption (Ctrl+C). Subclasses can choose different behavior."""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Cancel/kill the running job."""
        pass

    @abstractmethod
    def _get_log_path(self) -> Path:
        """Get path to log file."""
        pass


class LocalJob(Job):
    """Job that runs locally via subprocess."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        timeout_s: int = 900,
        log_dir: str = "logs/local",
        cwd: Optional[str] = None,
    ):
        super().__init__(name, log_dir, timeout_s)
        self.cmd = cmd
        self.cwd = cwd or get_repo_root()
        self._proc: Optional[subprocess.Popen] = None
        self._exit_code: Optional[int] = None
        self._start_time: Optional[float] = None

    def submit(self) -> None:
        """Submit job for execution."""
        if self._submitted:
            return

        # Prepare log file
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up environment for color output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"] = "1"
        env["CLICOLOR_FORCE"] = "1"

        # Start process in new process group (for clean cancellation)
        self._proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setpgrp,
        )
        self._submitted = True
        self._start_time = time.time()

    def is_complete(self) -> bool:
        """Check if job has completed."""
        if not self._submitted:
            return False

        if self._exit_code is not None:
            return True

        # Poll process
        exit_code = self._proc.poll()
        if exit_code is not None:
            # Process finished - drain remaining output
            self._drain_output()
            self._exit_code = exit_code
            return True

        # Still running - read available output
        self._read_output()
        return False

    def _read_output(self) -> None:
        """Read available output from process and append to log file."""
        if not (self._proc and self._proc.stdout):
            return

        # Non-blocking buffered read
        chunk = self._proc.stdout.read1(65536)  # type: ignore[attr-defined]
        if chunk:
            log_path = self._get_log_path()
            with open(log_path, "ab") as f:
                f.write(chunk)

    def _drain_output(self) -> None:
        """Drain all remaining output from process and append to log file."""
        if not (self._proc and self._proc.stdout):
            return

        log_path = self._get_log_path()
        with open(log_path, "ab") as f:
            for line in self._proc.stdout:
                f.write(line)

    def get_logs(self) -> str:
        """Get current logs."""
        log_path = self._get_log_path()
        if log_path.exists():
            return log_path.read_text(errors="ignore")
        return ""

    def _fetch_result(self) -> JobResult:
        """Fetch result when job completes."""
        duration = None
        if self._start_time:
            duration = time.time() - self._start_time

        return JobResult(
            name=self.name,
            exit_code=self._exit_code if self._exit_code is not None else 1,
            logs_path=str(self._get_log_path()),
            duration_s=duration,
        )

    def _get_log_path(self) -> Path:
        """Get path to log file."""
        return self.log_dir / f"{self.name}.log"

    def _handle_interrupt(self) -> None:
        """Handle Ctrl+C by killing the local process."""
        self.cancel()

    def cancel(self) -> None:
        """Cancel/kill the running job."""
        if not (self._proc and self._exit_code is None):
            return

        print(f"Killing local job process (PID {self._proc.pid})...")
        try:
            # Kill the process group
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._proc.wait()

            self._exit_code = -1  # Mark as cancelled

        except (ProcessLookupError, PermissionError) as e:
            print(f"Could not kill process: {e}")


class RemoteJob(Job):
    """Job that runs remotely via SkyPilot using the Python SDK.

    Can be used individually or with class methods for batch operations.
    """

    def __init__(
        self,
        name: str,
        task_yaml: Optional[str] = None,
        task: Optional[sky.Task] = None,
        cluster_name: Optional[str] = None,
        timeout_s: int = 3600,
        log_dir: str = "logs/remote",
        job_id: Optional[int] = None,
    ):
        """Initialize a remote job.

        Args:
            name: Job name for logging/display
            task_yaml: Path to a SkyPilot task YAML file
            task: A sky.Task object (alternative to task_yaml)
            cluster_name: Name of the cluster to run on (will be auto-generated if None)
            timeout_s: Job timeout in seconds
            log_dir: Directory to store logs
            job_id: Existing job ID to resume (if provided, skips launch)
        """
        super().__init__(name, log_dir, timeout_s)

        if not task_yaml and not task and not job_id:
            raise ValueError("Must provide either task_yaml, task, or job_id")

        self.task_yaml = task_yaml
        self.task = task
        self.cluster_name = cluster_name or f"job-{name}"
        self._job_id: Optional[int] = job_id
        self._request_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._is_resumed = bool(job_id)
        self._job_status: Optional[sky.JobStatus] = None

    def submit(self) -> None:
        """Submit job to SkyPilot using the SDK."""
        if self._submitted:
            return

        # If resuming existing job, just mark as submitted
        if self._is_resumed:
            self._submitted = True
            self._start_time = time.time()
            return

        try:
            # Load task from YAML or use provided task
            if self.task_yaml:
                task = sky.Task.from_yaml(self.task_yaml)
            else:
                task = self.task

            # Launch the job using managed jobs
            self._request_id = sky.jobs.launch(task, name=self.cluster_name)
            self._submitted = True
            self._start_time = time.time()

            # Wait for the request to complete and get the job ID
            job_id, _controller_handle = sky.get(self._request_id)
            self._job_id = job_id

        except Exception as e:
            # Log launch failure
            log_path = self._get_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"SkyPilot launch failed: {e}\n")
            self._submitted = True  # Mark as submitted even if failed
            self._job_id = None

    def is_complete(self) -> bool:
        """Check if remote job has completed."""
        if not self._submitted:
            return False

        if not self._job_id:
            return True  # No job to track (launch failed)

        try:
            # Get job status from SkyPilot
            status_dict = sky.get(sky.job_status(self.cluster_name, job_ids=[self._job_id]))
            self._job_status = status_dict.get(self._job_id)

            if self._job_status is None:
                return True  # Job not found

            # Check if job is in a terminal state
            return self._job_status in (
                sky.JobStatus.SUCCEEDED,
                sky.JobStatus.FAILED,
                sky.JobStatus.FAILED_SETUP,
                sky.JobStatus.FAILED_DRIVER,
                sky.JobStatus.CANCELLED,
            )

        except Exception:
            # If we can't get status, assume not complete
            return False

    def get_logs(self) -> str:
        """Fetch latest logs from SkyPilot."""
        if not self._job_id:
            return ""

        log_path = self._get_log_path()

        try:
            # Use StringIO to capture logs
            log_buffer = StringIO()

            # Get logs without following
            sky.jobs.tail_logs(
                job_id=self._job_id,
                follow=False,
                tail=10000,
                output_stream=log_buffer,  # type: ignore
            )

            logs = log_buffer.getvalue()

            if logs:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(logs)

            return logs

        except Exception:
            # Return existing logs if fetch fails
            if log_path.exists():
                return log_path.read_text(errors="ignore")
            return ""

    def _fetch_result(self) -> JobResult:
        """Fetch result from completed job."""
        # Determine exit code from job status
        exit_code = 0
        if self._job_status == sky.JobStatus.SUCCEEDED:
            exit_code = 0
        elif self._job_status in (
            sky.JobStatus.FAILED,
            sky.JobStatus.FAILED_SETUP,
            sky.JobStatus.FAILED_DRIVER,
        ):
            exit_code = 1
        elif self._job_status == sky.JobStatus.CANCELLED:
            exit_code = 130  # Interrupted
        elif not self._job_id:
            exit_code = 1  # Launch failure

        duration = None
        if self._start_time:
            duration = time.time() - self._start_time

        return JobResult(
            name=self.name,
            exit_code=exit_code,
            logs_path=str(self._get_log_path()),
            job_id=str(self._job_id) if self._job_id else None,
            duration_s=duration,
        )

    def _get_log_path(self) -> Path:
        """Get path to log file."""
        job_id_str = str(self._job_id) if self._job_id else "unknown"
        return self.log_dir / f"{self.name}.{job_id_str}.log"

    def _handle_interrupt(self) -> None:
        """Handle Ctrl+C by warning about running remote job (does not cancel)."""
        if self._job_id:
            print(
                f"\n⚠️  Remote job '{self.name}' (Job ID: {self._job_id}) "
                f"is still running on cluster '{self.cluster_name}'"
            )
            print(f"    To cancel it later, run: sky jobs cancel {self._job_id}")
        else:
            print(f"\n⚠️  Remote job '{self.name}' was not successfully launched")

    def cancel(self) -> None:
        """Cancel the running job using the SkyPilot SDK."""
        if self._job_id:
            try:
                print(f"Cancelling remote job '{self.name}' (Job ID: {self._job_id})...")
                sky.jobs.cancel(job_ids=[self._job_id])
                print(f"✅ Job {self._job_id} cancelled successfully")
            except Exception as e:
                print(f"⚠️  Failed to cancel job {self._job_id}: {e}")
                print(f"    Try manually: sky jobs cancel {self._job_id}")
        else:
            print(f"Remote job '{self.name}' was not successfully launched")
