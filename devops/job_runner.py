"""Job runner for local and remote execution.

Provides a unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with support for both async and sync execution.

Example usage:
    # Local job (sync)
    job = LocalJob(name="test", cmd=["pytest", "tests/"], timeout_s=900)
    result = job.wait(stream_output=True)
    print(f"Exit code: {result.exit_code}")

    # Remote job (sync) - same API as LocalJob!
    job = RemoteJob(
        name="train",
        cmd=["uv", "run", "./tools/run.py", "train", "arena"],
        timeout_s=3600,
        resources={"accelerators": "V100:4", "use_spot": True},
        num_nodes=2,
    )
    result = job.wait(stream_output=True)
    print(f"Job ID: {result.job_id}, Exit code: {result.exit_code}")

    # Remote job (async - poll for completion)
    job = RemoteJob(name="train", cmd=["uv", "run", "./tools/run.py", "train", "arena"])
    job.submit()
    while not job.is_complete():
        time.sleep(10)
    result = job.get_result()

    # Attach to existing remote job
    job = RemoteJob(name="train", cmd=["echo", "dummy"], job_id=12345)
    result = job.wait(stream_output=True)
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import sky
import sky.exceptions
import sky.jobs

from devops.skypilot.utils.job_helpers import (
    check_job_statuses,
    get_job_id_from_request_id,
    get_request_id_from_launch_output,
    tail_job_log,
)
from metta.common.util.fs import get_repo_root


@dataclass
class JobResult:
    """Result of a completed job execution.

    All error information is available in the log file at logs_path.
    """

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
        """Submit job for execution (async - returns immediately).

        Any errors during submission should be written to the log file
        and reflected in the job's completion state.
        """
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

    def wait(
        self,
        stream_output: bool = False,
        poll_interval_s: float = 0.5,
        on_job_id_ready: Optional[callable] = None,
    ) -> JobResult:
        """Wait for job to complete (sync).

        Args:
            stream_output: Stream logs to console as they arrive
            poll_interval_s: Seconds between status checks (default: 0.5s)
            on_job_id_ready: Optional callback called when job_id becomes available.
                           Signature: callback(job_id: int) -> None

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

            # Job complete - stream any remaining output
            if stream_output:
                logs = self.get_logs()
                if logs and len(logs) > printed_bytes:
                    new_content = logs[printed_bytes:]
                    print(new_content, end="", flush=True)

            # Get final result
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
        cmd: Optional[list[str]] = None,
        timeout_s: int = 3600,
        log_dir: str = "logs/remote",
        cluster_name: Optional[str] = None,
        resources: Optional[dict] = None,
        num_nodes: int = 1,
        job_id: Optional[int] = None,
    ):
        """Initialize a remote job.

        Args:
            name: Job name for logging/display
            cmd: Command to run (similar to LocalJob API)
            timeout_s: Job timeout in seconds
            log_dir: Directory to store logs
            cluster_name: Name of the cluster to run on (will be auto-generated if None)
            resources: Resource requirements dict (e.g., {"accelerators": "V100:4", "use_spot": True})
            num_nodes: Number of nodes to use (default: 1)
            job_id: Existing job ID to resume (if provided, skips launch)
        """
        super().__init__(name, log_dir, timeout_s)

        if not cmd and not job_id:
            raise ValueError("Must provide either cmd or job_id")

        self.cmd = cmd
        self.cluster_name = cluster_name or f"job-{name}"
        self.resources = resources or {}
        self.num_nodes = num_nodes
        self._job_id: Optional[int] = job_id
        self._request_id: Optional[str] = None
        self._run_id: Optional[str] = None  # Run ID from launch.py (e.g., "jack.20251010.175808")
        self._start_time: Optional[float] = None
        self._is_resumed = bool(job_id)
        self._job_status: Optional[str] = None
        self._exit_code: Optional[int] = None  # Set if launch fails
        # Generate timestamp-based identifier for failed jobs
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def submit(self) -> None:
        """Submit job to SkyPilot using devops/skypilot/launch.py.

        Any launch errors are written to the log file immediately.
        """
        if self._submitted:
            return

        # If resuming existing job, just mark as submitted
        if self._is_resumed:
            self._submitted = True
            self._start_time = time.time()
            return

        # Prepare log file
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use the existing launch.py infrastructure which handles all SkyPilot complexity
            # Build the launch command - cmd should be a module path + args format
            # e.g., ["uv", "run", "./tools/run.py", "train", "arena", "run=test_123"]

            # Extract module path and args from self.cmd
            # The cmd format is expected to be: ["uv", "run", "./tools/run.py", <module_path>, ...args]
            if not self.cmd or len(self.cmd) < 4:
                raise ValueError(f"Invalid cmd format for remote job: {self.cmd}")

            # Find the module path (first arg after ./tools/run.py)
            module_path = None
            extra_args = []

            for i, arg in enumerate(self.cmd):
                if "run.py" in arg:
                    if i + 1 < len(self.cmd):
                        module_path = self.cmd[i + 1]
                        extra_args = self.cmd[i + 2 :]
                    break

            if not module_path:
                raise ValueError(f"Could not find module path in cmd: {self.cmd}")

            # Build launch.py command
            launch_cmd = [
                "uv",
                "run",
                "--active",
                "./devops/skypilot/launch.py",
                module_path,
            ]

            # Add extra arguments
            if extra_args:
                launch_cmd.extend(extra_args)

            # Run the launch command - stream output to log file in real-time
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    launch_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Stream output to log file line by line
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()

                proc.wait()

                if proc.returncode != 0:
                    # Launch failed - mark as submitted but with no job ID
                    # The error details are already in the log file
                    self._submitted = True
                    self._job_id = None
                    self._exit_code = proc.returncode
                    return

            # Launch succeeded - extract job info from log
            log_content = log_path.read_text()

            # Extract run ID (for display purposes)
            run_id_match = re.search(r"Using auto-generated run ID:\s*(\S+)", log_content)
            if run_id_match:
                self._run_id = run_id_match.group(1)

            # Use library function to extract request_id
            self._request_id = get_request_id_from_launch_output(log_content)

            self._submitted = True
            self._start_time = time.time()

        except Exception as e:
            # Write error to log file
            error_msg = f"Launch failed with exception: {e}\n"
            log_path.write_text(error_msg)

            # Mark as submitted but failed
            self._submitted = True
            self._job_id = None
            self._exit_code = 1

    def wait(
        self,
        stream_output: bool = False,
        poll_interval_s: float = 0.5,
        on_job_id_ready: Optional[callable] = None,
    ) -> JobResult:
        """Override wait() to call callback when job_id becomes available."""
        if not self._submitted:
            self.submit()

        start_time = time.time()
        printed_bytes = 0
        job_id_reported = False  # Track if we've called the callback

        try:
            while not self.is_complete():
                # Call callback when job_id first becomes available
                if not job_id_reported and self._job_id and on_job_id_ready:
                    on_job_id_ready(self._job_id)
                    job_id_reported = True

                # Check timeout
                if (time.time() - start_time) > self.timeout_s:
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

            # Job complete - stream any remaining output
            if stream_output:
                logs = self.get_logs()
                if logs and len(logs) > printed_bytes:
                    new_content = logs[printed_bytes:]
                    print(new_content, end="", flush=True)

            # Get final result
            result = self.get_result()
            assert result is not None
            return result

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted! Cleaning up job '{self.name}'...")
            self._handle_interrupt()
            raise

    def is_complete(self) -> bool:
        """Check if remote job has completed using SkyPilot SDK."""
        if not self._submitted:
            return False

        # If launch failed (no job ID but has exit code), we're complete
        if self._exit_code is not None:
            return True

        try:
            # If we don't have a job_id yet, try to get it from request_id
            if not self._job_id and self._request_id:
                # Use library function to map request_id to job_id
                job_id_str = get_job_id_from_request_id(self._request_id, wait_seconds=0.5)
                if job_id_str:
                    self._job_id = int(job_id_str)
                    print(f"✓ Mapped request {self._request_id[:8]}... to job ID: {self._job_id}")
                else:
                    # Request not mapped to job yet - keep waiting
                    return False

            # Still no job ID - job hasn't been assigned yet
            if not self._job_id:
                return False

            # Use library function to check job status
            job_statuses = check_job_statuses([self._job_id])
            job_info = job_statuses.get(self._job_id)

            if not job_info:
                # Job not found - consider it complete
                return True

            # Update our cached status
            self._job_status = job_info["status"]

            # Check if job is in a terminal state
            return self._job_status in (
                "SUCCEEDED",
                "FAILED",
                "FAILED_SETUP",
                "FAILED_DRIVER",
                "CANCELLED",
                "UNKNOWN",
                "ERROR",
            )

        except sky.exceptions.ClusterNotUpError:
            # Jobs controller not up - can't check status
            return False
        except Exception:
            # If we can't get status, assume not complete
            return False

    def get_logs(self) -> str:
        """Fetch latest logs from SkyPilot using SDK or return local launch logs."""
        log_path = self._get_log_path()

        # If we have no job ID (launch failed), return launch logs
        if not self._job_id:
            if log_path.exists():
                return log_path.read_text(errors="ignore")
            return ""

        # Use library function to get job logs
        logs = tail_job_log(str(self._job_id))

        if logs:
            # Cache logs to file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(logs)
            return logs

        # If library function returned None/empty, try reading cached logs
        if log_path.exists():
            return log_path.read_text(errors="ignore")

        return ""

    def _fetch_result(self) -> JobResult:
        """Fetch result from completed job."""
        # Determine exit code - check launch failure first
        if self._exit_code is not None:
            # Launch failed
            exit_code = self._exit_code
        elif self._job_status == "SUCCEEDED":
            exit_code = 0
        elif self._job_status in ("FAILED", "FAILED_SETUP", "FAILED_DRIVER", "UNKNOWN", "ERROR"):
            exit_code = 1
        elif self._job_status == "CANCELLED":
            exit_code = 130  # Interrupted
        elif not self._job_id:
            # No job ID and no exit code - shouldn't happen but handle it
            exit_code = 1
        else:
            exit_code = 0

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
        job_id_str = str(self._job_id) if self._job_id else self._timestamp
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
        """Cancel the running job using SkyPilot SDK."""
        if self._job_id:
            try:
                print(f"Cancelling remote job '{self.name}' (Job ID: {self._job_id})...")
                # Use SDK to cancel job
                sky.jobs.cancel(job_ids=[self._job_id])
                print(f"✅ Job {self._job_id} cancelled successfully")
            except sky.exceptions.ClusterNotUpError:
                print(f"⚠️  Jobs controller not up - cannot cancel job {self._job_id}")
                print(f"    Try manually later: sky jobs cancel {self._job_id}")
            except Exception as e:
                error_msg = str(e).lower()
                # Check if already terminated
                if "already in terminal state" in error_msg or "terminal state" in error_msg:
                    print(f"ℹ️  Job {self._job_id} is already in a terminal state")
                else:
                    print(f"⚠️  Failed to cancel job {self._job_id}: {e}")
                    print(f"    Try manually: sky jobs cancel {self._job_id}")
        else:
            print(f"Remote job '{self.name}' was not successfully launched")
