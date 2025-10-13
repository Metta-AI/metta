"""Job runner for local and remote execution.

Moved from devops/job_runner.py to be shared across systems.

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
        module="experiments.recipes.arena.train",
        args=["run=test", "trainer.total_timesteps=100000"],
        timeout_s=3600,
        base_args=["--gpus=4", "--no-spot"]
    )
    result = job.wait(stream_output=True)
    print(f"Job ID: {result.job_id}, Exit code: {result.exit_code}")

    # Remote job (async - poll for completion)
    job = RemoteJob(
        name="train",
        module="experiments.recipes.arena.train",
        args=["run=test"]
    )
    job.submit()
    while not job.is_complete():
        time.sleep(10)
    result = job.get_result()

    # Attach to existing remote job
    job = RemoteJob(
        name="train",
        module="experiments.recipes.arena.train",
        args=["run=test"],
        job_id=12345
    )
    result = job.wait(stream_output=True)
"""

from __future__ import annotations

import os
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

from devops.skypilot.utils.job_helpers import check_job_statuses, tail_job_log
from devops.skypilot.utils.testing_helpers import LaunchedJob, SkyPilotTestLauncher
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
    ) -> JobResult:
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

        # Prepare log file (clear any existing content from previous runs)
        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()  # Delete old log file to avoid showing stale output

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
    """Job that runs remotely via SkyPilot using SkyPilotTestLauncher."""

    def __init__(
        self,
        name: str,
        module: Optional[str] = None,
        args: Optional[list[str]] = None,
        timeout_s: int = 3600,
        log_dir: str = "logs/remote",
        base_args: Optional[list[str]] = None,
        job_id: Optional[int] = None,
        skip_git_check: bool = False,
    ):
        """Initialize a remote job.

        Args:
            name: Job name for logging/display
            module: Module path to run (e.g., "experiments.recipes.arena.train")
            args: Arguments to pass to module (e.g., ["run=test", "trainer.total_timesteps=100000"])
            timeout_s: Job timeout in seconds
            log_dir: Directory to store logs
            base_args: Base arguments for launch (e.g., ["--gpus=4", "--no-spot"])
            job_id: Existing job ID to resume (if provided, skips launch)
            skip_git_check: Skip git state validation (default: False)

        Raises:
            ValueError: If neither module nor job_id is provided
        """
        super().__init__(name, log_dir, timeout_s)

        # Either module or job_id must be provided
        if not module and not job_id:
            raise ValueError("Must provide either module or job_id")

        self.module = module
        self.args = args or []
        self.base_args = base_args or ["--no-spot", "--gpus=4", "--nodes", "1"]
        self.skip_git_check = skip_git_check
        self._job_id: Optional[int] = job_id
        self._launched_job: Optional["LaunchedJob"] = None
        self._start_time: Optional[float] = None
        self._is_resumed = bool(job_id)
        self._job_status: Optional[str] = None
        self._exit_code: Optional[int] = None  # Set if launch fails
        # Generate timestamp-based identifier for failed jobs
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Track whether we've fetched full history yet
        self._full_logs_fetched = False

    def submit(self) -> None:
        """Submit job to SkyPilot using SkyPilotTestLauncher.

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
            # Use SkyPilotTestLauncher - handles all launch complexity
            launcher = SkyPilotTestLauncher(base_name="job", skip_git_check=self.skip_git_check)
            run_name = launcher.generate_run_name(self.name)

            self._launched_job = launcher.launch_job(
                module=self.module,
                run_name=run_name,
                base_args=self.base_args,
                extra_args=self.args,
                test_config={"name": self.name},
                enable_ci_tests=False,
            )

            self._job_id = int(self._launched_job.job_id) if self._launched_job.job_id else None
            self._submitted = True
            self._start_time = time.time()

            # Write initial log if launch failed
            if not self._launched_job.success:
                log_path.write_text(f"SkyPilot launch failed\nRequest ID: {self._launched_job.request_id}\n")
                self._exit_code = 1

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
        poll_interval_s: float = 5.0,
        on_job_id_ready: Optional[callable] = None,
    ) -> JobResult:
        """Override wait() to call callback when job_id becomes available.

        Args:
            stream_output: Stream logs to console
            poll_interval_s: Seconds between status checks (default: 5.0 for remote jobs)
            on_job_id_ready: Callback when job_id becomes available
        """
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

        # No job ID means launch failed
        if not self._job_id:
            return True

        try:
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
        """Fetch logs from SkyPilot and maintain complete log history in local cache.

        Strategy:
        - First call: Fetch all available logs (very large tail) to capture everything including wandb links
        - Subsequent calls: Fetch recent logs and append only new content by comparing with cached logs
        - Local cache file always contains complete log history for artifact extraction
        - The wait() method tracks printed_bytes offset for streaming output
        """
        log_path = self._get_log_path()

        # If we have no job ID (launch failed), return launch logs
        if not self._job_id:
            if log_path.exists():
                return log_path.read_text(errors="ignore")
            return ""

        # Read existing cached logs to determine what we already have
        existing_logs = ""
        if log_path.exists():
            existing_logs = log_path.read_text(errors="ignore")
            existing_len = len(existing_logs)
        else:
            existing_len = 0

        # Determine how many lines to fetch
        if not self._full_logs_fetched:
            # First fetch: Get all available logs (use very large tail)
            # This captures wandb links and early output for artifact extraction
            lines = 1000000
        else:
            # Subsequent fetches: Get enough recent lines to overlap with cached content
            # This allows us to detect where new content starts
            lines = 500

        # Fetch logs from remote job
        fetched_logs = tail_job_log(str(self._job_id), lines=lines)

        if fetched_logs:
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if not self._full_logs_fetched:
                # First fetch: Write all logs to cache
                log_path.write_text(fetched_logs)
                self._full_logs_fetched = True
                return fetched_logs
            else:
                # Subsequent fetches: Append only new content
                # If fetched logs are longer than cached, there's new content
                if len(fetched_logs) > existing_len:
                    # New content exists - find where it starts
                    # Strategy: Find overlap by checking if end of cached logs appears in fetched logs
                    overlap_found = False
                    if existing_logs:
                        # Take last 500 chars from existing logs as overlap marker
                        overlap_marker = existing_logs[-min(500, len(existing_logs)) :]
                        if overlap_marker in fetched_logs:
                            # Find where the overlap marker appears in fetched logs
                            marker_pos = fetched_logs.rfind(overlap_marker)
                            # New content is everything AFTER the marker
                            new_content = fetched_logs[marker_pos + len(overlap_marker) :]
                            if new_content:
                                # Append new content to cache file
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(new_content)
                                overlap_found = True

                    if not overlap_found:
                        # No clear overlap - just use the fetched logs as they're longer
                        # This handles cases where logs rotated or we're far behind
                        log_path.write_text(fetched_logs)

                # Return updated cache content
                return log_path.read_text(errors="ignore")

        # If fetch returned nothing, return cached logs
        if existing_logs:
            return existing_logs

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
            print(f"\n⚠️  Remote job '{self.name}' (Job ID: {self._job_id}) is still running in the cloud")
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
