"""Job runner for local and remote execution.

Provides a unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with support for both async and sync execution.

Example usage:
    # Local job
    job = LocalJob(name="test", cmd=["pytest", "tests/"])
    job.submit()
    result = job.wait(stream_output=True)

    # Remote job
    job = RemoteJob(name="train", module="recipe.train", args=["run=test"])
    job.submit()
    while not job.is_complete():
        print(job.get_logs())
        time.sleep(10)
    result = job.get_result()
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from devops.skypilot.utils.job_helpers import tail_job_log
from devops.skypilot.utils.testing_helpers import LaunchedJob, SkyPilotTestLauncher
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
                    # Job timed out - create timeout result
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
            self.cancel()
            raise

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
        self._log_file = open(log_path, "wb")
        self._proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setpgrp if os.name != "nt" else None,  # Unix only
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
            self._log_file.close()
            return True

        # Still running - read available output
        self._read_output()
        return False

    def _read_output(self) -> None:
        """Read available output from process."""
        if self._proc and self._proc.stdout:
            # Read all available lines without blocking
            import select

            # Keep reading while data is available
            while select.select([self._proc.stdout], [], [], 0)[0]:
                line = self._proc.stdout.readline()
                if line:
                    self._log_file.write(line)
                    self._log_file.flush()
                else:
                    break

    def _drain_output(self) -> None:
        """Drain all remaining output from process."""
        if self._proc and self._proc.stdout:
            for line in self._proc.stdout:
                self._log_file.write(line)
            self._log_file.flush()

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

    def cancel(self) -> None:
        """Cancel/kill the running job."""
        if self._proc and self._exit_code is None:
            print(f"Killing local job process (PID {self._proc.pid})...")
            try:
                # Kill process group to ensure child processes are also killed
                import signal

                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)

                # Give it a moment to terminate gracefully
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(pgid, signal.SIGKILL)
                    self._proc.wait()

                self._exit_code = -1  # Mark as cancelled
                if hasattr(self, "_log_file") and self._log_file:
                    self._log_file.close()

            except (ProcessLookupError, PermissionError) as e:
                # Process already terminated or permission denied
                print(f"Could not kill process: {e}")


class RemoteJob(Job):
    """Job that runs remotely via SkyPilot."""

    def __init__(
        self,
        name: str,
        module: str,
        args: list[str],
        timeout_s: int = 3600,
        log_dir: str = "logs/remote",
        base_args: Optional[list[str]] = None,
        skip_git_check: bool = False,
        job_id: Optional[str] = None,
    ):
        super().__init__(name, log_dir, timeout_s)
        self.module = module
        self.args = args
        self.base_args = base_args or ["--no-spot", "--gpus=4", "--nodes", "1"]
        self.skip_git_check = skip_git_check
        self._job_id: Optional[str] = job_id  # Can resume existing job
        self._launched_job: Optional[LaunchedJob] = None
        self._start_time: Optional[float] = None
        self._is_resumed = bool(job_id)  # Track if this is a resumed job

    def submit(self) -> None:
        """Submit job to SkyPilot."""
        if self._submitted:
            return

        # If resuming existing job, skip launch
        if self._is_resumed:
            self._submitted = True
            self._start_time = time.time()
            return

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

        self._job_id = self._launched_job.job_id
        self._submitted = True
        self._start_time = time.time()

        # Write initial log if launch failed
        if not self._launched_job.success:
            log_path = self._get_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"SkyPilot launch failed\nRequest ID: {self._launched_job.request_id}\n")

    def is_complete(self) -> bool:
        """Check if remote job has completed."""
        if not self._submitted:
            return False

        # For resumed jobs or failed launches, check if we have a job_id
        if not self._job_id:
            return True  # No job to track (launch failed)

        # If not resumed, check if launch succeeded
        if not self._is_resumed and (not self._launched_job or not self._launched_job.success):
            return True  # Failed to launch

        # Check logs for completion marker
        logs = self.get_logs()
        return "Exit code:" in logs

    def get_logs(self) -> str:
        """Fetch latest logs from SkyPilot."""
        if not self._job_id:
            return ""

        log_path = self._get_log_path()

        try:
            logs = tail_job_log(self._job_id, lines=10000)
            if logs:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(logs)
                return logs
        except Exception:
            pass

        # Return existing logs if fetch fails
        if log_path.exists():
            return log_path.read_text(errors="ignore")
        return ""

    def _fetch_result(self) -> JobResult:
        """Fetch result from completed job."""
        logs = self.get_logs()

        # Extract exit code from logs
        exit_code = 1  # Default to failure
        match = re.search(r"Exit code: (\d+)", logs)
        if match:
            exit_code = int(match.group(1))
        elif not self._is_resumed and (not self._launched_job or not self._launched_job.success):
            exit_code = 1  # Launch failure (only for new jobs, not resumed)

        duration = None
        if self._start_time:
            duration = time.time() - self._start_time

        return JobResult(
            name=self.name,
            exit_code=exit_code,
            logs_path=str(self._get_log_path()),
            job_id=self._job_id,
            duration_s=duration,
        )

    def _get_log_path(self) -> Path:
        """Get path to log file."""
        job_id_str = self._job_id or "unknown"
        return self.log_dir / f"{self.name}.{job_id_str}.log"

    def cancel(self) -> None:
        """Note: Remote jobs cannot be cancelled from here.

        Remote jobs continue running in the cloud. Use SkyPilot CLI to manage:
            sky cancel <job_id>
        """
        if self._job_id:
            print(f"⚠️  Remote job '{self.name}' (Job ID: {self._job_id}) is still running in the cloud")
            print(f"    To cancel: sky cancel {self._job_id}")
        else:
            print(f"Remote job '{self.name}' was not successfully launched")


# Backwards compatibility - keep old function-based API
def run_local(
    name: str,
    cmd: list[str],
    timeout_s: int = 900,
    log_dir: str = "logs/local",
    cwd: Optional[str] = None,
    stream_output: bool = False,
) -> JobResult:
    """Run a command locally via subprocess (legacy API).

    Args:
        name: Job name (used for log filename)
        cmd: Command to run
        timeout_s: Timeout in seconds
        log_dir: Directory to write logs
        cwd: Working directory
        stream_output: If True, stream output to console

    Returns:
        JobResult with exit code and logs path
    """
    job = LocalJob(name=name, cmd=cmd, timeout_s=timeout_s, log_dir=log_dir, cwd=cwd)
    return job.wait(stream_output=stream_output)


def run_remote(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 3600,
    log_dir: str = "logs/remote",
    base_args: Optional[list[str]] = None,
    skip_git_check: bool = False,
    job_id: Optional[str] = None,
) -> "RemoteJobLegacy":
    """Run a job on SkyPilot (legacy API).

    Returns:
        RemoteJobLegacy that provides backwards-compatible API
    """
    job = RemoteJob(
        name=name,
        module=module,
        args=args,
        timeout_s=timeout_s,
        log_dir=log_dir,
        base_args=base_args,
        skip_git_check=skip_git_check,
        job_id=job_id,
    )
    job.submit()
    return RemoteJobLegacy(job)


class RemoteJobLegacy:
    """Legacy wrapper for RemoteJob to maintain backwards compatibility."""

    def __init__(self, job: RemoteJob):
        self._job = job
        self.name = job.name
        self.logs_path = str(job._get_log_path())
        self.job_id = job._job_id
        self.exit_code: Optional[int] = None

    def wait(self, timeout_s: Optional[int] = None, stream_output: bool = False) -> int:
        """Wait for job to complete."""
        if timeout_s:
            self._job.timeout_s = timeout_s

        result = self._job.wait(stream_output=stream_output)
        self.exit_code = result.exit_code
        return result.exit_code

    def get_logs(self) -> str:
        """Get current logs."""
        return self._job.get_logs()
