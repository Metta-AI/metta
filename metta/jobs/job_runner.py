"""Job runner for local and remote execution.

Unified interface for running jobs locally via subprocess or remotely via SkyPilot.
Supports both sync (wait) and async (submit + poll) execution patterns.
"""

from __future__ import annotations

import os
import shlex
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
from metta.common.util.retry import retry_function
from metta.jobs.job_config import JobConfig


@dataclass
class JobResult:
    """Result of a completed job execution."""

    name: str
    exit_code: int
    logs_path: str
    job_id: Optional[str] = None
    duration_s: Optional[float] = None

    def get_logs(self) -> str:
        if self.logs_path and Path(self.logs_path).exists():
            return Path(self.logs_path).read_text(errors="ignore")
        return ""

    @property
    def success(self) -> bool:
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
        pass

    @abstractmethod
    def get_logs(self) -> str:
        pass

    def get_result(self) -> Optional[JobResult]:
        if self.is_complete() and not self._result:
            self._result = self._fetch_result()
        return self._result

    @abstractmethod
    def _fetch_result(self) -> JobResult:
        pass

    def wait(
        self,
        stream_output: bool = False,
        poll_interval_s: float = 0.5,
    ) -> JobResult:
        """Wait for job to complete (sync), optionally streaming output."""
        if not self._submitted:
            self.submit()

        start_time = time.time()
        printed_bytes = 0

        try:
            while not self.is_complete():
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

                if stream_output:
                    logs = self.get_logs()
                    if logs and len(logs) > printed_bytes:
                        new_content = logs[printed_bytes:]
                        print(new_content, end="", flush=True)
                        printed_bytes = len(logs)

                time.sleep(poll_interval_s)

            if stream_output:
                logs = self.get_logs()
                if logs and len(logs) > printed_bytes:
                    new_content = logs[printed_bytes:]
                    print(new_content, end="", flush=True)

            result = self.get_result()
            assert result is not None
            return result

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted! Cleaning up job '{self.name}'...")
            self._handle_interrupt()
            raise

    @abstractmethod
    def _handle_interrupt(self) -> None:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass

    @abstractmethod
    def _get_log_path(self) -> Path:
        pass

    @property
    def log_path(self) -> str:
        """Public property to get the log file path."""
        return str(self._get_log_path())

    @property
    def job_id(self) -> str | None:
        """Public property to get the job ID (PID for local, SkyPilot ID for remote)."""
        return None


class LocalJob(Job):
    """Job that runs locally via subprocess."""

    def __init__(
        self,
        config: JobConfig,
        log_dir: str = "logs/local",
        cwd: Optional[str] = None,
    ):
        super().__init__(config.name, log_dir, config.timeout_s)
        self.cwd = cwd or get_repo_root()

        if "cmd" in config.metadata:
            cmd = config.metadata["cmd"]
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
            self.cmd = cmd
        else:
            self.cmd = ["uv", "run", "./tools/run.py", config.module]
            for k, v in config.args.items():
                self.cmd.append(f"{k}={v}")
            for k, v in config.overrides.items():
                self.cmd.append(f"{k}={v}")

        self._proc: Optional[subprocess.Popen] = None
        self._exit_code: Optional[int] = None
        self._start_time: Optional[float] = None

    def submit(self) -> None:
        """Start local subprocess and begin capturing output.

        Output handling:
        - Merged stdout/stderr to single stream for unified logging
        - Process group created (os.setpgrp) for clean cancellation of child processes
        - Log file cleared on submit to avoid stale output from previous runs
        - Environment vars set to force unbuffered/colored output
        """
        if self._submitted:
            return

        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"] = "1"
        env["CLICOLOR_FORCE"] = "1"

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
        """Check completion and incrementally capture output.

        Important: Always call this during polling to capture output!
        Side effects:
        - If still running: reads available output (non-blocking)
        - If just completed: drains remaining output (blocking)
        - Output written incrementally to log file for live streaming
        """
        if not self._submitted:
            return False
        if self._exit_code is not None:
            return True

        exit_code = self._proc.poll()
        if exit_code is not None:
            self._drain_output()
            self._exit_code = exit_code
            return True

        self._read_output()
        return False

    def _read_output(self) -> None:
        """Non-blocking read of available output (called while job runs)."""
        if not (self._proc and self._proc.stdout):
            return
        chunk = self._proc.stdout.read1(65536)  # type: ignore[attr-defined]
        if chunk:
            log_path = self._get_log_path()
            with open(log_path, "ab") as f:
                f.write(chunk)

    def _drain_output(self) -> None:
        """Blocking read of all remaining output (called when job completes)."""
        if not (self._proc and self._proc.stdout):
            return
        log_path = self._get_log_path()
        with open(log_path, "ab") as f:
            for line in self._proc.stdout:
                f.write(line)

    def get_logs(self) -> str:
        log_path = self._get_log_path()
        if log_path.exists():
            return log_path.read_text(errors="ignore")
        return ""

    def _fetch_result(self) -> JobResult:
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
        return self.log_dir / f"{self.name}.log"

    @property
    def job_id(self) -> str | None:
        """Return PID of local process if available."""
        if self._proc:
            return str(self._proc.pid)
        return None

    def _handle_interrupt(self) -> None:
        self.cancel()

    def cancel(self) -> None:
        """Kill local process and all children.

        Process group handling:
        - Kills entire process group (created via preexec_fn=os.setpgrp)
        - Ensures child processes spawned by the job are also terminated
        - First tries SIGTERM (graceful), escalates to SIGKILL if needed
        """
        if not (self._proc and self._exit_code is None):
            return

        print(f"Killing local job process (PID {self._proc.pid})...")
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._proc.wait()

            self._exit_code = -1

        except (ProcessLookupError, PermissionError) as e:
            print(f"Could not kill process: {e}")


class RemoteJob(Job):
    """Job that runs remotely via SkyPilot by calling launch.py directly."""

    def __init__(
        self,
        config: JobConfig,
        log_dir: str = "logs/remote",
        job_id: Optional[int] = None,
        skip_git_check: bool = False,
    ):
        super().__init__(config.name, log_dir, config.timeout_s)

        if not config.remote and not job_id:
            raise ValueError("RemoteJob requires config.remote to be set (or job_id for resuming)")

        arg_list = []
        for k, v in config.args.items():
            arg_list.append(f"{k}={v}")
        for k, v in config.overrides.items():
            arg_list.append(f"{k}={v}")

        if config.remote:
            base_args = [f"--gpus={config.remote.gpus}", f"--nodes={config.remote.nodes}"]
            if not config.remote.spot:
                base_args.insert(0, "--no-spot")
        else:
            base_args = ["--no-spot", "--gpus=4", "--nodes", "1"]

        self.config = config
        self.module = config.module
        self.args = arg_list
        self.base_args = base_args
        self.skip_git_check = skip_git_check
        self._job_id: Optional[int] = job_id
        self._request_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._is_resumed = bool(job_id)
        self._job_status: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._full_logs_fetched = False
        self._run_name: Optional[str] = None  # WandB run name (only for training jobs)

    def _generate_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.name.replace("/", "_")
        return f"job_{safe_name}_{timestamp}"

    def _launch_via_script(self, run_name: str | None) -> tuple[Optional[str], Optional[int], str]:
        """Launch job via launch.py script, returning (request_id, job_id, output).

        SkyPilot launch flow:
        1. Call launch.py subprocess with job config
        2. Extract request_id from output (required for all operations)
        3. Poll for job_id (may not be immediately available)
        4. Job_id=None is OK - we'll fetch it later via is_complete()

        Args:
            run_name: WandB run name (only passed for training jobs, None otherwise)
        """
        cmd = [
            "devops/skypilot/launch.py",
            *self.base_args,
            self.module,
        ]

        # Only pass run= for training jobs (they use WandB for experiment tracking)
        if run_name:
            cmd.append(f"run={run_name}")

        cmd.extend(self.args)

        if self.skip_git_check:
            cmd.append("--skip-git-check")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=get_repo_root())
        full_output = result.stdout + "\n" + result.stderr
        request_id = get_request_id_from_launch_output(full_output)

        if not request_id:
            if "sky-jobs-controller" in full_output.lower() and "not up" in full_output.lower():
                raise Exception("Jobs controller appears to be down")
            debug_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
            raise Exception(f"Failed to get request ID from launch output. Debug: {debug_info}")

        try:

            def get_job_id_with_wait() -> str:
                job_id = get_job_id_from_request_id(request_id, wait_seconds=2.0)
                if not job_id:
                    raise Exception("Job ID not available yet")
                return job_id

            job_id_str = retry_function(get_job_id_with_wait, max_retries=2, initial_delay=2.0)
            job_id = int(job_id_str)
        except Exception:
            job_id = None

        return request_id, job_id, full_output

    def submit(self, max_attempts: int = 3) -> None:
        """Submit job to SkyPilot by calling launch.py directly.

        Retry behavior:
        - Retries entire launch process (subprocess + ID extraction) up to max_attempts
        - Launch errors written to log file for debugging
        - On failure: marks submitted=True but sets exit_code=1 (failed launch)

        Resume mode:
        - If job_id provided at construction, skips launch (attaching to existing job)
        - Used for resuming after script restart or reattaching to orphaned jobs
        """
        if self._submitted:
            return

        if self._is_resumed:
            self._submitted = True
            self._start_time = time.time()
            return

        log_path = self._get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Only generate run_name for training jobs (they use WandB)
            run_name = self._generate_run_name() if self.config.is_training_job else None
            self._run_name = run_name  # Store for WandB URL construction (None for non-training)
            request_id, job_id, _ = retry_function(
                lambda: self._launch_via_script(run_name),
                max_retries=max_attempts - 1,
            )
            self._request_id = request_id
            self._job_id = job_id
            self._submitted = True
            self._start_time = time.time()

        except Exception as e:
            error_msg = f"Launch failed after {max_attempts} attempts: {e}\n"
            log_path.write_text(error_msg)
            self._submitted = True
            self._job_id = None
            self._request_id = None
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
        """Check if remote job has finished via SkyPilot API.

        Status polling:
        - Queries SkyPilot for current job status (RUNNING, SUCCEEDED, FAILED, etc.)
        - Caches status in _job_status to avoid redundant API calls
        - Returns True if job_id unavailable (launch failed - nothing to poll)

        Terminal statuses: SUCCEEDED, FAILED, FAILED_SETUP, FAILED_DRIVER, CANCELLED, UNKNOWN, ERROR
        """
        if not self._submitted:
            return False

        if self._exit_code is not None:
            return True

        if not self._job_id:
            return True

        try:
            job_statuses = check_job_statuses([self._job_id])
            job_info = job_statuses.get(self._job_id)

            if not job_info:
                return True

            self._job_status = job_info["status"]

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
            return False
        except Exception:
            return False

    def get_logs(self) -> str:
        """Fetch logs from SkyPilot and maintain complete log history in local cache.

        Workflow:
        1. First call: Fetch all available logs (1M lines) to capture wandb links and early output
        2. Subsequent calls: Fetch recent logs (500 lines) and detect overlap with cached content
        3. Append only new content by finding where cached logs end in fetched logs
        4. Maintains complete history in local file for artifact extraction
        """
        log_path = self._get_log_path()

        if not self._job_id:
            if log_path.exists():
                return log_path.read_text(errors="ignore")
            return ""

        existing_logs = ""
        if log_path.exists():
            existing_logs = log_path.read_text(errors="ignore")
            existing_len = len(existing_logs)
        else:
            existing_len = 0

        lines = 1000000 if not self._full_logs_fetched else 500
        fetched_logs = tail_job_log(str(self._job_id), lines=lines)

        if fetched_logs:
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if not self._full_logs_fetched:
                log_path.write_text(fetched_logs)
                self._full_logs_fetched = True
                return fetched_logs
            else:
                if len(fetched_logs) > existing_len:
                    overlap_found = False
                    if existing_logs:
                        overlap_marker = existing_logs[-min(500, len(existing_logs)) :]
                        if overlap_marker in fetched_logs:
                            marker_pos = fetched_logs.rfind(overlap_marker)
                            new_content = fetched_logs[marker_pos + len(overlap_marker) :]
                            if new_content:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(new_content)
                                overlap_found = True

                    if not overlap_found:
                        log_path.write_text(fetched_logs)

                return log_path.read_text(errors="ignore")

        if existing_logs:
            return existing_logs
        return ""

    def _fetch_result(self) -> JobResult:
        if self._exit_code is not None:
            exit_code = self._exit_code
        elif self._job_status == "SUCCEEDED":
            exit_code = 0
        elif self._job_status in ("FAILED", "FAILED_SETUP", "FAILED_DRIVER", "UNKNOWN", "ERROR"):
            exit_code = 1
        elif self._job_status == "CANCELLED":
            exit_code = 130
        elif not self._job_id:
            exit_code = 1
        else:
            # Unknown status - default to failure for safety
            exit_code = 1

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
        job_id_str = str(self._job_id) if self._job_id else self._timestamp
        return self.log_dir / f"{self.name}.{job_id_str}.log"

    @property
    def job_id(self) -> str | None:
        """Return SkyPilot job ID if available."""
        if self._job_id:
            return str(self._job_id)
        return None

    @property
    def request_id(self) -> str | None:
        """Return SkyPilot request ID if available."""
        return self._request_id

    @property
    def run_name(self) -> str | None:
        """Return the WandB run name used for this job."""
        return self._run_name

    def _handle_interrupt(self) -> None:
        if self._job_id:
            print(f"\n⚠️  Remote job '{self.name}' (Job ID: {self._job_id}) is still running in the cloud")
            print(f"    To cancel it later, run: sky jobs cancel {self._job_id}")
        else:
            print(f"\n⚠️  Remote job '{self.name}' was not successfully launched")

    def cancel(self) -> None:
        """Cancel remote job via SkyPilot API.

        Behavior:
        - Uses sky.jobs.cancel() to send cancellation request
        - Job continues in cloud briefly (graceful shutdown)
        - If job_id unavailable (launch failed), nothing to cancel
        - Provides manual command if API call fails
        - Handles "already in terminal state" gracefully
        """
        if self._job_id:
            try:
                print(f"Cancelling remote job '{self.name}' (Job ID: {self._job_id})...")
                sky.jobs.cancel(job_ids=[self._job_id])
                print(f"✅ Job {self._job_id} cancelled successfully")
            except sky.exceptions.ClusterNotUpError:
                print(f"⚠️  Jobs controller not up - cannot cancel job {self._job_id}")
                print(f"    Try manually later: sky jobs cancel {self._job_id}")
            except Exception as e:
                error_msg = str(e).lower()
                if "already in terminal state" in error_msg or "terminal state" in error_msg:
                    print(f"ℹ️  Job {self._job_id} is already in a terminal state")
                else:
                    print(f"⚠️  Failed to cancel job {self._job_id}: {e}")
                    print(f"    Try manually: sky jobs cancel {self._job_id}")
        else:
            print(f"Remote job '{self.name}' was not successfully launched")
