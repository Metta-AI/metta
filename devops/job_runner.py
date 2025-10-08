"""Job runner for local and remote execution.

Provides a simple, unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with synchronous wait and log fetching.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from devops.skypilot.utils.job_helpers import tail_job_log
from devops.skypilot.utils.testing_helpers import LaunchedJob, SkyPilotTestLauncher
from metta.common.util.fs import get_repo_root


class LocalJobResult:
    """A completed local job."""

    def __init__(self, name: str, logs_path: str, returncode: int):
        self.name = name
        self.logs_path = logs_path
        self.exit_code = returncode
        self.job_id: Optional[str] = None

    def get_logs(self) -> str:
        """Read logs from file."""
        if self.logs_path and Path(self.logs_path).exists():
            return Path(self.logs_path).read_text(errors="ignore")
        return ""


class RemoteJob:
    """A SkyPilot remote job."""

    def __init__(self, name: str, launched_job: LaunchedJob, logs_path: Path):
        self.name = name
        self.logs_path = str(logs_path)
        self.job_id: Optional[str] = launched_job.job_id
        self.exit_code: Optional[int] = None
        self._launched_job = launched_job
        self._logs_path_obj = logs_path

    def wait(self, timeout_s: Optional[int] = None, stream_output: bool = False) -> int:
        """Poll until job completes or times out.

        Args:
            timeout_s: Timeout in seconds
            stream_output: If True, stream new log lines to console as they arrive

        Returns:
            Exit code of the job
        """
        if not self._launched_job.success or not self.job_id:
            self.exit_code = 1
            return 1

        start = time.time()
        printed_bytes = 0

        while True:
            logs = self.get_logs()

            # Stream new output to console
            if stream_output and logs and len(logs) > printed_bytes:
                new_content = logs[printed_bytes:]
                print(new_content, end="", flush=True)
                printed_bytes = len(logs)

            # Check for completion marker
            if "Exit code:" in logs:
                match = re.search(r"Exit code: (\d+)", logs)
                if match:
                    self.exit_code = int(match.group(1))
                    return self.exit_code

            if timeout_s and (time.time() - start) > timeout_s:
                self.exit_code = 124
                return 124

            time.sleep(10)

    def get_logs(self) -> str:
        """Fetch latest logs from SkyPilot."""
        if not self.job_id:
            return ""

        try:
            logs = tail_job_log(self.job_id, lines=10000)
            if logs:
                self._logs_path_obj.parent.mkdir(parents=True, exist_ok=True)
                with open(self._logs_path_obj, "w") as f:
                    f.write(logs)
                return logs
        except Exception:
            pass

        # Return existing logs if fetch fails
        if self._logs_path_obj.exists():
            return self._logs_path_obj.read_text(errors="ignore")
        return ""


def run_local(
    name: str,
    cmd: list[str],
    timeout_s: int = 900,
    log_dir: str = "logs/local",
    cwd: Optional[str] = None,
    stream_output: bool = False,
) -> LocalJobResult:
    """Run a command locally via subprocess.

    Args:
        name: Job name (used for log filename)
        cmd: Command to run (e.g., ["uv", "run", "./tools/run.py", "recipe", ...])
        timeout_s: Timeout in seconds
        log_dir: Directory to write logs
        cwd: Working directory (defaults to current git repo root)
        stream_output: If True, stream output to console in real-time

    Returns:
        LocalJobResult with exit code and logs path
    """
    cwd = cwd or get_repo_root()
    log_path = Path(log_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if stream_output:
            # Use Popen for streaming output
            # Force color output and unbuffered mode
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FORCE_COLOR"] = "1"
            env["CLICOLOR_FORCE"] = "1"

            with open(log_path, "wb") as lf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                # Stream output to both console and log file (preserve colors)
                for line in proc.stdout:
                    # Write raw bytes to preserve ANSI color codes
                    print(line.decode("utf-8", errors="replace"), end="", flush=True)
                    lf.write(line)

                # Wait for process to complete
                proc.wait(timeout=timeout_s)
                return LocalJobResult(name=name, logs_path=str(log_path), returncode=proc.returncode)
        else:
            with open(log_path, "w") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    text=True,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
            return LocalJobResult(name=name, logs_path=str(log_path), returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as lf:
            lf.write(f"\n\n[TIMEOUT] Job exceeded {timeout_s} seconds\n")
        return LocalJobResult(name=name, logs_path=str(log_path), returncode=124)


def run_remote(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 3600,
    log_dir: str = "logs/remote",
    base_args: Optional[list[str]] = None,
    skip_git_check: bool = False,
) -> RemoteJob:
    """Run a job on SkyPilot.

    Args:
        name: Job name
        module: Recipe module path
        args: Additional arguments
        timeout_s: Timeout in seconds
        log_dir: Directory to write logs
        base_args: Base SkyPilot args (defaults to --no-spot --gpus=4 --nodes=1)
        skip_git_check: Skip git state validation

    Returns:
        RemoteJob that can be waited on and polled for logs
    """
    launcher = SkyPilotTestLauncher(base_name="job", skip_git_check=skip_git_check)
    run_name = launcher.generate_run_name(name)
    base_args = base_args or ["--no-spot", "--gpus=4", "--nodes", "1"]

    launched_job = launcher.launch_job(
        module=module,
        run_name=run_name,
        base_args=base_args,
        extra_args=args,
        test_config={"name": name},
        enable_ci_tests=False,
    )

    log_path = Path(log_dir) / f"{name}.{launched_job.job_id or 'unknown'}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not launched_job.success:
        log_path.write_text(f"SkyPilot launch failed\nRequest ID: {launched_job.request_id}\n")

    return RemoteJob(name=name, launched_job=launched_job, logs_path=log_path)
