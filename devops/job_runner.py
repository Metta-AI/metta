"""Job runner for local and remote execution.

Provides a simple, unified interface for running jobs locally via subprocess
or remotely via SkyPilot, with synchronous wait and log fetching.
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from devops.skypilot.utils.job_helpers import tail_job_log
from devops.skypilot.utils.testing_helpers import LaunchedJob, SkyPilotTestLauncher


def repo_root() -> str:
    """Get repository root dynamically via git."""
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, text=True, capture_output=True)
    return result.stdout.strip()


class LocalJob:
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

    def wait(self, timeout_s: Optional[int] = None) -> int:
        """Poll until job completes or times out."""
        if not self._launched_job.success or not self.job_id:
            self.exit_code = 1
            return 1

        start = time.time()
        while True:
            logs = self.get_logs()
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
) -> LocalJob:
    """Run a command locally via subprocess.

    Args:
        name: Job name (used for log filename)
        cmd: Command to run (e.g., ["uv", "run", "./tools/run.py", "recipe", ...])
        timeout_s: Timeout in seconds
        log_dir: Directory to write logs
        cwd: Working directory (defaults to current git repo root)

    Returns:
        LocalJob with exit code and logs path
    """
    cwd = cwd or repo_root()
    log_path = Path(log_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                text=True,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        return LocalJob(name=name, logs_path=str(log_path), returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as lf:
            lf.write(f"\n\n[TIMEOUT] Job exceeded {timeout_s} seconds\n")
        return LocalJob(name=name, logs_path=str(log_path), returncode=124)


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
