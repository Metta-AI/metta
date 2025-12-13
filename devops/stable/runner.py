"""Job runner for stable release validation and CI."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import wandb
from pydantic import BaseModel, Field

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


class JobStatus(StrEnum):
    NOT_STARTED = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


Operator = Literal[">=", ">", "<=", "<", "==", "in"]


class AcceptanceCriterion(BaseModel):
    metric: str
    threshold: float | tuple[float, float]
    operator: Operator = ">="


class Job(BaseModel):
    name: str
    cmd: list[str]
    timeout_s: int = 3600
    remote_gpus: int | None = None
    remote_nodes: int | None = None
    dependencies: list[str] = Field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = Field(default_factory=list)
    wandb_run_name: str | None = None

    status: JobStatus = JobStatus.NOT_STARTED
    exit_code: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    logs_path: str | None = None
    skypilot_job_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    acceptance_passed: bool | None = None
    error: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.remote_gpus is not None or self.remote_nodes is not None

    @property
    def wandb_url(self) -> str | None:
        if not self.wandb_run_name:
            return None
        return f"https://wandb.ai/{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/runs/{self.wandb_run_name}"


class Runner:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.state_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[Future, Job] = {}
        self._output_lock = threading.Lock()

    def add_job(self, job: Job) -> None:
        self.jobs[job.name] = job

    def run_all(self) -> dict[str, Job]:
        """Run all jobs, respecting dependencies. Returns final job states."""
        self._executor = ThreadPoolExecutor(max_workers=min(32, max(1, len(self.jobs))))
        self._futures = {}
        last_status_time = time.time()

        try:
            while True:
                self._check_completed_futures()

                pending = [j for j in self.jobs.values() if j.status == JobStatus.NOT_STARTED]
                running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]

                if not pending and not running:
                    break

                if time.time() - last_status_time >= 600:
                    self._print_status_summary()
                    last_status_time = time.time()

                ready = self._get_ready_jobs(pending)
                for job in ready:
                    self._start_job(job)

                remote_running = [j for j in running if j.is_remote]
                if remote_running:
                    self._poll_remote_jobs()
                    time.sleep(5)
                elif self._futures:
                    time.sleep(0.1)
        finally:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._fetch_metrics_and_evaluate()
        return self.jobs

    def _check_completed_futures(self) -> None:
        completed = [f for f in self._futures if f.done()]
        for future in completed:
            job = self._futures.pop(future)
            try:
                future.result()
            except Exception as e:
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error = str(e)

    def _get_ready_jobs(self, pending: list[Job]) -> list[Job]:
        ready = []
        for job in pending:
            deps_satisfied = True
            for dep_name in job.dependencies:
                dep = self.jobs.get(dep_name)
                if not dep:
                    job.status = JobStatus.SKIPPED
                    job.error = f"Missing dependency: {dep_name}"
                    deps_satisfied = False
                    break
                if dep.status in (JobStatus.FAILED, JobStatus.SKIPPED):
                    job.status = JobStatus.SKIPPED
                    job.error = f"Dependency {dep.status.value.lower()}: {dep_name}"
                    deps_satisfied = False
                    break
                if dep.status in (JobStatus.NOT_STARTED, JobStatus.RUNNING):
                    deps_satisfied = False
                    break
                else:
                    assert dep.status == JobStatus.SUCCEEDED

            if deps_satisfied and job.status == JobStatus.NOT_STARTED:
                ready.append(job)
        return ready

    def _start_job(self, job: Job) -> None:
        assert self._executor is not None
        if job.is_remote:
            future = self._executor.submit(self._launch_remote_job, job)
        else:
            future = self._executor.submit(self._run_local_job, job)
        self._futures[future] = job

    def _print(self, msg: str) -> None:
        with self._output_lock:
            print(msg, flush=True)

    def _complete_job(self, job: Job, status: JobStatus, exit_code: int, error: str | None = None) -> None:
        job.status = status
        job.exit_code = exit_code
        job.error = error
        job.completed_at = datetime.now().isoformat()
        assert job.started_at is not None
        started = datetime.fromisoformat(job.started_at)
        job.duration_s = (datetime.fromisoformat(job.completed_at) - started).total_seconds()
        result = "PASSED" if status == JobStatus.SUCCEEDED else f"FAILED: {error}"
        self._print(f"[{job.name}] {result} (duration={job.duration_s:.0f}s)")

    def _print_status_summary(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._print(f"\n[{timestamp}] === Status Summary ===")
        for job in self.jobs.values():
            elapsed = ""
            if job.started_at:
                start = datetime.fromisoformat(job.started_at)
                elapsed = f" ({int((datetime.now() - start).total_seconds())}s)"
            sky_info = f" [sky:{job.skypilot_job_id}]" if job.skypilot_job_id else ""
            self._print(f"  {job.name}: {job.status.value}{elapsed}{sky_info}")
        self._print("")

    def _run_local_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        job.logs_path = str(self.logs_dir / f"{job.name}.log")

        self._print(f"[{job.name}] Starting: {' '.join(job.cmd)}")

        try:
            with open(job.logs_path, "w") as log_file:
                proc = subprocess.Popen(
                    job.cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    text=True,
                )
                try:
                    proc.wait(timeout=job.timeout_s)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
                job.exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            job.exit_code = 124
            job.error = f"Timeout after {job.timeout_s}s"
        except Exception as e:
            job.exit_code = 1
            job.error = str(e)

        job.completed_at = datetime.now().isoformat()
        job.duration_s = (
            datetime.fromisoformat(job.completed_at) - datetime.fromisoformat(job.started_at)
        ).total_seconds()
        job.status = JobStatus.SUCCEEDED if job.exit_code == 0 else JobStatus.FAILED

        status_icon = "PASSED" if job.exit_code == 0 else "FAILED"
        self._print(f"[{job.name}] {status_icon} (exit_code={job.exit_code}, duration={job.duration_s:.0f}s)")

    def _launch_remote_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        self._print(f"[{job.name}] Launching remote: {' '.join(job.cmd)}")

        try:
            proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            job.logs_path = str(self.logs_dir / f"{job.name}.log")
            with open(job.logs_path, "w") as log_file:
                assert proc.stdout is not None
                last_flush = time.monotonic()
                for line in proc.stdout:
                    log_file.write(line)
                    now = time.monotonic()
                    if now - last_flush >= 0.5:
                        log_file.flush()
                        last_flush = now
                    if "Job ID:" in line:
                        job.skypilot_job_id = line.split(":")[-1].strip()
                log_file.flush()

            proc.wait()
            if proc.returncode != 0:
                job.status = JobStatus.FAILED
                job.exit_code = proc.returncode
                job.error = f"launch.py exited with code {proc.returncode}"
                self._print(f"[{job.name}] FAILED to launch (exit_code={proc.returncode})")
                return

            self._print(f"[{job.name}] Launched: job_id={job.skypilot_job_id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.exit_code = 1
            job.error = str(e)
            self._print(f"[{job.name}] FAILED to launch: {e}")

    def _poll_remote_jobs(self) -> None:
        import sky
        import sky.jobs.client.sdk as sky_jobs_sdk

        remote_running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING and j.is_remote]
        if not remote_running:
            return

        now = datetime.now()
        for job in remote_running:
            if not job.started_at:
                continue
            elapsed = (now - datetime.fromisoformat(job.started_at)).total_seconds()
            if elapsed > job.timeout_s:
                job.status = JobStatus.FAILED
                job.exit_code = 124
                job.completed_at = now.isoformat()
                job.duration_s = elapsed
                job.error = f"Timeout after {job.timeout_s}s"
                self._print(f"[{job.name}] TIMEOUT after {job.timeout_s}s")
                if job.skypilot_job_id:
                    subprocess.run(["sky", "jobs", "cancel", "-y", job.skypilot_job_id], capture_output=True)

        remote_running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING and j.is_remote]
        if not remote_running:
            return

        job_ids = [int(j.skypilot_job_id) for j in remote_running if j.skypilot_job_id]
        if not job_ids:
            return

        request_id = sky_jobs_sdk.queue(refresh=False, job_ids=job_ids)
        queue_data = sky.get(request_id)
        status_by_id = {str(j.get("job_id")): j for j in queue_data}

        for job in remote_running:
            if not job.skypilot_job_id:
                continue

            sky_job = status_by_id.get(job.skypilot_job_id)
            if not sky_job:
                elapsed = (now - datetime.fromisoformat(job.started_at)).total_seconds()
                if elapsed > 300:
                    self._complete_job(job, JobStatus.FAILED, 1, "Job not found in SkyPilot queue")
                continue

            sky_status = str(sky_job.get("status", "")).upper()
            if "SUCCEEDED" in sky_status:
                self._complete_job(job, JobStatus.SUCCEEDED, 0)
            elif any(s in sky_status for s in ("FAILED", "CANCELLED")):
                self._complete_job(job, JobStatus.FAILED, 1, f"SkyPilot status: {sky_status}")

    def _fetch_metrics_and_evaluate(self) -> None:
        jobs_needing_metrics = [j for j in self.jobs.values() if j.status == JobStatus.SUCCEEDED and j.wandb_run_name]
        if not jobs_needing_metrics:
            return

        api = wandb.Api()
        for job in jobs_needing_metrics:
            run = api.run(f"{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/{job.wandb_run_name}")
            job.metrics = dict(run.summary)

            if job.acceptance:
                job.acceptance_passed = self._evaluate_acceptance(job)
                if not job.acceptance_passed:
                    job.status = JobStatus.FAILED
                    job.error = "Acceptance criteria not met"

    def _evaluate_acceptance(self, job: Job) -> bool:
        for c in job.acceptance:
            actual = job.metrics.get(c.metric)
            if actual is None:
                return False

            if c.operator == "in":
                assert isinstance(c.threshold, tuple)
                low, high = c.threshold
                if not (low <= actual <= high):
                    return False
            else:
                assert isinstance(c.threshold, float | int)
                threshold = c.threshold
                match c.operator:
                    case ">=" if actual < threshold:
                        return False
                    case ">" if actual <= threshold:
                        return False
                    case "<=" if actual > threshold:
                        return False
                    case "<" if actual >= threshold:
                        return False
                    case "==" if actual != threshold:
                        return False

        return True


def print_summary(jobs: dict[str, Job], tail_lines: int = 50) -> bool:
    print("\n" + "=" * 60)
    print("Job Summary")
    print("=" * 60)

    succeeded = [j for j in jobs.values() if j.status == JobStatus.SUCCEEDED]
    failed = [j for j in jobs.values() if j.status == JobStatus.FAILED]
    skipped = [j for j in jobs.values() if j.status == JobStatus.SKIPPED]

    for job in jobs.values():
        icon = {"succeeded": "✅", "failed": "❌", "skipped": "⏭️", "running": "⏳", "pending": "⏸️"}
        duration = f" [{job.duration_s:.0f}s]" if job.duration_s else ""
        print(f"  {icon.get(job.status, '?')} {job.name}{duration}")
        if job.error:
            print(f"      Error: {job.error}")
        if job.wandb_url:
            print(f"      WandB: {job.wandb_url}")

    print()
    print(f"Total: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped")

    if failed:
        print("\n" + "=" * 60)
        print("Failed Job Logs")
        print("=" * 60)
        for job in failed:
            print(f"\n--- {job.name} ---")
            if job.logs_path and Path(job.logs_path).exists():
                lines = Path(job.logs_path).read_text().splitlines()
                for line in lines[-tail_lines:]:
                    print(line)
            elif job.error:
                print(job.error)
            else:
                print("(no logs available)")

        print("\n❌ VALIDATION FAILED")
        return False
    else:
        print("\n✅ VALIDATION PASSED")
        return True
