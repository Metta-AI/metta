"""Stable job runner used by GitHub Actions."""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import sky
import sky.jobs.client.sdk as sky_jobs_sdk
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
    metric_name: str | None = None


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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_s: float | None = None
    logs_path: str | None = None
    skypilot_job_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    acceptance_passed: bool | None = None
    criterion_results: dict[str, bool] = Field(default_factory=dict)
    error: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.remote_gpus is not None or self.remote_nodes is not None

    @property
    def wandb_url(self) -> str | None:
        if not self.wandb_run_name:
            return None
        return f"https://wandb.ai/{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/runs/{self.wandb_run_name}"


def _now() -> datetime:
    return datetime.now()


def _duration_s(started_at: datetime, completed_at: datetime) -> float:
    return (completed_at - started_at).total_seconds()


class _AcceptanceEvaluator:
    def __init__(self) -> None:
        self._api: wandb.Api | None = None

    def evaluate(self, job: Job) -> None:
        if job.status != JobStatus.SUCCEEDED:
            return
        if not job.wandb_run_name:
            return
        if self._api is None:
            self._api = wandb.Api()
        try:
            run = self._api.run(f"{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/{job.wandb_run_name}")
            job.metrics = dict(run.summary)
        except Exception as e:
            job.acceptance_passed = False
            job.error = f"WandB fetch failed: {e}"
            return

        if not job.acceptance:
            return
        job.acceptance_passed = self._passes_acceptance(job)
        if not job.acceptance_passed:
            job.error = "Acceptance criteria not met"

    def _passes_acceptance(self, job: Job) -> bool:
        for c in job.acceptance:
            actual = job.metrics.get(c.metric)
            if actual is None:
                job.criterion_results[c.metric] = False
                return False

            passed = False
            if c.operator == "in":
                assert isinstance(c.threshold, tuple)
                low, high = c.threshold
                passed = low <= actual <= high
            else:
                assert isinstance(c.threshold, (int, float))
                threshold = float(c.threshold)
                match c.operator:
                    case ">=":
                        passed = actual >= threshold
                    case ">":
                        passed = actual > threshold
                    case "<=":
                        passed = actual <= threshold
                    case "<":
                        passed = actual < threshold
                    case "==":
                        passed = actual == threshold

            job.criterion_results[c.metric] = passed
            if not passed:
                return False
        return True


class _JobHandle(Protocol):
    @property
    def futures(self) -> list[Future]: ...


@dataclass(frozen=True)
class _LocalHandle:
    future: Future[int]

    @property
    def futures(self) -> list[Future]:
        return [self.future]


@dataclass(frozen=True)
class _RemoteHandle:
    launch_future: Future[str | None]

    @property
    def futures(self) -> list[Future]:
        return [self.launch_future]


class Runner:
    _STATUS_EVERY = timedelta(minutes=10)
    _REMOTE_POLL_EVERY_S = 5.0
    _REMOTE_NOT_FOUND_GRACE = timedelta(minutes=5)

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.state_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.jobs: dict[str, Job] = {}
        self._handles: dict[str, _JobHandle] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._output_lock = threading.Lock()
        self._acceptance = _AcceptanceEvaluator()

        self._dependents: dict[str, list[str]] = {}
        self._remaining_deps: dict[str, int] = {}

        self._remote_not_found_deadline: dict[str, datetime] = {}

    def add_job(self, job: Job) -> None:
        self.jobs[job.name] = job

    def run_all(self) -> dict[str, Job]:
        self._executor = ThreadPoolExecutor(max_workers=min(32, max(1, len(self.jobs))))
        self._build_dependency_index()

        last_status = _now()
        next_remote_poll = time.monotonic()

        try:
            while self._has_incomplete_jobs():
                self._start_ready_jobs()

                if _now() - last_status >= self._STATUS_EVERY:
                    self._print_status_summary()
                    last_status = _now()

                self._drain_completed_futures()

                if self._has_remote_running():
                    now_mono = time.monotonic()
                    if now_mono >= next_remote_poll:
                        self._poll_remote_jobs()
                        next_remote_poll = now_mono + self._REMOTE_POLL_EVERY_S

                self._wait_for_progress(next_remote_poll)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._executor = None

        return self.jobs

    def _build_dependency_index(self) -> None:
        self._dependents = {name: [] for name in self.jobs}
        self._remaining_deps = {name: 0 for name in self.jobs}

        for job in self.jobs.values():
            missing = [d for d in job.dependencies if d not in self.jobs]
            if missing:
                self._skip_job(job, f"Missing dependency: {missing[0]}")
                continue

            self._remaining_deps[job.name] = len(job.dependencies)
            for dep in job.dependencies:
                self._dependents[dep].append(job.name)

        for job in self.jobs.values():
            if job.status != JobStatus.NOT_STARTED:
                continue
            if any(self.jobs[d].status in (JobStatus.FAILED, JobStatus.SKIPPED) for d in job.dependencies):
                dep = next(d for d in job.dependencies if self.jobs[d].status in (JobStatus.FAILED, JobStatus.SKIPPED))
                self._skip_job(job, f"Dependency {self.jobs[dep].status.value.lower()}: {dep}")

    def _has_incomplete_jobs(self) -> bool:
        return any(j.status in (JobStatus.NOT_STARTED, JobStatus.RUNNING) for j in self.jobs.values())

    def _has_remote_running(self) -> bool:
        return any(j.status == JobStatus.RUNNING and j.is_remote for j in self.jobs.values())

    def _ready_jobs(self) -> list[Job]:
        ready = []
        for job in self.jobs.values():
            if job.status != JobStatus.NOT_STARTED:
                continue
            if self._remaining_deps.get(job.name, 0) != 0:
                continue
            ready.append(job)
        return ready

    def _start_ready_jobs(self) -> None:
        for job in self._ready_jobs():
            self._start_job(job)

    def _start_job(self, job: Job) -> None:
        assert self._executor is not None

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        job.logs_path = str(self.logs_dir / f"{job.name}.log")

        if job.is_remote:
            self._print(f"[{job.name}] Launching remote: {' '.join(job.cmd)}")
            launch_future = self._executor.submit(_run_remote_launch, job.cmd, Path(job.logs_path))
            self._handles[job.name] = _RemoteHandle(launch_future=launch_future)
        else:
            self._print(f"[{job.name}] Starting: {' '.join(job.cmd)}")
            future = self._executor.submit(_run_local_cmd, job.cmd, Path(job.logs_path), job.timeout_s)
            self._handles[job.name] = _LocalHandle(future=future)

    def _drain_completed_futures(self) -> None:
        completed: list[tuple[str, Future]] = []
        for name, handle in list(self._handles.items()):
            for f in handle.futures:
                if f.done():
                    completed.append((name, f))
        for name, future in completed:
            self._handles.pop(name, None)
            self._on_future_done(self.jobs[name], future)

    def _on_future_done(self, job: Job, future: Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            self._finish(job, JobStatus.FAILED, 1, str(e))
            return

        if job.is_remote:
            job_id = result
            if not job_id:
                self._finish(job, JobStatus.FAILED, 1, "Remote launch succeeded but no job id was found")
                return
            job.skypilot_job_id = job_id
            self._remote_not_found_deadline[job.name] = _now() + self._REMOTE_NOT_FOUND_GRACE
            self._print(f"[{job.name}] Launched: job_id={job.skypilot_job_id}")
            return

        exit_code = int(result)
        status = JobStatus.SUCCEEDED if exit_code == 0 else JobStatus.FAILED
        error = f"Timeout after {job.timeout_s}s" if exit_code == 124 else None
        self._finish(job, status, exit_code, error=error)

    def _poll_remote_jobs(self) -> None:
        running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING and j.is_remote]
        if not running:
            return

        now = _now()

        # Check timeouts
        for job in running:
            if job.started_at and now - job.started_at > timedelta(seconds=job.timeout_s):
                self._finish(job, JobStatus.FAILED, 124, f"Timeout after {job.timeout_s}s")
                if job.skypilot_job_id:
                    subprocess.run(["sky", "jobs", "cancel", "-y", job.skypilot_job_id], capture_output=True)

        running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING and j.is_remote]
        job_ids = [int(j.skypilot_job_id) for j in running if j.skypilot_job_id]
        if not job_ids:
            return

        # Poll SkyPilot - if it fails, just log and retry next cycle
        # Timeouts above are the safety net for stuck jobs
        try:
            request_id = sky_jobs_sdk.queue(refresh=False, job_ids=job_ids)
            queue_data = sky.get(request_id)
            if not isinstance(queue_data, list):
                raise TypeError(f"Expected list, got {type(queue_data).__name__}")
            status_by_id = {
                str(item.get("job_id") if isinstance(item, dict) else getattr(item, "job_id", None)): item
                for item in queue_data
            }
        except Exception as e:
            self._print(f"[WARN] SkyPilot poll failed (will retry): {e}")
            return

        for job in running:
            if not job.skypilot_job_id:
                continue

            sky_job = status_by_id.get(job.skypilot_job_id)
            if not sky_job:
                deadline = self._remote_not_found_deadline.get(job.name)
                if deadline and now >= deadline:
                    self._finish(job, JobStatus.FAILED, 1, "Job not found in SkyPilot queue")
                continue

            status = sky_job.get("status") if isinstance(sky_job, dict) else getattr(sky_job, "status", None)
            sky_status = str(status or "").upper()
            if "SUCCEEDED" in sky_status:
                self._finish(job, JobStatus.SUCCEEDED, 0)
            elif any(s in sky_status for s in ("FAILED", "CANCELLED")):
                self._finish(job, JobStatus.FAILED, 1, f"SkyPilot status: {sky_status}")

    def _finish(self, job: Job, status: JobStatus, exit_code: int, error: str | None = None) -> None:
        job.status = status
        job.exit_code = exit_code
        job.error = error
        job.completed_at = _now()
        if job.started_at:
            job.duration_s = _duration_s(job.started_at, job.completed_at)

        if status == JobStatus.SUCCEEDED:
            self._acceptance.evaluate(job)
            if job.acceptance_passed is False:
                result = f"FAILED: {job.error or 'Acceptance criteria not met'}"
            else:
                result = "PASSED"
        else:
            result = f"FAILED: {job.error or exit_code}"

        duration = f"{job.duration_s:.0f}s" if job.duration_s is not None else "-"
        self._print(f"[{job.name}] {result} (duration={duration})")

        self._on_job_terminal(job)

    def _on_job_terminal(self, job: Job) -> None:
        for dependent_name in self._dependents.get(job.name, []):
            dependent = self.jobs[dependent_name]
            if dependent.status != JobStatus.NOT_STARTED:
                continue

            if job.status in (JobStatus.FAILED, JobStatus.SKIPPED):
                self._skip_job(dependent, f"Dependency {job.status.value.lower()}: {job.name}")
                continue

            self._remaining_deps[dependent_name] -= 1

    def _skip_job(self, job: Job, reason: str) -> None:
        job.status = JobStatus.SKIPPED
        job.exit_code = 0
        job.error = reason
        self._print(f"[{job.name}] SKIPPED: {reason}")
        self._on_job_terminal(job)

    def _wait_for_progress(self, next_remote_poll: float) -> None:
        futures: list[Future] = []
        for handle in self._handles.values():
            futures.extend(handle.futures)
        if not futures:
            time.sleep(max(0.0, min(0.5, next_remote_poll - time.monotonic())))
            return

        timeout = max(0.0, min(0.5, next_remote_poll - time.monotonic()))
        wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)

    def _print(self, msg: str) -> None:
        with self._output_lock:
            print(msg, flush=True)

    def _print_status_summary(self) -> None:
        ts = _now().strftime("%Y-%m-%d %H:%M:%S")
        self._print(f"\n[{ts}] === Status Summary ===")
        now = _now()
        for job in self.jobs.values():
            elapsed = ""
            if job.started_at and job.status == JobStatus.RUNNING:
                elapsed = f" ({int((now - job.started_at).total_seconds())}s)"
            sky = f" [sky:{job.skypilot_job_id}]" if job.skypilot_job_id else ""
            self._print(f"  {job.name}: {job.status.value}{elapsed}{sky}")
        self._print("")


def _run_local_cmd(cmd: list[str], log_path: Path, timeout_s: int) -> int:
    with open(log_path, "w") as log_file:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            return 124


def _run_remote_launch(cmd: list[str], log_path: Path) -> str | None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    job_id: str | None = None
    with open(log_path, "w") as log_file:
        assert proc.stdout is not None
        last_flush = time.monotonic()
        for line in proc.stdout:
            log_file.write(line)
            now = time.monotonic()
            if now - last_flush >= 0.5:
                log_file.flush()
                last_flush = now
            if "Job ID:" in line:
                job_id = line.split(":")[-1].strip()
        log_file.flush()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"launch.py exited with code {proc.returncode}")
    return job_id
