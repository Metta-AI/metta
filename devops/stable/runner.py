"""Job runner for stable release validation and CI."""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import wandb
from pydantic import BaseModel

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class Operator(StrEnum):
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"
    EQ = "=="


class AcceptanceCriterion(BaseModel):
    metric: str
    threshold: float
    operator: Operator = Operator.GE


class Job(BaseModel):
    name: str
    cmd: list[str]
    timeout_s: int = 3600
    remote: dict | None = None
    dependencies: list[str] = []
    acceptance: list[AcceptanceCriterion] = []
    wandb_run_name: str | None = None

    status: JobStatus = JobStatus.PENDING
    exit_code: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    logs_path: str | None = None
    skypilot_job_id: str | None = None
    metrics: dict[str, float] = {}
    acceptance_passed: bool | None = None
    error: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.remote is not None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.SKIPPED)

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

    def add_job(self, job: Job) -> None:
        self.jobs[job.name] = job

    def run_all(self) -> dict[str, Job]:
        """Run all jobs, respecting dependencies. Returns final job states."""
        while True:
            pending = [j for j in self.jobs.values() if j.status == JobStatus.PENDING]
            running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]

            if not pending and not running:
                break

            ready = self._get_ready_jobs(pending)
            if ready:
                self._start_jobs(ready)

            if running:
                self._poll_remote_jobs()

            if not ready and running:
                time.sleep(5)

        self._fetch_metrics_and_evaluate()
        return self.jobs

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
                    job.error = f"Dependency failed: {dep_name}"
                    deps_satisfied = False
                    break
                if not dep.is_terminal:
                    deps_satisfied = False
                    break
            if deps_satisfied and job.status == JobStatus.PENDING:
                ready.append(job)
        return ready

    def _start_jobs(self, jobs: list[Job]) -> None:
        local_jobs = [j for j in jobs if not j.is_remote]
        remote_jobs = [j for j in jobs if j.is_remote]

        with ThreadPoolExecutor(max_workers=len(jobs) or 1) as executor:
            futures = {}
            for job in local_jobs:
                futures[executor.submit(self._run_local_job, job)] = job
            for job in remote_jobs:
                futures[executor.submit(self._launch_remote_job, job)] = job

            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)

    def _run_local_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        job.logs_path = str(self.logs_dir / f"{job.name}.log")

        print(f"[{job.name}] Starting: {' '.join(job.cmd)}")

        try:
            with open(job.logs_path, "w") as log_file:
                result = subprocess.run(
                    job.cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=job.timeout_s,
                    cwd=os.getcwd(),
                )
                job.exit_code = result.returncode
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

        print(f"[{job.name}] exit_code={job.exit_code}")

    def _launch_remote_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        remote = job.remote or {}
        gpus = remote.get("gpus", 1)

        print(f"[{job.name}] Launching remote: gpus={gpus}")

        try:
            result = subprocess.run(
                ["sky", "jobs", "launch", "-y", "-n", job.name, "--gpus", f"A100:{gpus}", "--"] + job.cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                job.status = JobStatus.FAILED
                job.exit_code = result.returncode
                job.error = result.stderr
                return

            for line in result.stdout.split("\n"):
                if "Job ID:" in line:
                    job.skypilot_job_id = line.split(":")[-1].strip()
                    break

            print(f"[{job.name}] Launched: job_id={job.skypilot_job_id}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.exit_code = 1
            job.error = str(e)

    def _poll_remote_jobs(self) -> None:
        remote_running = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING and j.is_remote]
        if not remote_running:
            return

        try:
            result = subprocess.run(
                ["sky", "jobs", "queue", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return

            queue_data = json.loads(result.stdout) if result.stdout else []
            status_by_id = {str(j.get("job_id")): j for j in queue_data}

            for job in remote_running:
                if not job.skypilot_job_id:
                    continue

                sky_job = status_by_id.get(job.skypilot_job_id)
                if not sky_job:
                    # Job disappeared from queue - mark as failed
                    job.status = JobStatus.FAILED
                    job.exit_code = 1
                    job.completed_at = datetime.now().isoformat()
                    job.error = "Job disappeared from SkyPilot queue"
                else:
                    sky_status = sky_job.get("status", "").upper()
                    if sky_status == "SUCCEEDED":
                        job.status = JobStatus.SUCCEEDED
                        job.exit_code = 0
                        job.completed_at = datetime.now().isoformat()
                    elif sky_status in ("FAILED", "FAILED_SETUP", "CANCELLED"):
                        job.status = JobStatus.FAILED
                        job.exit_code = 1
                        job.completed_at = datetime.now().isoformat()
                        job.error = f"SkyPilot status: {sky_status}"

                if job.completed_at and job.started_at:
                    job.duration_s = (
                        datetime.fromisoformat(job.completed_at) - datetime.fromisoformat(job.started_at)
                    ).total_seconds()

        except Exception as e:
            print(f"Warning: Failed to poll SkyPilot jobs: {e}")

    def _fetch_metrics_and_evaluate(self) -> None:
        for job in self.jobs.values():
            if job.status != JobStatus.SUCCEEDED:
                continue
            if not job.wandb_run_name:
                continue

            try:
                api = wandb.Api()
                run = api.run(f"{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/{job.wandb_run_name}")
                job.metrics = dict(run.summary)
            except Exception as e:
                print(f"Warning: Failed to fetch metrics for {job.name}: {e}")

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

            match c.operator:
                case Operator.GE if actual < c.threshold:
                    return False
                case Operator.GT if actual <= c.threshold:
                    return False
                case Operator.LE if actual > c.threshold:
                    return False
                case Operator.LT if actual >= c.threshold:
                    return False
                case Operator.EQ if actual != c.threshold:
                    return False

        return True


def create_job(
    name: str,
    cmd: list[str],
    timeout_s: int = 3600,
    gpus: int | None = None,
    nodes: int = 1,
    dependencies: list[str] | None = None,
    acceptance: list[AcceptanceCriterion] | None = None,
    wandb_run_name: str | None = None,
) -> Job:
    remote = {"gpus": gpus, "nodes": nodes} if gpus else None
    return Job(
        name=name,
        cmd=cmd,
        timeout_s=timeout_s,
        remote=remote,
        dependencies=dependencies or [],
        acceptance=acceptance or [],
        wandb_run_name=wandb_run_name,
    )


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
