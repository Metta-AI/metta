#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from devops.datadog.datadog_client import DatadogMetricsClient
from devops.stable.asana_bugs import check_blockers
from devops.stable.datadog_metrics import jobs_to_metrics
from devops.stable.registry import Suite, discover_jobs, specs_to_jobs
from devops.stable.runner import Job, JobStatus, Runner

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _failed(job: Job) -> bool:
    return job.status.value == "failed" or (job.status.value == "succeeded" and job.acceptance_passed is False)


def _status(job: Job) -> str:
    if _failed(job):
        return "FAILED"
    return job.status.value.upper()


def _write_summary(runner: Runner, state_dir: Path) -> None:
    jobs = list(runner.jobs.values())
    failed = [j for j in jobs if _failed(j)]
    passed = [j for j in jobs if j.status.value == "succeeded" and not _failed(j)]
    skipped = [j for j in jobs if j.status.value == "skipped"]

    header = f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped"
    table = []
    for job in jobs:
        name = job.name.split(".")[-1]
        duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
        table.append(f"{name:<40} {_status(job):<10} {duration}")

    # Discord summary (file)
    state_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"**Jobs**: {header}", "", "```", *table, "```"]
    (state_dir / "discord_summary.txt").write_text("\n".join(lines))

    # GitHub summary (env var)
    gh_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_path:
        md = [
            "# Stable Release Validation",
            "",
            f"**Result**: {'PASSED' if not failed else 'FAILED'}",
            f"**Jobs**: {header}",
            "",
            "| Job | Status | Duration |",
            "|-----|--------|----------|",
        ]
        for job in jobs:
            duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
            md.append(f"| {job.name} | {_status(job)} | {duration} |")
        with open(gh_path, "a") as f:
            f.write("\n".join(md))


def _print_failed_logs(runner: Runner, tail_lines: int = 50) -> None:
    failed = [j for j in runner.jobs.values() if _failed(j)]
    if not failed:
        return

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


def _exit_details(runner: Runner, has_blockers: bool) -> tuple[int, list[str]]:
    jobs = list(runner.jobs.values())
    job_failures = [j for j in jobs if j.status == JobStatus.FAILED]
    acceptance_failures = [
        j for j in jobs if j.status == JobStatus.SUCCEEDED and j.acceptance_passed is False
    ]

    exit_code = 0
    if job_failures:
        exit_code |= 1
    if acceptance_failures:
        exit_code |= 2
    if has_blockers:
        exit_code |= 4

    details: list[str] = []
    if job_failures:
        details.append(f"Job failures: {', '.join(j.name for j in job_failures)}")
    if acceptance_failures:
        details.append(f"Acceptance failures: {', '.join(j.name for j in acceptance_failures)}")
    if has_blockers:
        details.append("Blocking bugs found in Asana")

    return exit_code, details


@app.callback()
def main(
    suite: Annotated[Suite | None, typer.Option(help="Which jobs to run: ci, stable, or all")] = None,
    skip_submitting_metrics: Annotated[
        bool, typer.Option("--skip-submitting-metrics", help="Skip submitting metrics to Datadog")
    ] = False,
    dump_metrics: Annotated[
        Path | None, typer.Option("--dump-metrics", help="Write metrics JSON to file for inspection")
    ] = None,
):
    # Get Datadog client up front to fail fast in case of missing credentials
    datadog_client: DatadogMetricsClient | None = None
    if not skip_submitting_metrics:
        datadog_client = DatadogMetricsClient()

    has_blockers = False
    if suite != Suite.CI:
        print("Checking for blocking bugs in Asana...")
        blocker_result = check_blockers()
        if blocker_result is False:
            print("\nBlocking bugs found. Will fail after running jobs.")
            has_blockers = True
        elif blocker_result is None:
            print("Asana check skipped (not configured or unavailable)")
        print()

    version = datetime.now().strftime("%Y.%m.%d-%H%M%S")
    user = os.environ.get("USER", "unknown")
    prefix = f"{user}.{suite or 'all'}.{version}"
    state_dir = Path("devops/stable/state")

    specs = discover_jobs(suite)
    jobs = specs_to_jobs(specs, prefix)

    runner = Runner(state_dir)
    for j in jobs:
        runner.add_job(j)

    print(f"Running {suite} jobs: {version}")
    print(f"Jobs: {len(runner.jobs)}")
    for j in runner.jobs.values():
        remote = "remote" if j.is_remote else "local"
        print(f"  - {j.name} ({remote})")

    runner.run_all()

    # Emit Datadog metrics for completed jobs
    metrics = jobs_to_metrics(runner.jobs)
    if metrics:
        # Dump metrics to file if requested
        if dump_metrics:
            payload = [m.to_dict() for m in metrics]
            dump_metrics.parent.mkdir(parents=True, exist_ok=True)
            dump_metrics.write_text(json.dumps(payload, indent=2))
            print(f"\nWrote {len(metrics)} metrics to {dump_metrics}")

        if not skip_submitting_metrics:
            logger.info("Emitting %d Datadog metrics from job results", len(metrics))
            assert datadog_client is not None
            datadog_client.submit(metrics)
            logger.info("Successfully emitted Datadog metrics")
        else:
            logger.info("Skipping submission of %d Datadog metrics from job results", len(metrics))
    else:
        logger.debug("No metrics")

    failed = [j for j in runner.jobs.values() if _failed(j)]
    _print_failed_logs(runner)
    _write_summary(runner, state_dir)

    exit_code, details = _exit_details(runner, has_blockers)
    if exit_code == 0:
        sys.exit(0)

    print("\nFAILED: Stable release validation did not pass")
    if details:
        for detail in details:
            print(f"- {detail}")
    print("Exit code meanings: 1=job failures, 2=acceptance failures, 4=blocking bugs")
    sys.exit(exit_code)


if __name__ == "__main__":
    app()
