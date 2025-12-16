#!/usr/bin/env python3
from __future__ import annotations

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
from devops.stable.runner import Job, Runner

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


@app.callback()
def main(
    suite: Annotated[Suite | None, typer.Option(help="Which jobs to run: ci, stable, or all")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print metrics instead of submitting to Datadog")] = False,
):
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
    try:
        metrics = jobs_to_metrics(runner.jobs)
        if metrics:
            if dry_run:
                print("\n" + "=" * 60)
                print("Datadog Metrics (DRY RUN - not submitted)")
                print("=" * 60)
                import json

                for metric in metrics:
                    print(json.dumps(metric.to_dict(), indent=2, default=str))
                print(f"\nTotal: {len(metrics)} metrics")
            else:
                logger.info("Emitting %d Datadog metrics from job results", len(metrics))
                client = DatadogMetricsClient()
                client.submit(metrics)
                logger.info("Successfully emitted Datadog metrics")
        else:
            if dry_run:
                print("\nNo Datadog metrics to emit")
            else:
                logger.debug("No Datadog metrics to emit")
    except Exception as e:
        # Don't fail the workflow if Datadog emission fails
        logger.error("Failed to emit Datadog metrics: %s", e, exc_info=True)
        if dry_run:
            print(f"\nError generating metrics: {e}")

    failed = [j for j in runner.jobs.values() if _failed(j)]
    _print_failed_logs(runner)
    _write_summary(runner, state_dir)

    if has_blockers:
        print("\nFAILED: Blocking bugs found in Asana")
        sys.exit(1)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    app()
