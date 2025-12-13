#!/usr/bin/env python3
"""Stable release validation CLI.

Usage:
    ./cli.py                    # Run all jobs (ci + stable)
    ./cli.py --suite=ci         # Run CI jobs only
    ./cli.py --suite=stable     # Run stable jobs only
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from devops.stable.asana_bugs import check_blockers
from devops.stable.registry import Suite, discover_jobs, specs_to_jobs
from devops.stable.runner import Job, Runner

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _is_acceptance_failed(job: Job) -> bool:
    return job.status.value == "succeeded" and job.acceptance_passed is False


def _is_failed(job: Job) -> bool:
    return job.status.value == "failed" or _is_acceptance_failed(job)


def _is_succeeded(job: Job) -> bool:
    return job.status.value == "succeeded" and job.acceptance_passed in (None, True)


def write_github_summary(runner: Runner) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    jobs = list(runner.jobs.values())
    succeeded = [j for j in jobs if _is_succeeded(j)]
    failed = [j for j in jobs if _is_failed(j)]
    skipped = [j for j in jobs if j.status.value == "skipped"]

    lines = [
        "# Stable Release Validation",
        "",
        f"**Result**: {'PASSED' if not failed else 'FAILED'}",
        f"**Jobs**: {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped",
        "",
        "## Job Results",
        "",
        "| Job | Status | Duration |",
        "|-----|--------|----------|",
    ]

    for job in runner.jobs.values():
        status = job.status.value
        if _is_acceptance_failed(job):
            status = "failed"
        duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
        lines.append(f"| {job.name} | {status.upper()} | {duration} |")

    if failed:
        lines.extend(["", "## Failed Jobs", ""])
        for job in failed:
            lines.append(f"### {job.name}")
            lines.append(f"- Exit code: {job.exit_code}")
            if job.error:
                lines.append(f"- Error: {job.error}")
            if job.wandb_url:
                lines.append(f"- WandB: {job.wandb_url}")
            lines.append("")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))


def print_failed_logs(runner: Runner, tail_lines: int = 50) -> None:
    failed = [j for j in runner.jobs.values() if _is_failed(j)]
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


def write_discord_summary(runner: Runner, state_dir: Path) -> None:
    jobs = list(runner.jobs.values())
    succeeded = [j for j in jobs if _is_succeeded(j)]
    failed = [j for j in jobs if _is_failed(j)]
    skipped = [j for j in jobs if j.status.value == "skipped"]

    lines = [
        f"**Jobs**: {len(succeeded)} passed, {len(failed)} failed, {len(skipped)} skipped",
        "",
        "```",
        f"{'Job':<45} {'Status':<8} {'Duration':<10}",
        "-" * 65,
    ]

    for job in runner.jobs.values():
        status = job.status.value
        if _is_acceptance_failed(job):
            status = "failed"
        name = job.name.split(".")[-1]
        duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
        lines.append(f"{name:<45} {status.upper():<8} {duration:<10}")

    lines.append("```")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "discord_summary.txt").write_text("\n".join(lines))


@app.callback()
def main(
    suite: Annotated[Suite | None, typer.Option(help="Which jobs to run: ci, stable, or all")] = None,
):
    """Run job validation."""
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
    state_dir = Path("devops/stable/state") / version

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

    failed = [j for j in runner.jobs.values() if _is_failed(j)]
    success = len(failed) == 0

    print_failed_logs(runner)
    write_github_summary(runner)
    write_discord_summary(runner, state_dir)

    if has_blockers:
        print("\nFAILED: Blocking bugs found in Asana")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
