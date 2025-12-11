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
from devops.stable.runner import Job, Runner, print_summary
from metta.common.util.collections import group_by

app = typer.Typer(add_completion=False, invoke_without_command=True)


def get_state_dir(version: str) -> Path:
    return Path("devops/stable/state") / version


def generate_version() -> str:
    return datetime.now().strftime("%Y.%m.%d-%H%M%S")


def _jobs_by_status(jobs: list[Job]) -> dict[str, list[Job]]:
    return group_by(jobs, lambda j: j.status.value)


def write_github_summary(runner: Runner) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    by_status = _jobs_by_status(list(runner.jobs.values()))
    succeeded, failed, skipped = by_status["succeeded"], by_status["failed"], by_status["skipped"]

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
        icon = {"succeeded": "✅", "failed": "❌", "skipped": "⏭️"}.get(job.status.value, "❓")
        duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
        lines.append(f"| {job.name} | {icon} | {duration} |")

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


def write_discord_summary(runner: Runner, state_dir: Path) -> None:
    by_status = _jobs_by_status(list(runner.jobs.values()))
    succeeded, failed, skipped = by_status["succeeded"], by_status["failed"], by_status["skipped"]

    lines = [
        f"**Jobs**: {len(succeeded)} passed, {len(failed)} failed, {len(skipped)} skipped",
        "",
        "```",
        f"{'Job':<45} {'Status':<8} {'Duration':<10}",
        "-" * 65,
    ]

    for job in runner.jobs.values():
        icon = {"succeeded": "✅", "failed": "❌", "skipped": "⏭️"}.get(job.status.value, "❓")
        name = job.name.split(".")[-1]
        duration = f"{job.duration_s:.0f}s" if job.duration_s else "-"
        lines.append(f"{name:<45} {icon:<8} {duration:<10}")

    lines.append("```")

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "discord_summary.txt").write_text("\n".join(lines))


@app.callback()
def main(
    suite: Annotated[Suite | None, typer.Option(help="Which jobs to run: ci, stable, or all")] = None,
):
    """Run job validation."""
    if suite != Suite.CI:
        print("Checking for blocking bugs in Asana...")
        blocker_result = check_blockers()
        if blocker_result is False:
            print("\nBlocking bugs found. Fix them before releasing.")
            sys.exit(1)
        elif blocker_result is None:
            print("Asana check skipped (not configured or unavailable)")
        print()

    version = generate_version()
    user = os.environ.get("USER", "unknown")
    prefix = f"{user}.{suite or 'all'}.{version}"
    state_dir = get_state_dir(version)

    specs = discover_jobs(suite)
    jobs = specs_to_jobs(specs, prefix)

    runner = Runner(state_dir)
    for j in jobs:
        runner.add_job(j)

    print(f"Running {suite} jobs: {version}")
    print(f"State: {state_dir}")
    print(f"Logs: {runner.logs_dir}")
    print(f"Jobs: {len(runner.jobs)}")
    print()

    for j in runner.jobs.values():
        remote = "remote" if j.is_remote else "local"
        print(f"  - {j.name} ({remote})")
    print()

    runner.run_all()
    success = print_summary(runner.jobs)
    write_github_summary(runner)
    write_discord_summary(runner, state_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
