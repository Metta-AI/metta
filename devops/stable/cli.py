#!/usr/bin/env python3
"""Stable release validation CLI.

Usage:
    ./cli.py                         # Run full validation
    ./cli.py --version=v1.0.0        # Use specific version
    ./cli.py --job arena             # Filter jobs by name
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import typer

from devops.stable.runner import Runner, print_summary
from devops.stable.suite import get_all_jobs

app = typer.Typer(add_completion=False, invoke_without_command=True)


def get_state_dir(version: str) -> Path:
    return Path("devops/stable/state") / version


def generate_version() -> str:
    return datetime.now().strftime("%Y.%m.%d-%H%M%S")


def write_github_summary(runner: Runner) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    succeeded = [j for j in runner.jobs.values() if j.status.value == "succeeded"]
    failed = [j for j in runner.jobs.values() if j.status.value == "failed"]
    skipped = [j for j in runner.jobs.values() if j.status.value == "skipped"]

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
    succeeded = [j for j in runner.jobs.values() if j.status.value == "succeeded"]
    failed = [j for j in runner.jobs.values() if j.status.value == "failed"]
    skipped = [j for j in runner.jobs.values() if j.status.value == "skipped"]

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
def release(
    version: str = typer.Option(None, help="Version string (default: timestamp)"),
    job: str = typer.Option(None, help="Filter jobs by name pattern"),
    no_interactive: bool = typer.Option(False, "--no-interactive", help="Non-interactive mode"),
    skip_commit_match: bool = typer.Option(False, "--skip-commit-match", help="Skip commit verification"),
):
    """Run stable release validation."""
    version = version or generate_version()
    user = os.environ.get("USER", "unknown")
    prefix = f"{user}.stable.{version}"
    state_dir = get_state_dir(version)

    runner = Runner(state_dir)
    jobs = get_all_jobs(prefix)
    if job:
        jobs = [j for j in jobs if job in j.name]
    for j in jobs:
        runner.add_job(j)

    print(f"Running stable release validation: {version}")
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
