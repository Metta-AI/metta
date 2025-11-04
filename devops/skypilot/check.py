#!/usr/bin/env python3

import datetime

import sky
import sky.jobs
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

NOW = datetime.datetime.now(datetime.timezone.utc)
TOP_N = 10
console = Console()


def human_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def get_running_clusters():
    request_id = sky.status(all_users=True)
    clusters = sky.get(request_id)
    running_clusters = []
    for c in clusters:
        if c.get("status") == sky.ClusterStatus.UP:
            launched_ts = c.get("launched_at")
            if launched_ts is None:
                continue
            launched_at = datetime.datetime.fromtimestamp(launched_ts, tz=datetime.timezone.utc)
            age = NOW - launched_at
            running_clusters.append(
                {
                    "name": c["name"],
                    "user": c["user_name"],
                    "launched_at": launched_at,
                    "age": age,
                }
            )
    running_clusters.sort(key=lambda x: x["age"], reverse=True)
    return running_clusters


def get_running_jobs():
    request_id = sky.jobs.queue(refresh=False, all_users=True)
    jobs = sky.get(request_id)
    running_jobs = []
    for job in jobs:
        if job.get("status") == sky.jobs.ManagedJobStatus.RUNNING:
            submitted_at = datetime.datetime.fromtimestamp(job["submitted_at"], tz=datetime.timezone.utc)
            duration_seconds = job.get("job_duration", (NOW - submitted_at).total_seconds())
            running_jobs.append(
                {
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "user": job.get("user_name", "unknown"),
                    "submitted_at": submitted_at,
                    "duration_seconds": duration_seconds,
                }
            )
    running_jobs.sort(key=lambda x: x["duration_seconds"], reverse=True)
    return running_jobs


def print_cluster_table(clusters):
    if not clusters:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("NAME", style="cyan", no_wrap=True)
    table.add_column("USER", style="green")
    table.add_column("LAUNCHED", justify="right")
    table.add_column("AGE", justify="right")

    for c in clusters:
        table.add_row(c["name"], c["user"], c["launched_at"].strftime("%Y-%m-%d %H:%M"), str(c["age"]).split(".")[0])

    aligned_table = Align.center(table)

    panel = Panel(
        aligned_table,
        title=f"[bold bright_blue]🚀 Running Clusters — Top {len(clusters)} by Age[/bold bright_blue]",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()


def print_job_table(jobs):
    if not jobs:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID/NAME", style="cyan")
    table.add_column("USER", style="green")
    table.add_column("SUBMITTED", justify="right")
    table.add_column("DURATION", justify="right")

    for j in jobs:
        job_id_name = f"{j['job_id']}/{j['job_name']}"
        table.add_row(
            job_id_name,
            j["user"],
            j["submitted_at"].strftime("%Y-%m-%d %H:%M"),
            human_duration(j["duration_seconds"]),
        )

    aligned_table = Align.center(table)

    panel = Panel(
        aligned_table,
        title=f"[bold bright_blue]⏱️ Running Jobs — Top {len(jobs)} by Duration[/bold bright_blue]",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(panel)
    console.print()


def main():
    console.rule("[bold blue]SkyPilot Audit Report")
    console.print(f"[dim]Generated at {NOW.isoformat()}[/dim]\n")

    clusters = get_running_clusters()
    print_cluster_table(clusters[:TOP_N])

    jobs = get_running_jobs()
    print_job_table(jobs[:TOP_N])


if __name__ == "__main__":
    main()
