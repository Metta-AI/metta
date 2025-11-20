#!/usr/bin/env python3
"""Advanced monitoring and management for the worker pool."""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


class WorkerPoolMonitor:
    """Monitor and manage the worker pool via PostgreSQL."""

    def __init__(self, db_url: Optional[str] = None):
        """Initialize the monitor with database connection."""
        self.db_url = db_url or os.environ.get("POSTGRES_URL")
        if not self.db_url:
            console.print("[red]Error: POSTGRES_URL not set[/red]")
            sys.exit(1)

    def get_worker_status(self) -> list[dict]:
        """Get current status of all workers."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        worker_id,
                        hostname,
                        status,
                        current_job_id,
                        last_heartbeat,
                        started_at
                    FROM worker_status
                    WHERE last_heartbeat > NOW() - INTERVAL '10 minutes'
                    ORDER BY worker_id
                """)
                return cursor.fetchall()

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Queue status breakdown
                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM job_queue
                    GROUP BY status
                """)
                status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

                # Recent job throughput
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE completed_at > NOW() - INTERVAL '1 hour') as last_hour,
                        COUNT(*) FILTER (WHERE completed_at > NOW() - INTERVAL '10 minutes') as last_10min,
                        AVG(EXTRACT(EPOCH FROM (completed_at - claimed_at)))
                            FILTER (WHERE status = 'completed') as avg_duration_seconds
                    FROM job_queue
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                throughput = cursor.fetchone()

                return {"status_counts": status_counts, "throughput": throughput}

    def display_dashboard(self):
        """Display a live dashboard of worker pool status."""
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    workers = self.get_worker_status()
                    stats = self.get_queue_stats()

                    # Create worker table
                    worker_table = Table(title="Worker Pool Status")
                    worker_table.add_column("Worker ID", style="cyan")
                    worker_table.add_column("Status", style="green")
                    worker_table.add_column("Current Job")
                    worker_table.add_column("Last Heartbeat")
                    worker_table.add_column("Uptime")

                    now = datetime.now(timezone.utc)
                    for worker in workers:
                        heartbeat_delta = now - worker["last_heartbeat"].replace(tzinfo=timezone.utc)
                        heartbeat_str = f"{heartbeat_delta.seconds}s ago"

                        # Color code based on heartbeat age
                        if heartbeat_delta.seconds > 60:
                            heartbeat_str = f"[red]{heartbeat_str}[/red]"
                        elif heartbeat_delta.seconds > 30:
                            heartbeat_str = f"[yellow]{heartbeat_str}[/yellow]"
                        else:
                            heartbeat_str = f"[green]{heartbeat_str}[/green]"

                        uptime = now - worker["started_at"].replace(tzinfo=timezone.utc)
                        uptime_str = str(uptime).split(".")[0]  # Remove microseconds

                        status_color = {"idle": "green", "busy": "yellow", "offline": "red"}.get(
                            worker["status"], "white"
                        )

                        worker_table.add_row(
                            worker["worker_id"],
                            f"[{status_color}]{worker['status']}[/{status_color}]",
                            worker["current_job_id"] or "-",
                            heartbeat_str,
                            uptime_str,
                        )

                    # Create queue stats table
                    queue_table = Table(title="Queue Statistics")
                    queue_table.add_column("Metric", style="cyan")
                    queue_table.add_column("Value", style="white")

                    status_counts = stats["status_counts"]
                    throughput = stats["throughput"]

                    queue_table.add_row("Pending Jobs", str(status_counts.get("pending", 0)))
                    queue_table.add_row(
                        "Running Jobs", str(status_counts.get("running", 0) + status_counts.get("claimed", 0))
                    )
                    queue_table.add_row("Completed Jobs", str(status_counts.get("completed", 0)))
                    queue_table.add_row("Failed Jobs", str(status_counts.get("failed", 0)))
                    queue_table.add_row("", "")  # Spacer
                    queue_table.add_row("Jobs (Last Hour)", str(throughput["last_hour"] or 0))
                    queue_table.add_row("Jobs (Last 10 Min)", str(throughput["last_10min"] or 0))
                    if throughput["avg_duration_seconds"]:
                        avg_duration = f"{throughput['avg_duration_seconds']:.1f}s"
                    else:
                        avg_duration = "N/A"
                    queue_table.add_row("Avg Job Duration", avg_duration)

                    # Display both tables
                    console.clear()
                    console.print(worker_table)
                    console.print("")
                    console.print(queue_table)
                    console.print("\n[dim]Press Ctrl+C to exit[/dim]")

                    live.update(worker_table)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

                time.sleep(2)

    def cleanup_stale_jobs(self, older_than_hours: int = 24):
        """Clean up stale jobs from the queue."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                # Mark stale claimed jobs as failed
                cursor.execute(
                    """
                    UPDATE job_queue
                    SET status = 'failed',
                        error_message = 'Job staled (no progress)',
                        completed_at = NOW()
                    WHERE status IN ('claimed', 'running')
                    AND claimed_at < NOW() - INTERVAL '%s hours'
                """,
                    (older_than_hours,),
                )

                stale_count = cursor.rowcount

                # Delete old completed/failed jobs
                cursor.execute("""
                    DELETE FROM job_queue
                    WHERE status IN ('completed', 'failed')
                    AND completed_at < NOW() - INTERVAL '7 days'
                """)

                deleted_count = cursor.rowcount
                conn.commit()

        console.print(f"[green]Marked {stale_count} stale jobs as failed[/green]")
        console.print(f"[green]Deleted {deleted_count} old jobs[/green]")

    def requeue_failed_jobs(self):
        """Requeue failed jobs for retry."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE job_queue
                    SET status = 'pending',
                        worker_id = NULL,
                        claimed_at = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        retry_count = retry_count + 1
                    WHERE status = 'failed'
                    AND retry_count < 3
                    RETURNING job_id
                """)

                requeued = cursor.fetchall()
                conn.commit()

        console.print(f"[green]Requeued {len(requeued)} failed jobs[/green]")
        for job_id in requeued:
            console.print(f"  - {job_id[0]}")


def main():
    """Main entry point for the pool monitor."""
    parser = argparse.ArgumentParser(description="Worker Pool Monitor and Manager")
    parser.add_argument("command", choices=["dashboard", "cleanup", "requeue"], help="Command to execute")
    parser.add_argument("--db-url", help="PostgreSQL connection URL (or set POSTGRES_URL env var)")

    args = parser.parse_args()

    monitor = WorkerPoolMonitor(db_url=args.db_url)

    if args.command == "dashboard":
        monitor.display_dashboard()
    elif args.command == "cleanup":
        monitor.cleanup_stale_jobs()
    elif args.command == "requeue":
        monitor.requeue_failed_jobs()


if __name__ == "__main__":
    main()
