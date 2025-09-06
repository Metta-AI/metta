#!/usr/bin/env -S uv run
"""
Check logs for skypilot test jobs launched by launch_job_tests.py.

This script reads the JSON output file and displays the tail of each job's log.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from devops.skypilot.utils.job_helpers import (
    tail_job_log,
)
from metta.common.util.text_styles import bold, cyan, green, red, yellow


def display_job_logs(job_info: dict, log_content: str) -> None:
    """Display formatted job information and log tail."""
    # Header with job info
    print("\n" + "=" * 80)
    print(f"{bold('Job ID:')} {yellow(job_info['job_id'])}")
    print(f"{bold('Run Name:')} {cyan(job_info['run_name'])}")
    print(f"{bold('Nodes:')} {job_info['nodes']} | {bold('Condition:')} {job_info['condition_name']}")
    if job_info.get("ci_tests_enabled"):
        print(f"{bold('CI Tests:')} {green('Enabled')}")
    print("=" * 80)

    # Log content
    if log_content:
        print(log_content)
    else:
        print(yellow("No log content available"))

    print("\n" + "-" * 80)


def check_job_status(job_id: str) -> str:
    """Get the current status of a job."""
    try:
        cmd = ["sky", "jobs", "queue", "--job-id", job_id]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Parse the status from the output
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Skip header
                # The status is typically in the second column
                parts = lines[1].split()
                if len(parts) > 1:
                    return parts[1]
        return "UNKNOWN"
    except Exception:
        return "ERROR"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check logs for skypilot test jobs")
    parser.add_argument("--input-file", default="skypilot_test_jobs.json", help="Input JSON file with job information")
    parser.add_argument(
        "--tail-lines", type=int, default=100, help="Number of lines to tail from each log (default: 100)"
    )

    args = parser.parse_args()

    # Load job data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(red(f"Error: Input file '{input_path}' not found"))
        print("Run launch_job_tests.py first to create the job file")
        sys.exit(1)

    with open(input_path, "r") as f:
        data = json.load(f)

    # Get list of jobs to check
    launched_jobs = data.get("launched_jobs", [])

    if not launched_jobs:
        print(yellow("No launched jobs found in the input file"))
        sys.exit(0)

    # Summary header
    print(bold(f"\n=== Checking {len(launched_jobs)} Job Logs ==="))
    print(f"Input file: {input_path}")
    print(f"Tail lines: {args.tail_lines}")

    # Quick status summary first
    print(f"\n{bold('Job Status Summary:')}")
    print("-" * 60)

    status_counts = {}
    for job in launched_jobs:
        if job.get("job_id"):
            status = check_job_status(job["job_id"])
            status_counts[status] = status_counts.get(status, 0) + 1

            # Color code the status
            if status in ["RUNNING", "PROVISIONING"]:
                status_colored = cyan(status)
            elif status == "SUCCEEDED":
                status_colored = green(status)
            elif status in ["FAILED", "CANCELLED"]:
                status_colored = red(status)
            else:
                status_colored = yellow(status)

            print(f"  Job {yellow(job['job_id'])}: {status_colored} ({job['condition_name']})")

    print("-" * 60)

    # Status summary
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    # Check each job's logs
    print(f"\n{bold('Detailed Logs:')}")

    for job in launched_jobs:
        job_id = job.get("job_id")

        if not job_id:
            print(f"{yellow('Skipping job without ID:')} {job['run_name']}")
            continue

        if status == "UNKNOWN":
            print(f"{yellow('Skipping job with status UNKNOWN:')} {job['run_name']}")
            continue

        # Get the log tail
        log_content = tail_job_log(job_id, args.tail_lines)

        assert log_content
        display_job_logs(job, log_content)


if __name__ == "__main__":
    main()
