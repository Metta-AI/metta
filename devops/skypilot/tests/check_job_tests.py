#!/usr/bin/env -S uv run
"""
Check logs for skypilot test jobs launched by launch_job_tests.py.

This script reads the JSON output file and displays the tail of each job's log.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from devops.skypilot.utils.job_helpers import check_job_statuses, tail_job_log
from metta.common.util.text_styles import bold, cyan, green, red, yellow


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check logs for skypilot test jobs")
    parser.add_argument(
        "-f", "--input-file", default="skypilot_test_jobs.json", help="Input JSON file with job information"
    )
    parser.add_argument(
        "-n", "--tail-lines", type=int, default=100, help="Number of lines to tail from each log (default: 100)"
    )
    parser.add_argument("-s", "--status-only", action="store_true", help="Only show job status, not logs")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.basename(args.input_file)
    input_path = Path(script_dir) / input_filename
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
    print(bold(f"\n=== Checking {len(launched_jobs)} Job Logs from {input_path}==="))

    # Get all job IDs and check their statuses
    job_ids = [int(job["job_id"]) for job in launched_jobs if job.get("job_id")]
    job_statuses = check_job_statuses(job_ids)

    # Quick status summary first
    print(f"\n{bold('Job Status Summary:')}")
    print("-" * 60)

    status_counts = {}
    for job in launched_jobs:
        if job.get("job_id"):
            job_id = int(job["job_id"])
            job_info = job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
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

    if args.status_only:
        return

    # Check each job's logs
    print(f"\n{bold('Detailed Logs:')}")

    for job in launched_jobs:
        job_id_str = job.get("job_id")

        if not job_id_str:
            print(f"\n{yellow('Skipping job without ID:')} {job['run_name']}")
            continue

        job_id = int(job_id_str)
        job_info = job_statuses.get(job_id, {})
        status = job_info.get("status", "UNKNOWN")

        # Display job header
        print("\n" + "=" * 80)
        print(f"{bold('Job ID:')} {yellow(job_id_str)} ({status})")
        print(f"{bold('Run Name:')} {cyan(job['run_name'])}")
        print(f"{bold('Nodes:')} {job['nodes']} | {bold('Condition:')} {job['condition_name']}")
        if job.get("ci_tests_enabled"):
            print(f"{bold('CI Tests:')} {green('Enabled')}")
        print("=" * 80)

        # Get and display log content
        log_content = tail_job_log(job_id_str, args.tail_lines)
        if log_content:
            print(log_content)
        else:
            print(yellow("No log content available"))

        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
