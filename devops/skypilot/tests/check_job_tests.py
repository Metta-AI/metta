#!/usr/bin/env -S uv run
"""
Check logs for skypilot test jobs launched by launch_job_tests.py.

This script reads the JSON output file and displays a summary table of jobs with parsed log information.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

from devops.skypilot.utils.job_helpers import check_job_statuses, tail_job_log
from metta.common.util.text_styles import bold, cyan, green, red, yellow


def parse_job_summary(log_content: str) -> Dict[str, Optional[str]]:
    """Parse job summary information from log content."""
    summary = {
        "exit_code": None,
        "termination_reason": None,
        "metta_run_id": None,
        "skypilot_task_id": None,
    }

    if not log_content:
        return summary

    # Parse [SUMMARY] lines
    for line in log_content.split("\n"):
        if "[SUMMARY] Exit code:" in line:
            match = re.search(r"Exit code: (\d+)", line)
            if match:
                summary["exit_code"] = match.group(1)
        elif "[SUMMARY] Termination reason:" in line:
            match = re.search(r"Termination reason: (.+?)(?:\s*\[|$)", line)
            if match:
                summary["termination_reason"] = match.group(1).strip()
        elif "[SUMMARY] Metta Run ID:" in line:
            match = re.search(r"Metta Run ID: (.+?)(?:\s*\[|$)", line)
            if match:
                summary["metta_run_id"] = match.group(1).strip()
        elif "[SUMMARY] Skypilot Task ID:" in line:
            match = re.search(r"Skypilot Task ID: (.+?)(?:\s*\[|$)", line)
            if match:
                summary["skypilot_task_id"] = match.group(1).strip()

    return summary


def format_status(status: str) -> str:
    """Format status with color coding."""
    if status in ["RUNNING", "PROVISIONING"]:
        return cyan(status)
    elif status == "SUCCEEDED":
        return green(status)
    elif status in ["FAILED", "CANCELLED", "FAILED_SETUP"]:
        return red(status)
    else:
        return yellow(status)


def format_exit_code(code: Optional[str]) -> str:
    """Format exit code with color coding."""
    if code is None:
        return "-"
    elif code == "0":
        return green(code)
    else:
        return red(code)


def format_termination_reason(reason: Optional[str]) -> str:
    """Format termination reason with color coding."""
    if reason is None:
        return "-"
    elif reason == "job_completed":
        return green(reason)
    elif reason in ["heartbeat_timeout", "runtime_timeout"]:
        return yellow(reason)
    else:
        return red(reason)


def print_quick_summary(jobs: list, job_statuses: Dict) -> Dict[str, int]:
    """Print a quick status summary and return status counts."""
    print(f"\n{bold('Job Status Summary:')}")
    print("-" * 60)

    status_counts = {}
    for job in jobs:
        if job.get("job_id"):
            job_id = int(job["job_id"])
            job_info = job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

            # Color code the status
            status_colored = format_status(status)
            print(f"  Job {yellow(job['job_id'])}: {status_colored} ({job['condition_name']})")

    print("-" * 60)

    # Display status counts
    for status, count in sorted(status_counts.items()):
        status_colored = format_status(status)
        print(f"{status_colored}: {count}")

    return status_counts


def parse_all_job_summaries(jobs: list, tail_lines: int) -> Dict[int, Dict[str, Optional[str]]]:
    """Parse job summaries from logs for all jobs."""
    print(f"\n{cyan('Parsing job logs for detailed information...')}")
    job_summaries = {}

    for job in jobs:
        job_id_str = job.get("job_id")
        if job_id_str:
            job_id = int(job_id_str)
            # Get log content to parse summary info
            log_content = tail_job_log(job_id_str, tail_lines)
            job_summaries[job_id] = parse_job_summary(log_content)

    return job_summaries


def print_detailed_table(jobs: list, job_statuses: Dict, job_summaries: Dict) -> None:
    """Print a detailed summary table of all jobs with parsed log information."""
    print(f"\n{bold('Detailed Job Status:')}")
    print("─" * 120)

    # Header
    headers = ["Job ID", "Status", "Exit", "Termination", "Nodes", "Condition", "CI", "Run Name"]
    col_widths = [8, 12, 6, 20, 6, 20, 4, 40]

    # Print headers
    header_line = ""
    for header, width in zip(headers, col_widths, strict=False):
        header_line += f"{header:^{width}} │ "
    print(header_line.rstrip(" │"))
    print("─" * 120)

    # Print job rows
    for job in jobs:
        job_id_str = job.get("job_id")
        if not job_id_str:
            continue

        job_id = int(job_id_str)
        job_info = job_statuses.get(job_id, {})
        status = job_info.get("status", "UNKNOWN")
        summary = job_summaries.get(job_id, {})

        # Format values
        job_id_fmt = yellow(job_id_str)
        status_fmt = format_status(status)
        exit_fmt = format_exit_code(summary.get("exit_code"))
        term_fmt = format_termination_reason(summary.get("termination_reason"))
        nodes_fmt = str(job["nodes"])
        condition_fmt = job["condition_name"][:20]
        ci_fmt = green("✓") if job.get("ci_tests_enabled") else ""
        run_name_fmt = cyan(job["run_name"][:40])

        # Build row
        row_values = [job_id_fmt, status_fmt, exit_fmt, term_fmt, nodes_fmt, condition_fmt, ci_fmt, run_name_fmt]

        row = ""
        for value, width in zip(row_values, col_widths, strict=False):
            # Account for ANSI color codes when calculating padding
            visible_len = len(re.sub(r"\x1b\[[0-9;]+m", "", str(value)))
            padding = width - visible_len
            row += f"{value}{' ' * max(0, padding)} │ "
        print(row.rstrip(" │"))

    print("─" * 120)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check status for skypilot test jobs")
    parser.add_argument(
        "-f", "--input-file", default="skypilot_test_jobs.json", help="Input JSON file with job information"
    )
    parser.add_argument("-l", "--logs", action="store_true", help="Show detailed logs for each job")
    parser.add_argument(
        "-n", "--tail-lines", type=int, default=200, help="Number of lines to tail from each log (default: 200)"
    )

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
    print(bold(f"\n=== Checking {len(launched_jobs)} Jobs from {input_path}==="))

    # Get all job IDs and check their statuses
    job_ids = [int(job["job_id"]) for job in launched_jobs if job.get("job_id")]
    job_statuses = check_job_statuses(job_ids)

    # Quick status summary first - provide immediate feedback
    print_quick_summary(launched_jobs, job_statuses)

    # Parse job summaries from logs (this might take a moment)
    job_summaries = parse_all_job_summaries(launched_jobs, args.tail_lines)

    # Print the detailed summary table
    print_detailed_table(launched_jobs, job_statuses, job_summaries)

    # If logs flag is set, show detailed logs
    if args.logs:
        print(f"\n{bold('Detailed Logs:')}")

        for job in launched_jobs:
            job_id_str = job.get("job_id")

            if not job_id_str:
                print(f"\n{yellow('Skipping job without ID:')} {job['run_name']}")
                continue

            job_id = int(job_id_str)
            job_info = job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            summary = job_summaries.get(job_id, {})

            # Display job header
            print("\n" + "=" * 80)
            print(f"{bold('Job ID:')} {yellow(job_id_str)} ({format_status(status)})")
            print(f"{bold('Run Name:')} {cyan(job['run_name'])}")
            print(f"{bold('Nodes:')} {job['nodes']} | {bold('Condition:')} {job['condition_name']}")
            if job.get("ci_tests_enabled"):
                print(f"{bold('CI Tests:')} {green('Enabled')}")

            # Show parsed summary info
            if any(summary.values()):
                print(f"{bold('Exit Code:')} {format_exit_code(summary.get('exit_code'))}")
                print(f"{bold('Termination:')} {format_termination_reason(summary.get('termination_reason'))}")

            print("=" * 80)

            # Get and display log content
            log_content = tail_job_log(job_id_str, args.tail_lines)
            if log_content:
                print(log_content)
            else:
                print(yellow("No log content available"))

            print("\n" + "-" * 80)

    # Print hints
    print(f"\n{bold('Hints:')}")
    print(f"  • Use {cyan('-l')} or {cyan('--logs')} to view detailed job logs")
    print(f"  • Use {cyan('-n <lines>')} to change the number of log lines to tail")
    print("  • Jobs with exit code 0 and termination reason 'job_completed' are successful")


if __name__ == "__main__":
    main()
