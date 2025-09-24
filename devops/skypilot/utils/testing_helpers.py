"""
Framework for launching and checking SkyPilot test jobs.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from devops.skypilot.utils.job_helpers import (
    check_git_state,
    check_job_statuses,
    get_job_id_from_request_id,
    get_request_id_from_launch_output,
    tail_job_log,
)
from metta.common.util.retry import retry_function
from metta.common.util.text_styles import bold, cyan, green, magenta, red, yellow


@dataclass
class TestCondition:
    """Configuration for a test condition."""

    name: str
    extra_args: list[str]
    description: str
    ci: bool = False


@dataclass
class LaunchedJob:
    """Information about a launched job."""

    job_id: Optional[str]
    request_id: Optional[str]
    run_name: str
    test_config: dict[str, Any]
    launch_time: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "run_name": self.run_name,
            **self.test_config,
            "launch_time": self.launch_time,
            "success": self.success,
        }


class SkyPilotTestLauncher:
    """Manages launching SkyPilot test jobs."""

    def __init__(self, base_name: str = "skypilot_test", skip_git_check: bool = False):
        self.base_name = base_name
        self.skip_git_check = skip_git_check
        self.launched_jobs: list[LaunchedJob] = []
        self.failed_launches: list[LaunchedJob] = []

    def generate_run_name(self, test_name: str, extra_suffix: str = "") -> str:
        """Generate a descriptive run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{extra_suffix}" if extra_suffix else ""

        # Replace slashes with underscores for Sky compatibility
        safe_test_name = test_name.replace("/", "_")

        return f"{self.base_name}_{safe_test_name}{suffix}_{timestamp}"

    def check_git_state(self) -> bool:
        """Check if git state is clean. Returns True if clean or check is skipped."""
        if self.skip_git_check:
            return True

        print(f"\n{bold('Checking git state...')}")
        import gitta as git

        commit_hash = git.get_current_commit()
        error_message = check_git_state(commit_hash)

        if error_message:
            print(error_message)
            print("  - Skip check: add --skip-git-check flag")
            return False

        print(f"{green('✅ Git state is clean')}")
        return True

    def _execute_launch_command(self, cmd: list[str]) -> tuple[Optional[str], Optional[str], dict[str, Any]]:
        """Execute the launch command and extract IDs. Returns (job_id, request_id, debug_info)."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        full_output = result.stdout + "\n" + result.stderr

        # Extract request ID
        request_id = get_request_id_from_launch_output(full_output)

        if not request_id:
            # Check for known error patterns
            if "sky-jobs-controller" in full_output.lower() and "not up" in full_output.lower():
                raise Exception("Jobs controller appears to be down")

            # Include debug info in the exception
            debug_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
            raise Exception(f"Failed to get request ID from launch output. Debug: {debug_info}")

        print(f"  {green('✅ Launched successfully')} - Request ID: {yellow(request_id)}")

        # Try to get job ID with retries using retry_function
        def get_job_id_with_wait() -> str:
            job_id = get_job_id_from_request_id(request_id, wait_seconds=2.0)
            if not job_id:
                raise Exception("Job ID not available yet")
            return job_id

        try:
            job_id = retry_function(
                get_job_id_with_wait,
                max_retries=2,
                initial_delay=2.0,
            )
            print(f"  {green('✅ Job ID retrieved:')} {yellow(job_id)}")
        except Exception:
            # Job ID not available yet, but launch was successful
            job_id = None
            print(f"  {cyan('ℹ️  Job ID not available yet (may need more time)')}")

        return job_id, request_id, {}

    def launch_job(
        self,
        module: str,
        run_name: str,
        base_args: list[str],
        extra_args: list[str],
        test_config: dict[str, Any],
        enable_ci_tests: bool = False,
        max_attempts: int = 3,
    ) -> LaunchedJob:
        """Launch a single job and track its status with retry logic."""
        # Build the command
        cmd = [
            "devops/skypilot/launch.py",
            *base_args,
            module,
            f"run={run_name}",
            *extra_args,
        ]

        if enable_ci_tests:
            cmd.append("--run-ci-tests")

        if self.skip_git_check:
            cmd.append("--skip-git-check")

        # Display launch info
        print(f"\n{bold('Launching job:')} {magenta(run_name)}")
        print("  Test Configuration: {")
        for key, value in test_config.items():
            print(f"    {cyan(f'{key}:')} {value}")
        print("  }")

        try:
            job_id, request_id, debug_info = retry_function(
                lambda: self._execute_launch_command(cmd),
                max_retries=max_attempts - 1,
            )

            job = LaunchedJob(
                job_id=job_id,
                request_id=request_id,
                run_name=run_name,
                test_config=test_config,
                launch_time=datetime.now().isoformat(),
                success=True,
            )
            self.launched_jobs.append(job)
            return job

        except Exception as e:
            # All retries exhausted
            print(f"  {red(f'❌ Failed to launch job after {max_attempts} attempts')}")
            print(f"  {red('Final error:')} {str(e)}")

            job = LaunchedJob(
                job_id=None,
                request_id=None,
                run_name=run_name,
                test_config={**test_config, "error": str(e)},
                launch_time=datetime.now().isoformat(),
                success=False,
            )
            self.failed_launches.append(job)
            return job

    def save_results(self, output_file: str = "skypilot_test_jobs.json") -> Path:
        """Save launch results to JSON file."""
        output_data = {
            "test_run_info": {
                "base_name": self.base_name,
                "launch_time": datetime.now().isoformat(),
                "total_jobs": len(self.launched_jobs) + len(self.failed_launches),
                "successful_launches": len(self.launched_jobs),
                "failed_launches": len(self.failed_launches),
            },
            "launched_jobs": [job.to_dict() for job in self.launched_jobs],
            "failed_launches": [job.to_dict() for job in self.failed_launches],
        }

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_path

    def print_summary(self) -> None:
        """Print launch summary."""
        print(f"\n{bold('=== Launch Summary ===')}")
        if len(self.launched_jobs) > 0:
            print(f"{green('Successfully launched:')} {len(self.launched_jobs)} jobs")
        if len(self.failed_launches) > 0:
            print(f"{red('Failed to launch:')} {len(self.failed_launches)} jobs")

        if self.launched_jobs:
            self._print_summary_table()

        if self.failed_launches:
            print(f"\n{red('Failed launches:')}")
            for job in self.failed_launches:
                print(f"  • {job.run_name}")

    def _print_summary_table(self) -> None:
        """Print a summary table of launched jobs with dynamic column widths."""
        print("\n" + bold("Launched Jobs Summary:"))

        if self.launched_jobs:
            # Get all unique keys from test configs
            all_keys = set()
            for job in self.launched_jobs:
                all_keys.update(job.test_config.keys())

            headers = ["Job ID", "Request ID"] + sorted(all_keys)

            # Calculate column widths based on content
            col_widths = {}

            # Start with header widths
            for header in headers:
                col_widths[header] = len(header) + 2  # +2 for padding

            # Adjust based on actual data
            for job in self.launched_jobs:
                job_id = job.job_id or "pending"
                request_id = job.request_id[:8] + "..." if job.request_id else "N/A"

                col_widths["Job ID"] = max(col_widths["Job ID"], len(job_id) + 2)
                col_widths["Request ID"] = max(col_widths["Request ID"], len(request_id) + 2)

                for key in all_keys:
                    value = str(job.test_config.get(key, "-"))
                    # Cap at reasonable max width
                    display_len = min(len(value), 30) + 2
                    col_widths[key] = max(col_widths.get(key, 0), display_len)

            # Calculate total table width
            separator_width = 3  # " │ "
            table_width = sum(col_widths.values()) + (len(headers) - 1) * separator_width

            # Print table
            print("─" * table_width)

            # Print headers
            header_parts = []
            for header in headers:
                header_parts.append(f"{header:^{col_widths[header]}}")
            print(" │ ".join(header_parts))

            print("─" * table_width)

            # Print rows
            for job in self.launched_jobs:
                job_id = job.job_id or "pending"
                request_id = job.request_id[:8] + "..." if job.request_id else "N/A"

                row_parts = []
                row_parts.append(f"{job_id:^{col_widths['Job ID']}}")
                row_parts.append(f"{request_id:^{col_widths['Request ID']}}")

                for key in sorted(all_keys):
                    value = str(job.test_config.get(key, "-"))
                    # Truncate if needed
                    if len(value) > 30:
                        value = value[:27] + "..."
                    row_parts.append(f"{value:^{col_widths[key]}}")

                print(" │ ".join(row_parts))

            print("─" * table_width)


class JobSummaryParser:
    """Parse job summary information from logs."""

    @staticmethod
    def parse_job_summary(log_content: str) -> dict[str, Optional[str]]:
        """Parse job summary information from log content."""
        summary: dict[str, Optional[str]] = {
            "exit_code": None,
            "termination_reason": None,
            "metta_run_id": None,
            "skypilot_task_id": None,
            "restart_count": None,
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
            elif "[SUMMARY] Restart Count:" in line:
                match = re.search(r"Restart Count: (\d+)", line)
                if match:
                    summary["restart_count"] = match.group(1)

        return summary


class StatusFormatter:
    """Format job status information with colors."""

    @staticmethod
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

    @staticmethod
    def format_exit_code(code: Optional[str]) -> str:
        """Format exit code with color coding."""
        if code is None:
            return "-"
        elif code == "0":
            return green(code)
        else:
            return red(code)

    @staticmethod
    def format_restart_count(count: Optional[str]) -> str:
        """Format restart count with color coding."""
        if count is None:
            return "-"
        elif count == "0":
            return green(count)
        else:
            return yellow(count)

    @staticmethod
    def format_termination_reason(reason: Optional[str]) -> str:
        """Format termination reason with color coding."""
        if reason is None:
            return "-"
        elif reason == "job_completed":
            return green(reason)
        elif reason in ["heartbeat_timeout", "max_runtime_reached"]:
            return yellow(reason)
        else:
            return red(reason)


class SkyPilotJobChecker:
    """Check status of SkyPilot jobs."""

    def __init__(self, input_file: str = "skypilot_test_jobs.json"):
        self.input_file = Path(input_file)
        self.jobs_data: dict = {}
        self.job_statuses: dict[int, dict] = {}
        self.job_summaries: dict[int, dict[str, Optional[str]]] = {}
        self.parser = JobSummaryParser()
        self.formatter = StatusFormatter()

    def load_jobs(self) -> bool:
        """Load job data from JSON file."""
        if not self.input_file.exists():
            print(red(f"Error: Input file '{self.input_file}' not found"))
            return False

        with open(self.input_file, "r") as f:
            self.jobs_data = json.load(f)

        return True

    def check_statuses(self) -> None:
        """Check status of all jobs."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])
        if not launched_jobs:
            print(yellow("No launched jobs found in the input file"))
            return

        # Get all job IDs and check their statuses
        job_ids = [int(job["job_id"]) for job in launched_jobs if job.get("job_id")]
        self.job_statuses = check_job_statuses(job_ids)

    def parse_all_summaries(self, tail_lines: int = 200) -> None:
        """Parse job summaries from logs for all jobs."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])

        print(f"\n{cyan('Parsing job logs for detailed information...')}")
        total_jobs = len([job for job in launched_jobs if job.get("job_id")])
        processed = 0

        for job in launched_jobs:
            job_id_str = job.get("job_id")
            if job_id_str:
                processed += 1
                print(
                    f"  [{processed}/{total_jobs}] Processing job {yellow(job_id_str)}...",
                    end="\r",
                )

                job_id = int(job_id_str)
                log_content = tail_job_log(job_id_str, tail_lines)

                # Clear the progress line
                print(" " * 100, end="\r")

                if log_content:
                    self.job_summaries[job_id] = self.parser.parse_job_summary(log_content)

                    # Print last few lines
                    log_lines = log_content.strip().split("\n")
                    last_lines = log_lines[-5:] if len(log_lines) > 5 else log_lines

                    print(f"{green(f'[{processed}/{total_jobs}]')} Job {yellow(job_id_str)}:")
                    print(f"  {bold('Last log lines:')}")
                    for line in last_lines:
                        truncated = line[:100] + "..." if len(line) > 100 else line
                        print(f"    {truncated}")
                    print()
                else:
                    print(
                        f"{yellow(f'[{processed}/{total_jobs}]')} Job {yellow(job_id_str)}: "
                        f"{yellow('No job logs found')}"
                    )
                    self.job_summaries[job_id] = {}

        print(f"{green('✔')} Parsed logs for {processed} jobs\n")

    def print_quick_summary(self) -> dict[str, int]:
        """Print a quick status summary and return status counts."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])

        print(f"\n{bold('Job Status Summary:')}")

        if not launched_jobs:
            print(yellow("No jobs found"))
            return {}

        # Collect status counts
        status_counts = {}
        job_data = []

        for job in launched_jobs:
            if job.get("job_id"):
                job_id = int(job["job_id"])
                job_info = self.job_statuses.get(job_id, {})
                status = job_info.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

                # Collect relevant data
                job_entry = {"job_id": job["job_id"], "status": status}

                # Add test config info
                for key, value in job.items():
                    if key not in [
                        "job_id",
                        "request_id",
                        "run_name",
                        "launch_time",
                        "success",
                        "description",
                        "recipe_module",
                    ]:
                        job_entry[key] = value

                job_data.append(job_entry)

        # Determine columns to display
        all_keys = set()
        for job in job_data:
            all_keys.update(job.keys())

        # Order columns: job_id, status, then alphabetically for the rest
        ordered_keys = ["job_id", "status"]
        other_keys = sorted(all_keys - set(ordered_keys))
        headers = ordered_keys + other_keys

        # Calculate column widths
        col_widths = {}
        for header in headers:
            # Start with header width
            col_widths[header] = len(header) + 2

            # Adjust based on data
            for job in job_data:
                value = str(job.get(header, "-"))
                # Different max widths for different types of columns
                if "description" in header.lower():
                    max_width = 50
                elif header in ["job_id", "status"]:
                    max_width = 15
                else:
                    max_width = 25

                display_len = min(len(value), max_width) + 2
                col_widths[header] = max(col_widths[header], display_len)

        # Calculate table width
        separator = " │ "
        table_width = sum(col_widths.values()) + len(separator) * (len(headers) - 1)

        # Print table
        print("─" * table_width)

        # Print headers
        header_parts = []
        for header in headers:
            # Make header more readable
            display_header = header.replace("_", " ").title()
            header_parts.append(f"{display_header:^{col_widths[header]}}")
        print(separator.join(header_parts))

        print("─" * table_width)

        # Print rows
        for job in job_data:
            row_parts = []

            for header in headers:
                value = str(job.get(header, "-"))

                # Format special columns
                if header == "job_id":
                    value_display = yellow(value)
                elif header == "status":
                    value_display = self.formatter.format_status(value)
                else:
                    value_display = value
                    # Truncate if needed
                    max_width = col_widths[header] - 2
                    if len(value) > max_width:
                        value_display = value[: max_width - 3] + "..."

                row_parts.append(self._format_cell(value_display, col_widths[header]))

            print(separator.join(row_parts))

        print("─" * table_width)

        # Print status summary
        print(f"\n{bold('Status Summary:')}")
        for status, count in sorted(status_counts.items()):
            print(f"  {self.formatter.format_status(status)}: {count}")

        return status_counts

    def print_detailed_table(self) -> None:
        """Print a detailed summary table with dynamic column widths."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])
        if not launched_jobs:
            return

        print(f"\n{bold('Detailed Job Status:')}")

        # Determine columns based on job data
        base_headers = ["Job ID", "Status", "Restarts", "Termination", "Exit Code"]

        # Get additional headers from job test configs
        extra_headers = set()
        for job in launched_jobs:
            extra_headers.update(job.keys())

        # Remove standard fields we handle separately
        exclude = {"job_id", "request_id", "run_name", "launch_time", "success"}
        extra_headers = sorted(extra_headers - exclude)

        headers = base_headers + extra_headers

        # Calculate column widths based on content
        col_widths = {}

        # Start with header widths (minimum 8 characters)
        for header in headers:
            col_widths[header] = max(len(header) + 2, 10)  # +2 for padding, min 10

        # Adjust widths based on actual data
        for job in launched_jobs:
            job_id_str = job.get("job_id")
            if not job_id_str:
                continue

            job_id = int(job_id_str)
            job_info = self.job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            summary = self.job_summaries.get(job_id, {})

            # Calculate width for base columns
            col_widths["Job ID"] = max(col_widths["Job ID"], len(job_id_str) + 2)
            col_widths["Status"] = max(col_widths["Status"], len(status) + 2)

            restart_val = summary.get("restart_count", "-")
            col_widths["Restarts"] = max(col_widths["Restarts"], len(str(restart_val)) + 2)

            term_val = summary.get("termination_reason", "-")
            col_widths["Termination"] = max(col_widths["Termination"], len(str(term_val)) + 2)

            exit_val = summary.get("exit_code", "-")
            col_widths["Exit Code"] = max(col_widths["Exit Code"], len(str(exit_val)) + 2)

            # Calculate width for extra columns
            for header in extra_headers:
                value = str(job.get(header, "-"))
                # Cap at reasonable max width (40 for description fields, 25 for others)
                max_width = 40 if "description" in header.lower() else 25
                display_len = min(len(value), max_width) + 2
                col_widths[header] = max(col_widths.get(header, 10), display_len)

        # Calculate total table width
        separator_width = 3  # " │ "
        table_width = sum(col_widths.values()) + (len(headers) - 1) * separator_width

        # Print top border
        print("─" * table_width)

        # Print headers
        header_parts = []
        for header in headers:
            header_parts.append(f"{header:^{col_widths[header]}}")
        print(" │ ".join(header_parts))

        # Print separator
        print("─" * table_width)

        # Print rows
        for job in launched_jobs:
            job_id_str = job.get("job_id")
            if not job_id_str:
                continue

            job_id = int(job_id_str)
            job_info = self.job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            summary = self.job_summaries.get(job_id, {})

            # Build row values
            row_parts = []

            # Job ID
            job_id_display = yellow(job_id_str)
            row_parts.append(self._format_cell(job_id_display, col_widths["Job ID"]))

            # Status
            status_display = self.formatter.format_status(status)
            row_parts.append(self._format_cell(status_display, col_widths["Status"]))

            # Restarts
            restarts_display = self.formatter.format_restart_count(summary.get("restart_count"))
            row_parts.append(self._format_cell(restarts_display, col_widths["Restarts"]))

            # Termination
            term_display = self.formatter.format_termination_reason(summary.get("termination_reason"))
            row_parts.append(self._format_cell(term_display, col_widths["Termination"]))

            # Exit Code
            exit_display = self.formatter.format_exit_code(summary.get("exit_code"))
            row_parts.append(self._format_cell(exit_display, col_widths["Exit Code"]))

            # Extra columns
            for header in extra_headers:
                value = str(job.get(header, "-"))
                # Truncate if needed based on column width
                max_width = col_widths[header] - 2
                if "description" in header.lower() and len(value) > max_width:
                    value = value[: max_width - 3] + "..."
                elif len(value) > max_width:
                    value = value[: max_width - 3] + "..."
                row_parts.append(self._format_cell(value, col_widths[header]))

            print(" │ ".join(row_parts))

        # Print bottom border
        print("─" * table_width)

    def _format_cell(self, value: str, width: int) -> str:
        """Format a cell value with proper width accounting for ANSI codes."""
        # Calculate visible length (without ANSI escape codes)
        visible_len = len(re.sub(r"\x1b\[[0-9;]+m", "", str(value)))
        padding = width - visible_len
        return f"{value}{' ' * max(0, padding)}"

    def show_detailed_logs(self, tail_lines: int = 200) -> None:
        """Show detailed logs for each job."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])

        print(f"\n{bold('Detailed Logs:')}")

        for job in launched_jobs:
            job_id_str = job.get("job_id")
            if not job_id_str:
                continue

            job_id = int(job_id_str)
            job_info = self.job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            summary = self.job_summaries.get(job_id, {})

            # Display job header
            print("\n" + "=" * 80)
            print(f"{bold('Job ID:')} {yellow(job_id_str)} ({self.formatter.format_status(status)})")
            print(f"{bold('Run Name:')} {cyan(job['run_name'])}")

            # Display test configuration
            for key, value in job.items():
                if key not in ["job_id", "request_id", "run_name", "launch_time", "success"]:
                    print(f"{bold(f'{key.title()}:')} {value}")

            # Show parsed summary info
            if any(summary.values()):
                print(f"{bold('Exit Code:')} {self.formatter.format_exit_code(summary.get('exit_code'))}")
                print(f"{bold('Restart Count:')} {self.formatter.format_restart_count(summary.get('restart_count'))}")
                print(
                    f"{bold('Termination:')} "
                    f"{self.formatter.format_termination_reason(summary.get('termination_reason'))}"
                )

            print("=" * 80)

            # Get and display log content
            log_content = tail_job_log(job_id_str, tail_lines)
            if log_content:
                print(log_content)
            else:
                print(yellow("No log content available"))

            print("\n" + "-" * 80)


class BaseTestRunner:
    """Base class for SkyPilot test runner scripts."""

    def __init__(
        self,
        prog_name: str,
        description: str,
        default_output_file: str,
        default_base_name: str,
        test_type: str = "Test",
    ):
        self.prog_name = prog_name
        self.description = description
        self.default_output_file = default_output_file
        self.default_base_name = default_base_name
        self.test_type = test_type

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with launch/check/kill subcommands."""
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            prog=self.prog_name,
        )

        # Create subparsers
        subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

        # Launch subcommand
        launch_parser = subparsers.add_parser(
            "launch",
            help="Launch test jobs",
            description=self.get_launch_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.get_launch_epilog(),
        )
        launch_parser.add_argument("--base-name", default=self.default_base_name, help="Base name for test runs")
        launch_parser.add_argument("--output-file", default=self.default_output_file, help="Output JSON file")
        launch_parser.add_argument("--skip-git-check", action="store_true", help="Skip git state validation")

        # Add any custom launch arguments
        self.add_custom_launch_args(launch_parser)

        # Check subcommand
        check_parser = subparsers.add_parser(
            "check",
            help="Check test results",
            description=f"Check the status and results of {self.test_type.lower()} jobs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                      # Check job statuses
  %(prog)s -l                   # Check with detailed logs
  %(prog)s -n 500               # Check with 500 lines of logs
  %(prog)s -f custom.json       # Check from custom file
        """,
        )
        check_parser.add_argument("-f", "--input-file", default=self.default_output_file, help="Input JSON file")
        check_parser.add_argument("-l", "--logs", action="store_true", help="Show detailed logs")
        check_parser.add_argument("-n", "--tail-lines", type=int, default=200, help="Log lines to tail")

        # Kill subcommand
        kill_parser = subparsers.add_parser(
            "kill",
            help="Kill all test jobs",
            description=f"Kill all {self.test_type.lower()} jobs from a test run",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                      # Kill all jobs from default file
  %(prog)s -f custom.json       # Kill jobs from custom file
        """,
        )
        kill_parser.add_argument("-f", "--input-file", default=self.default_output_file, help="Input JSON file")

        return parser

    def check_tests(self, args) -> None:
        """Generic check implementation - same for all test types."""
        # Create checker
        checker = SkyPilotJobChecker(input_file=args.input_file)

        # Load jobs
        if not checker.load_jobs():
            print(f"Run '{self.prog_name} launch' first to create the job file")
            sys.exit(1)

        # Get job count
        launched_jobs = checker.jobs_data.get("launched_jobs", [])
        if not launched_jobs:
            sys.exit(0)

        # Summary header
        test_info = checker.jobs_data.get("test_run_info", {})
        print(bold(f"\n=== Checking {len(launched_jobs)} {self.test_type} Jobs ==="))
        print(f"{cyan('Test run:')} {test_info.get('base_name', 'Unknown')}")
        print(f"{cyan('Launch time:')} {test_info.get('launch_time', 'Unknown')}")
        print(f"{cyan('Input file:')} {args.input_file}")

        # Check job statuses
        checker.check_statuses()

        # Quick status summary first
        checker.print_quick_summary()

        # Parse job summaries from logs
        checker.parse_all_summaries(args.tail_lines)

        # Print detailed table
        checker.print_detailed_table()

        # Show detailed logs if requested
        if args.logs:
            checker.show_detailed_logs(args.tail_lines)

        # Print hints
        print(f"\n{bold('Hints:')}")
        print(f"  • Use {cyan('check -l')} to view detailed job logs")
        print(f"  • Use {cyan('check -n <lines>')} to change log lines to tail")
        print(f"  • Use {cyan('sky jobs logs <job_id>')} to view a single job's full log")

    def kill_tests(self, args) -> None:
        """Kill all jobs from a test run."""
        # Load jobs data
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(red(f"Error: Input file '{input_path}' not found"))
            print(f"Run '{self.prog_name} launch' first to create the job file")
            sys.exit(1)

        import json

        with open(input_path, "r") as f:
            jobs_data = json.load(f)

        # Get launched jobs with valid job IDs
        launched_jobs = jobs_data.get("launched_jobs", [])
        job_ids = [job["job_id"] for job in launched_jobs if job.get("job_id")]

        if not job_ids:
            print(yellow("No jobs found to kill"))
            return

        # Show what will be killed
        test_info = jobs_data.get("test_run_info", {})
        print(bold(f"\n=== Kill {len(job_ids)} {self.test_type} Jobs ==="))
        print(f"{cyan('Test run:')} {test_info.get('base_name', 'Unknown')}")
        print(f"{cyan('Launch time:')} {test_info.get('launch_time', 'Unknown')}")
        print(f"{cyan('Jobs to kill:')} {', '.join(job_ids)}")

        # Kill each job
        print(f"\n{cyan('Killing jobs...')}")
        killed_count = 0
        completed_count = 0
        failed_kills = []

        for i, job_id in enumerate(job_ids):
            print(f"  [{i + 1}/{len(job_ids)}] Killing job {yellow(job_id)}...", end="", flush=True)

            success, status = self._kill_job_with_retry(job_id)

            if success:
                if status == "Killed":
                    killed_count += 1
                    print(f" {red('✗')} {status}")
                else:  # "Already completed" or "Not found"
                    completed_count += 1
                    print(f" {green('✓')} {status}")
            else:
                failed_kills.append((job_id, status))
                print(f" {red('✗ Failed')}: {status}")

        # Summary
        print(f"\n{bold('=== Summary ===')}")
        if killed_count > 0:
            print(f"{yellow('Killed:')} {killed_count} jobs")
        if completed_count > 0:
            print(f"{green('Already completed:')} {completed_count} jobs")
        if failed_kills:
            print(f"{red('Failed:')} {len(failed_kills)} jobs")

        # Exit with error if any kills failed
        if failed_kills:
            sys.exit(1)

    def _kill_job_with_retry(self, job_id: str, max_attempts: int = 3) -> tuple[bool, str]:
        """Kill a single job with retry logic. Returns (success, status_message)."""

        def execute_kill():
            cmd = ["sky", "jobs", "cancel", "-y", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check stderr first, regardless of return code
            stdout_lower = result.stdout.lower()

            # Check if already terminated
            if "already in terminal state" in stdout_lower:
                return True, "Already completed"

            # Check if not found
            if "not found" in stdout_lower:
                return True, "Not found"

            # If return code is 0, it was actually killed
            if result.returncode == 0:
                return True, "Killed"

            # Otherwise it's an error
            raise Exception(result.stdout.strip() or "Failed")

        try:
            success, status = retry_function(
                execute_kill,
                max_retries=max_attempts - 1,
                initial_delay=1.0,
            )
            return success, status
        except Exception as e:
            return False, str(e)

    def run(self) -> None:
        """Main entry point for the test runner."""
        parser = self.create_parser()
        args = parser.parse_args()

        if args.command == "launch":
            self.launch_tests(args)
        elif args.command == "check":
            self.check_tests(args)
        elif args.command == "kill":
            self.kill_tests(args)

    # Methods to be implemented by subclasses
    def get_launch_description(self) -> str:
        """Get the launch subcommand description."""
        return f"Launch {self.test_type.lower()} jobs on the cluster"

    def get_launch_epilog(self) -> str:
        """Get the launch subcommand epilog with examples."""
        return """
Examples:
  %(prog)s                      # Launch with default settings
  %(prog)s --skip-git-check     # Launch without git validation
  %(prog)s --base-name custom   # Launch with custom base name
        """

    def add_custom_launch_args(self, parser: argparse.ArgumentParser) -> None:
        """Add any custom arguments to the launch subcommand."""
        pass

    def launch_tests(self, args) -> None:
        """Launch test jobs - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement launch_tests()")
