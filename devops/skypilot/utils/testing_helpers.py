"""
Framework for launching and checking SkyPilot test jobs.
"""

import json
import re
import subprocess
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
        return f"{self.base_name}_{test_name}{suffix}_{timestamp}"

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

    def launch_job(
        self,
        module: str,
        run_name: str,
        base_args: list[str],
        extra_args: list[str],
        test_config: dict[str, Any],
        enable_ci_tests: bool = False,
    ) -> LaunchedJob:
        """Launch a single job and track its status."""
        # Build the command
        cmd = [
            "devops/skypilot/launch.py",
            *base_args,
            module,
            "--args",
            f"run={run_name}",
            *extra_args,
        ]

        if enable_ci_tests:
            cmd.append("--run-ci-tests")

        if self.skip_git_check:
            cmd.append("--skip-git-check")

        # Display launch info
        print(f"\n{bold('Launching job:')} {magenta(run_name)}")
        for key, value in test_config.items():
            print(f"  {cyan(f'{key}:')} {value}")
        print(f"  {cyan('CI Tests:')} {'Yes' if enable_ci_tests else 'No'}")

        try:
            # Launch the job
            result = subprocess.run(cmd, capture_output=True, text=True)
            full_output = result.stdout + "\n" + result.stderr

            # Extract request ID
            request_id = get_request_id_from_launch_output(full_output)

            if request_id:
                print(f"  {green('✅ Launched successfully')} - Request ID: {yellow(request_id)}")

                # Try to get job ID
                job_id = get_job_id_from_request_id(request_id)

                if job_id:
                    print(f"  {green('✅ Job ID retrieved:')} {yellow(job_id)}")
                else:
                    print(f"  {cyan('⚠️  Job ID not available yet (may need more time)')}")

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
            else:
                raise Exception("Failed to get request ID from launch output")

        except Exception as e:
            print(f"  {red('❌ Failed to launch job')}")
            print(f"  {red('Error:')} {str(e)}")

            job = LaunchedJob(
                job_id=None,
                request_id=None,
                run_name=run_name,
                test_config=test_config,
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
        """Print a summary table of launched jobs."""
        print("\n" + bold("Launched Jobs Summary:"))
        print("─" * 80)

        # Print header based on available keys
        if self.launched_jobs:
            # Get all unique keys from test configs
            all_keys = set()
            for job in self.launched_jobs:
                all_keys.update(job.test_config.keys())

            headers = ["Job ID", "Request ID"] + sorted(all_keys)
            print(" │ ".join(f"{h:^15}" for h in headers))
            print("─" * 80)

            # Print rows
            for job in self.launched_jobs:
                job_id = job.job_id or "pending"
                request_id = job.request_id[:8] + "..." if job.request_id else "N/A"

                values = [job_id, request_id]
                for key in sorted(all_keys):
                    value = str(job.test_config.get(key, "-"))
                    values.append(value[:15])

                print(" │ ".join(f"{v:^15}" for v in values))

        print("─" * 80)


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
        print("-" * 60)

        status_counts = {}
        for job in launched_jobs:
            if job.get("job_id"):
                job_id = int(job["job_id"])
                job_info = self.job_statuses.get(job_id, {})
                status = job_info.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

                # Display job info based on available keys
                info_parts = [f"Job {yellow(job['job_id'])}: {self.formatter.format_status(status)}"]

                # Add any relevant test config info
                for key, value in sorted(job.items()):
                    if key not in ["job_id", "request_id", "run_name", "launch_time", "success"]:
                        info_parts.append(f"{key}={value}")

                print(f"  {' | '.join(info_parts)}")

        print("-" * 60)

        # Display status counts
        for status, count in sorted(status_counts.items()):
            print(f"{self.formatter.format_status(status)}: {count}")

        return status_counts

    def print_detailed_table(self) -> None:
        """Print a detailed summary table with custom columns."""
        launched_jobs = self.jobs_data.get("launched_jobs", [])
        if not launched_jobs:
            return

        print(f"\n{bold('Detailed Job Status:')}")
        print("─" * 130)

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

        # Print headers
        header_line = " │ ".join(f"{h:^15}" for h in headers)
        print(header_line)
        print("─" * 130)

        # Print rows
        for job in launched_jobs:
            job_id_str = job.get("job_id")
            if not job_id_str:
                continue

            job_id = int(job_id_str)
            job_info = self.job_statuses.get(job_id, {})
            status = job_info.get("status", "UNKNOWN")
            summary = self.job_summaries.get(job_id, {})

            # Base values
            values = [
                yellow(job_id_str),
                self.formatter.format_status(status),
                self.formatter.format_restart_count(summary.get("restart_count")),
                self.formatter.format_termination_reason(summary.get("termination_reason")),
                self.formatter.format_exit_code(summary.get("exit_code")),
            ]

            # Add extra values
            for header in extra_headers:
                value = str(job.get(header, "-"))
                values.append(value[:15])

            # Format row accounting for ANSI codes
            formatted_values = []
            for value in values:
                visible_len = len(re.sub(r"\x1b\[[0-9;]+m", "", str(value)))
                padding = 15 - visible_len
                formatted_values.append(f"{value}{' ' * max(0, padding)}")

            row = " │ ".join(formatted_values)
            print(row)

        print("─" * 130)

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
