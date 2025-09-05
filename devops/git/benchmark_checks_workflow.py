#!/usr/bin/env python3

import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO

REPO = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
WORKFLOW_FILENAME = "checks.yml"
WORKFLOW_NAME = "Test and Benchmark"
RUN_LINT = "true"
RUN_TEST = "true"
RUN_BENCHMARK = "true"

# Configuration constants
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds
POLL_INTERVAL = 10  # seconds


class WorkflowRunError(Exception):
    """Custom exception for workflow run errors"""

    pass


class WorkflowRunDetails:
    """Container for detailed workflow run information"""

    def __init__(self):
        self.total_duration: Optional[float] = None
        self.conclusion: Optional[str] = None
        self.setup_env_duration: Optional[float] = None
        self.run_tests_duration: Optional[float] = None
        self.job_durations: dict[str, float] = {}


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable string"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s}s"


def parse_time(timestamp: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def trigger_workflow(branch: str) -> str:
    """Trigger a workflow run and return the UUID tag used for identification."""
    run_id = str(uuid.uuid4())[:8]
    print(f"🚀 Triggering workflow on branch: {branch} (run_id={run_id})")

    result = subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            WORKFLOW_FILENAME,
            "--ref",
            branch,
            "-f",
            f"run_lint={RUN_LINT}",
            "-f",
            f"run_test={RUN_TEST}",
            "-f",
            f"run_benchmark={RUN_BENCHMARK}",
            "-f",
            f"run_id={run_id}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to trigger workflow: {result.stderr}")

    return run_id


def find_workflow_run(branch: str, run_id: str) -> str:
    """Poll GitHub Actions to find the workflow run matching the given run_id echoed in logs."""
    print(f"⏳ Searching for workflow run with run_id={run_id} on branch={branch}")

    for attempt in range(10):
        result = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                WORKFLOW_NAME,
                "--branch",
                branch,
                "--limit",
                "10",
                "--json",
                "databaseId,createdAt",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"⚠️ Failed to fetch run list (attempt {attempt + 1}/10): {result.stderr}")
            time.sleep(2)
            continue

        try:
            runs = json.loads(result.stdout)

            for run in runs:
                run_db_id = str(run["databaseId"])

                # Fetch logs and look for the echoed run ID
                log_result = subprocess.run(
                    ["gh", "run", "view", run_db_id, "--log"],
                    capture_output=True,
                    text=True,
                )

                if log_result.returncode != 0:
                    continue

                if f"RUN_ID={run_id}" in log_result.stdout:
                    print(f"✅ Found run {run_db_id} for run_id={run_id} (via logs)")
                    return run_db_id

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")

        time.sleep(5)

    raise RuntimeError(f"❌ Could not find matching run for run_id={run_id} on branch={branch}")


def get_job_details(run_id: str) -> dict[str, Any]:
    """Fetch job details for a workflow run"""
    result = subprocess.run(
        ["gh", "run", "view", run_id, "--json", "jobs"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise WorkflowRunError(f"Failed to get job details: {result.stderr}")

    try:
        data = json.loads(result.stdout)
        return {job["name"]: job for job in data["jobs"]}
    except (json.JSONDecodeError, KeyError) as e:
        raise WorkflowRunError(f"Failed to parse job details: {e}") from e


def get_step_timing(job_id: str, step_name: str) -> Optional[float]:
    """Get timing for a specific step within a job"""
    result = subprocess.run(
        ["gh", "api", f"/repos/{REPO}/actions/jobs/{job_id}"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        job_data = json.loads(result.stdout)

        for step in job_data.get("steps", []):
            # Use exact match instead of substring match
            if step.get("name", "") == step_name:
                if step["status"] == "completed" and step["conclusion"] == "success":
                    started = parse_time(step["started_at"])
                    completed = parse_time(step["completed_at"])
                    return (completed - started).total_seconds()

        return None

    except Exception:
        return None


def wait_for_run_completion(run_id: str) -> tuple[WorkflowRunDetails, str]:
    """Wait for workflow run to complete and collect detailed timing data"""
    retries = 0
    while True:
        # Get comprehensive run data
        result = subprocess.run(
            ["gh", "run", "view", run_id, "--json", "status,conclusion,startedAt,updatedAt"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            retries += 1
            if retries > MAX_RETRIES:
                raise WorkflowRunError(f"Failed to get run status after {MAX_RETRIES} retries: {result.stderr}")
            print(f"⚠️  Failed to get run status (attempt {retries}/{MAX_RETRIES}): {result.stderr}")
            time.sleep(RETRY_DELAY)
            continue

        # Reset retry counter on successful API call
        retries = 0

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise WorkflowRunError(f"Failed to parse run status JSON: {e}") from e

        if data["status"] == "completed":
            conclusion = data.get("conclusion", "unknown")
            details = WorkflowRunDetails()
            details.conclusion = conclusion

            if conclusion != "success":
                raise WorkflowRunError(f"Workflow run completed with conclusion: {conclusion}")

            # Calculate total duration
            if data.get("startedAt") and data.get("updatedAt"):
                started = parse_time(data["startedAt"])
                updated = parse_time(data["updatedAt"])
                details.total_duration = (updated - started).total_seconds()

            try:
                # Get all jobs
                jobs = get_job_details(run_id)

                # Find the unit-tests job
                unit_tests_job = None
                for job_name, job_data in jobs.items():
                    if job_name == "Unit Tests - All Packages":
                        unit_tests_job = job_data
                        break

                if unit_tests_job and unit_tests_job["conclusion"] == "success":
                    job_id = unit_tests_job["databaseId"]

                    # Get timing for specific steps
                    setup_env_time = get_step_timing(job_id, "Setup Environment")
                    run_tests_time = get_step_timing(job_id, "Run all package tests")

                    if setup_env_time is not None:
                        details.setup_env_duration = setup_env_time
                    if run_tests_time is not None:
                        details.run_tests_duration = run_tests_time

                # Collect all job durations
                for job_name, job_data in jobs.items():
                    if (
                        job_data["conclusion"] == "success"
                        and job_data.get("startedAt")
                        and job_data.get("completedAt")
                    ):
                        started = parse_time(job_data["startedAt"])
                        completed = parse_time(job_data["completedAt"])
                        details.job_durations[job_name] = (completed - started).total_seconds()

            except Exception as e:
                print(f"⚠️  Failed to get detailed timing data: {e}")

            return details, conclusion

        time.sleep(POLL_INTERVAL)


def trigger_all_runs(branches: list[str], repeats: int) -> dict[str, list[str]]:
    print("\n🚀 Triggering all workflow runs...")
    run_ids_by_branch = {branch: [] for branch in branches}
    for branch in branches:
        for i in range(repeats):
            print(f"▶️  Trigger {i + 1}/{repeats} for `{branch}`")
            try:
                uuid_tag = trigger_workflow(branch)
                time.sleep(5)  # Optional: give GitHub a head start
                run_number = find_workflow_run(branch, uuid_tag)
                print(f"🎯 Run number for triggered workflow: {run_number}")
                run_ids_by_branch[branch].append(uuid_tag)
            except Exception as e:
                print(f"❌ Failed to trigger workflow on `{branch}`: {e}")
    return run_ids_by_branch


def wait_for_all_runs(run_ids_by_branch: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    print("\n⏳ Waiting for all workflow runs to complete...")
    results_by_branch = {
        branch: {"successful": [], "failed": [], "detailed_timings": []} for branch in run_ids_by_branch
    }

    for branch, run_ids in run_ids_by_branch.items():
        for run_id in run_ids:
            print(f"🔍 Waiting on {branch} → run {run_id}")
            try:
                details, conclusion = wait_for_run_completion(run_id)
                results_by_branch[branch]["successful"].append(details.total_duration)
                results_by_branch[branch]["detailed_timings"].append(details)

                # Print summary for this run
                print(f"✅ {branch} → {format_duration(details.total_duration)} ({details.total_duration:.1f}s)")
                if details.setup_env_duration:
                    print(f"   └─ Setup Environment: {details.setup_env_duration:.1f}s")
                if details.run_tests_duration:
                    print(f"   └─ Run all package tests: {details.run_tests_duration:.1f}s")

            except WorkflowRunError as e:
                print(f"❌ {branch} run {run_id} failed: {e}")
                results_by_branch[branch]["failed"].append(run_id)
            except Exception as e:
                print(f"❌ Unexpected error for {branch} run {run_id}: {e}")
                results_by_branch[branch]["failed"].append(run_id)

    return results_by_branch


def summarize(results_by_branch: dict[str, dict[str, Any]]):
    print("\n📊 Benchmark Summary:")
    print("=" * 100)

    # Overall workflow timing summary
    print("\n🏃 WORKFLOW TOTAL DURATION:")
    print(f"{'Branch':<20} {'Min':>10} {'Mean':>10} {'Max':>10} {'StdDev':>10} {'Success':>8} {'Failed':>7}")
    print("-" * 85)

    for branch, results in results_by_branch.items():
        successful_times = results["successful"]
        failed_count = len(results["failed"])

        if successful_times:
            min_time = min(successful_times)
            mean_time = statistics.mean(successful_times)
            max_time = max(successful_times)
            std_dev = statistics.stdev(successful_times) if len(successful_times) > 1 else 0
            success_count = len(successful_times)

            # Format with both seconds and human-readable
            print(
                f"{branch:<20} "
                f"{format_duration(min_time):>10} "
                f"{format_duration(mean_time):>10} "
                f"{format_duration(max_time):>10} "
                f"{std_dev:>9.1f}s "
                f"{success_count:>8} "
                f"{failed_count:>7}"
            )
        else:
            print(f"{branch:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>8} {failed_count:>7}")

    # Unit test step timing summary
    print("\n🧪 UNIT TEST STEP TIMINGS:")

    for branch, results in results_by_branch.items():
        detailed_timings = results["detailed_timings"]
        if not detailed_timings:
            continue

        # Collect setup environment times
        setup_times = [d.setup_env_duration for d in detailed_timings if d.setup_env_duration is not None]
        test_times = [d.run_tests_duration for d in detailed_timings if d.run_tests_duration is not None]

        if setup_times or test_times:
            print(f"\n{branch}:")

            if setup_times:
                print(
                    f"  Setup Environment:      "
                    f"min={format_duration(min(setup_times)):>6}  "
                    f"mean={format_duration(statistics.mean(setup_times)):>6}  "
                    f"max={format_duration(max(setup_times)):>6}  "
                    f"(n={len(setup_times)})"
                )

            if test_times:
                print(
                    f"  Run all package tests:  "
                    f"min={format_duration(min(test_times)):>6}  "
                    f"mean={format_duration(statistics.mean(test_times)):>6}  "
                    f"max={format_duration(max(test_times)):>6}  "
                    f"(n={len(test_times)})"
                )

    # Job timing summary
    print("\n⚙️  JOB DURATION BREAKDOWN (mean):")
    all_job_names = set()
    for results in results_by_branch.values():
        for details in results["detailed_timings"]:
            all_job_names.update(details.job_durations.keys())

    if all_job_names:
        sorted_job_names = sorted(all_job_names)
        print(f"{'Job':<40} " + " ".join(f"{branch:<15}" for branch in results_by_branch.keys()))
        print("-" * (40 + 16 * len(results_by_branch)))

        for job_name in sorted_job_names:
            row = f"{job_name[:39]:<40}"
            for _branch, results in results_by_branch.items():
                job_times = []
                for d in results["detailed_timings"]:
                    if job_name in d.job_durations:
                        job_times.append(d.job_durations[job_name])

                if job_times:
                    mean_time = statistics.mean(job_times)
                    row += f" {format_duration(mean_time):>14}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    # Print failed run details if any
    total_failed = sum(len(results["failed"]) for results in results_by_branch.values())
    if total_failed > 0:
        print(f"\n⚠️  Total failed runs: {total_failed}")
        for branch, results in results_by_branch.items():
            if results["failed"]:
                print(f"  {branch}: {', '.join(results['failed'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GitHub workflow across branches.")
    parser.add_argument("branches", nargs="+", help="Branches to benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=10, help="Number of runs per branch")
    args = parser.parse_args()

    triggered = trigger_all_runs(args.branches, args.repeats)
    results = wait_for_all_runs(triggered)
    summarize(results)
