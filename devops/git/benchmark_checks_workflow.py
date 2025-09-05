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


def trigger_workflow(branch: str) -> str:
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

    time.sleep(2)  # allow GitHub to register the workflow run

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
            "1",
            "--json",
            "databaseId",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch run list: {result.stderr}")
    runs = json.loads(result.stdout)
    if not runs:
        raise RuntimeError("No runs found after triggering workflow")
    return str(runs[0]["databaseId"])


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
            if step_name in step.get("name", ""):
                if step["status"] == "completed" and step["conclusion"] == "success":
                    started = datetime.fromisoformat(step["started_at"].replace("Z", "+00:00"))
                    completed = datetime.fromisoformat(step["completed_at"].replace("Z", "+00:00"))
                    return (completed - started).total_seconds()
        return None
    except Exception:
        return None


def wait_for_run_completion(run_id: str) -> tuple[WorkflowRunDetails, str]:
    """Wait for workflow run to complete and collect detailed timing data"""
    retries = 0
    while True:
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
            details.total_duration = duration_seconds(data["startedAt"], data["updatedAt"])

            if conclusion != "success":
                raise WorkflowRunError(f"Workflow run completed with conclusion: {conclusion}")

            # Get detailed job and step timings
            try:
                jobs = get_job_details(run_id)

                # Find the unit-tests job
                unit_tests_job = jobs.get("Unit Tests - All Packages")
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
                        job_duration = duration_seconds(job_data["startedAt"], job_data["completedAt"])
                        details.job_durations[job_name] = job_duration

            except Exception as e:
                print(f"⚠️  Failed to get detailed timing data: {e}")

            return details, conclusion

        time.sleep(POLL_INTERVAL)


def duration_seconds(start: str, end: str) -> float:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end_time - start_time).total_seconds()


def trigger_all_runs(branches: list[str], repeats: int) -> dict[str, list[str]]:
    print("\n🚀 Triggering all workflow runs...")
    run_ids_by_branch = {branch: [] for branch in branches}
    for branch in branches:
        for i in range(repeats):
            print(f"▶️  Trigger {i + 1}/{repeats} for `{branch}`")
            try:
                run_id = trigger_workflow(branch)
                run_ids_by_branch[branch].append(run_id)
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
                print(f"✅ {branch} → {details.total_duration:.1f}s (conclusion: {conclusion})")
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
    print("\n🏃 WORKFLOW TOTAL DURATION (seconds):")
    print(f"{'Branch':<20} {'Min':>8} {'Mean':>8} {'Max':>8} {'StdDev':>8} {'Success':>8} {'Failed':>7}")
    print("-" * 75)

    for branch, results in results_by_branch.items():
        successful_times = results["successful"]
        failed_count = len(results["failed"])

        if successful_times:
            min_time = min(successful_times)
            mean_time = statistics.mean(successful_times)
            max_time = max(successful_times)
            std_dev = statistics.stdev(successful_times) if len(successful_times) > 1 else 0
            success_count = len(successful_times)
            print(
                f"{branch:<20} {min_time:8.1f} {mean_time:8.1f} "
                f"{max_time:8.1f} {std_dev:8.1f} {success_count:>8} {failed_count:>7}"
            )
        else:
            print(f"{branch:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>8} {failed_count:>7}")

    # Unit test step timing summary
    print("\n🧪 UNIT TEST STEP TIMINGS (seconds):")

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
                    f"min={min(setup_times):6.1f}s  "
                    f"mean={statistics.mean(setup_times):6.1f}s  "
                    f"max={max(setup_times):6.1f}s  "
                    f"(n={len(setup_times)})"
                )

            if test_times:
                print(
                    f"  Run all package tests:  "
                    f"min={min(test_times):6.1f}s  "
                    f"mean={statistics.mean(test_times):6.1f}s  "
                    f"max={max(test_times):6.1f}s  "
                    f"(n={len(test_times)})"
                )

    # Job timing summary
    print("\n⚙️  JOB DURATION BREAKDOWN (mean seconds):")
    all_job_names = set()
    for results in results_by_branch.values():
        for details in results["detailed_timings"]:
            all_job_names.update(details.job_durations.keys())

    if all_job_names:
        sorted_job_names = sorted(all_job_names)
        print(f"{'Job':<40} " + " ".join(f"{branch:<12}" for branch in results_by_branch.keys()))
        print("-" * (40 + 13 * len(results_by_branch)))

        for job_name in sorted_job_names:
            row = f"{job_name[:39]:<40}"
            for _branch, results in results_by_branch.items():
                job_times = [
                    d.job_durations.get(job_name) for d in results["detailed_timings"] if job_name in d.job_durations
                ]
                if job_times:
                    mean_time = statistics.mean(job_times)
                    row += f" {mean_time:>11.1f}"
                else:
                    row += f" {'N/A':>11}"
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
