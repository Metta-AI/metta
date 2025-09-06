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

# Matrix job patterns to aggregate
MATRIX_JOB_PATTERNS = {
    "unit-tests": "Unit Tests - ",  # Matches "Unit Tests - agent", "Unit Tests - common", etc.
}


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
        self.matrix_aggregates: dict[str, dict[str, Any]] = {}


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


def aggregate_matrix_jobs(jobs: dict[str, Any], details: WorkflowRunDetails) -> None:
    """Aggregate matrix job timings to find worst-case performance"""

    for matrix_name, pattern in MATRIX_JOB_PATTERNS.items():
        matrix_jobs = []

        # Find all jobs matching the matrix pattern
        for job_name, job_data in jobs.items():
            if job_name.startswith(pattern) and job_data["conclusion"] == "success":
                if job_data.get("startedAt") and job_data.get("completedAt"):
                    started = parse_time(job_data["startedAt"])
                    completed = parse_time(job_data["completedAt"])
                    duration = (completed - started).total_seconds()

                    # Extract the matrix value (e.g., "agent" from "Unit Tests - agent")
                    matrix_value = job_name[len(pattern) :]

                    matrix_jobs.append(
                        {
                            "name": job_name,
                            "value": matrix_value,
                            "duration": duration,
                            "job_id": job_data["databaseId"],
                        }
                    )

        if matrix_jobs:
            # Sort by duration to find worst case
            matrix_jobs.sort(key=lambda x: x["duration"], reverse=True)
            worst_case = matrix_jobs[0]

            details.matrix_aggregates[matrix_name] = {
                "worst_case": worst_case,
                "all_jobs": matrix_jobs,
                "worst_duration": worst_case["duration"],
                "mean_duration": statistics.mean(job["duration"] for job in matrix_jobs),
                "count": len(matrix_jobs),
            }

            # Also add the worst case to regular job durations with a special key
            details.job_durations[f"{matrix_name} (worst-case: {worst_case['value']})"] = worst_case["duration"]


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

                # Aggregate matrix jobs first
                aggregate_matrix_jobs(jobs, details)

                # Collect all job durations (non-matrix jobs)
                for job_name, job_data in jobs.items():
                    # Skip matrix jobs as they're already aggregated
                    if any(job_name.startswith(pattern) for pattern in MATRIX_JOB_PATTERNS.values()):
                        continue

                    if (
                        job_data["conclusion"] == "success"
                        and job_data.get("startedAt")
                        and job_data.get("completedAt")
                    ):
                        started = parse_time(job_data["startedAt"])
                        completed = parse_time(job_data["completedAt"])
                        details.job_durations[job_name] = (completed - started).total_seconds()

                # Try to get step timings from the worst-case matrix job
                if "unit-tests" in details.matrix_aggregates:
                    worst_case = details.matrix_aggregates["unit-tests"]["worst_case"]
                    job_id = worst_case["job_id"]

                    setup_env_time = get_step_timing(job_id, "Setup Environment")
                    run_tests_time = get_step_timing(job_id, f"Run {worst_case['value']} tests")

                    if setup_env_time is not None:
                        details.setup_env_duration = setup_env_time
                    if run_tests_time is not None:
                        details.run_tests_duration = run_tests_time

            except Exception as e:
                print(f"⚠️  Failed to get detailed timing data: {e}")

            return details, conclusion

        time.sleep(POLL_INTERVAL)


def trigger_all_runs(branches: list[str], repeats: int) -> dict[str, list[tuple[str, datetime]]]:
    print("\n🚀 Triggering all workflow runs...")
    triggered_by_branch = {branch: [] for branch in branches}

    for branch in branches:
        for i in range(repeats):
            print(f"▶️  Trigger {i + 1}/{repeats} for `{branch}`")
            try:
                uuid_tag = trigger_workflow(branch)
                triggered_by_branch[branch].append((uuid_tag, datetime.utcnow()))
            except Exception as e:
                print(f"❌ Failed to trigger workflow on `{branch}`: {e}")

    return triggered_by_branch


def resolve_run_numbers(triggered_runs: dict[str, list[tuple[str, datetime]]]) -> dict[str, list[str]]:
    minutes_to_wait = 10
    total_seconds = minutes_to_wait * 60

    # Countdown loop
    for remaining in range(total_seconds, 0, -1):
        if remaining % 60 == 0 and remaining > 30:
            print(f"⏳ Waiting for workflow logs to become available (sleeping {remaining // 60} minutes)...")
        elif remaining in {30, 20, 10, 5, 4, 3, 2, 1}:
            print(f"⏳ Waiting for workflow logs to become available (sleeping {remaining} seconds)...")
        time.sleep(1)

    resolved_by_branch = {branch: [] for branch in triggered_runs}

    for branch, entries in triggered_runs.items():
        for uuid_tag, _ in entries:
            try:
                run_number = find_workflow_run(branch, uuid_tag)
                print(f"🎯 Resolved run_id={uuid_tag} → {run_number}")
                resolved_by_branch[branch].append(run_number)
            except Exception as e:
                print(f"❌ Failed to resolve run_id {uuid_tag} for `{branch}`: {e}")

    return resolved_by_branch


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
                assert details.total_duration
                print(f"✅ {branch} → {format_duration(details.total_duration)} ({details.total_duration:.1f}s)")

                # Print matrix aggregates if any
                for matrix_name, matrix_data in details.matrix_aggregates.items():
                    worst = matrix_data["worst_case"]
                    print(f"   └─ {matrix_name} worst-case: {worst['value']} @ {format_duration(worst['duration'])}")
                    print(
                        f"      └─ Mean across {matrix_data['count']} jobs: "
                        "{format_duration(matrix_data['mean_duration'])}"
                    )

                if details.setup_env_duration:
                    print(f"   └─ Setup Environment: {details.setup_env_duration:.1f}s")
                if details.run_tests_duration:
                    print(f"   └─ Run tests: {details.run_tests_duration:.1f}s")

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

    # Matrix job analysis
    print("\n🔢 MATRIX JOB ANALYSIS:")

    for branch, results in results_by_branch.items():
        detailed_timings = results["detailed_timings"]
        if not detailed_timings:
            continue

        # Check if this branch uses matrix jobs
        has_matrix = any("unit-tests" in timing.matrix_aggregates for timing in detailed_timings)

        if has_matrix:
            # Collect matrix aggregates across all runs
            worst_durations = []
            mean_durations = []
            worst_job_names = []

            for timing in detailed_timings:
                if "unit-tests" in timing.matrix_aggregates:
                    aggregate = timing.matrix_aggregates["unit-tests"]
                    worst_durations.append(aggregate["worst_duration"])
                    mean_durations.append(aggregate["mean_duration"])
                    worst_job_names.append(aggregate["worst_case"]["value"])

            if worst_durations:
                # Find which job was worst most often
                from collections import Counter

                job_counter = Counter(worst_job_names)
                most_common_worst = job_counter.most_common(1)[0]

                print(f"\n{branch} (uses matrix strategy):")
                print("  unit-tests:")
                print(
                    f"    Worst-case duration: min={format_duration(min(worst_durations)):>6}  "
                    f"mean={format_duration(statistics.mean(worst_durations)):>6}  "
                    f"max={format_duration(max(worst_durations)):>6}"
                )
                print(f"    Average across matrix: {format_duration(statistics.mean(mean_durations)):>6}")
                print(
                    f"    Most often slowest: {most_common_worst[0]} "
                    "({most_common_worst[1]}/{len(worst_job_names)} times)"
                )
        else:
            print(f"\n{branch} (single job strategy)")

    # Unit test step timing comparison - THE KEY COMPARISON
    print("\n🧪 UNIT TEST STEP TIMING COMPARISON:")
    print("=" * 80)

    comparison_data = {}

    for branch, results in results_by_branch.items():
        detailed_timings = results["detailed_timings"]
        if not detailed_timings:
            continue

        # Determine strategy type
        has_matrix = any("unit-tests" in timing.matrix_aggregates for timing in detailed_timings)
        strategy_type = "matrix" if has_matrix else "single"

        # Collect setup environment times
        setup_times = [d.setup_env_duration for d in detailed_timings if d.setup_env_duration is not None]
        test_times = [d.run_tests_duration for d in detailed_timings if d.run_tests_duration is not None]

        if setup_times or test_times:
            comparison_data[branch] = {"strategy": strategy_type, "setup_times": setup_times, "test_times": test_times}

    # Print comparison table
    if comparison_data:
        print(f"\n{'Branch':<20} {'Strategy':<10} {'Setup Environment':<25} {'Run Tests':<25}")
        print("-" * 80)

        for branch, data in comparison_data.items():
            strategy_label = f"{data['strategy']:<10}"

            # Setup environment stats
            if data["setup_times"]:
                setup_stats = (
                    f"mean={format_duration(statistics.mean(data['setup_times'])):>6} (n={len(data['setup_times'])})"
                )
            else:
                setup_stats = "N/A"

            # Test run stats
            if data["test_times"]:
                test_mean = statistics.mean(data["test_times"])
                test_min = min(data["test_times"])
                test_max = max(data["test_times"])
                test_stats = (
                    f"mean={format_duration(test_mean):>6} [{format_duration(test_min)}-{format_duration(test_max)}]"
                )
                if data["strategy"] == "matrix":
                    test_stats += " *"
            else:
                test_stats = "N/A"

            print(f"{branch:<20} {strategy_label} {setup_stats:<25} {test_stats:<25}")

        if any(d["strategy"] == "matrix" for d in comparison_data.values()):
            print("\n* For matrix strategy, this represents the worst-case (slowest) job from each run")

        # Direct comparison if we have both strategies
        matrix_branches = [b for b, d in comparison_data.items() if d["strategy"] == "matrix"]
        single_branches = [b for b, d in comparison_data.items() if d["strategy"] == "single"]

        if matrix_branches and single_branches:
            print("\n📈 STRATEGY COMPARISON:")
            print("-" * 50)

            # Compare average test times
            matrix_test_times = []
            single_test_times = []

            for branch in matrix_branches:
                matrix_test_times.extend(comparison_data[branch]["test_times"])
            for branch in single_branches:
                single_test_times.extend(comparison_data[branch]["test_times"])

            if matrix_test_times and single_test_times:
                matrix_mean = statistics.mean(matrix_test_times)
                single_mean = statistics.mean(single_test_times)

                speedup = single_mean / matrix_mean if matrix_mean > 0 else 0

                print(f"Average test time (matrix worst-case): {format_duration(matrix_mean)}")
                print(f"Average test time (single job):        {format_duration(single_mean)}")
                print(f"Speedup factor:                        {speedup:.2f}x")

                if speedup > 1:
                    print(f"✅ Single job is {speedup:.2f}x faster than matrix worst-case")
                else:
                    print(f"⚠️  Matrix is {1 / speedup:.2f}x faster than single job")

    # Detailed breakdown per branch
    print("\n📋 DETAILED STEP TIMINGS PER BRANCH:")

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
                    f"  Run tests:              "
                    f"min={format_duration(min(test_times)):>6}  "
                    f"mean={format_duration(statistics.mean(test_times)):>6}  "
                    f"max={format_duration(max(test_times)):>6}  "
                    f"(n={len(test_times)})"
                )

    # Job timing summary (now includes worst-case matrix timings)
    print("\n⚙️  JOB DURATION BREAKDOWN (mean):")
    all_job_names = set()
    for results in results_by_branch.values():
        for details in results["detailed_timings"]:
            all_job_names.update(details.job_durations.keys())

    if all_job_names:
        sorted_job_names = sorted(all_job_names)
        print(f"{'Job':<50} " + " ".join(f"{branch:<15}" for branch in results_by_branch.keys()))
        print("-" * (50 + 16 * len(results_by_branch)))

        for job_name in sorted_job_names:
            row = f"{job_name[:49]:<50}"
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
    resolved = resolve_run_numbers(triggered)
    results = wait_for_all_runs(resolved)
    summarize(results)
