#!/usr/bin/env python3

import json
import os
import statistics
import subprocess
import sys
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

# Step name mappings for special cases
TEST_STEP_NAMES = {
    "All Packages": "Run all package tests",  # Single job uses different step name
    # Matrix jobs use pattern: "Run {package} tests"
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


def find_workflow_runs_batch(branch: str, run_ids: list[str]) -> dict[str, str]:
    """Find multiple workflow runs for a branch in a single pass.

    Returns a dict mapping run_id to run_number for found runs.
    """
    print(f"🔍 Searching for {len(run_ids)} workflow runs on branch={branch}")

    # Keep track of which run_ids we've found
    found_runs = {}
    remaining_ids = set(run_ids)

    # Cache for run logs to avoid re-fetching
    run_logs_cache = {}

    # Based on benchmarks: 20→50→100→200 provides good balance
    limits = [20, 50, 100, 200]

    for i, limit in enumerate(limits):
        if not remaining_ids:
            break  # Found everything

        # Fetch list of recent runs
        print(f"  Fetching {limit} recent runs...")
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
                str(limit),
                "--json",
                "databaseId,createdAt",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"⚠️  Failed to fetch run list: {result.stderr}")
            if i < len(limits) - 1:
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Failed to fetch workflow runs: {result.stderr}")

        try:
            runs = json.loads(result.stdout)
            new_runs = [r for r in runs if str(r["databaseId"]) not in run_logs_cache]
            new_runs_count = len(new_runs)

            if new_runs_count > 0:
                print(f"  Checking {new_runs_count} new runs (already cached: {len(run_logs_cache)})")

            # Check each NEW run's logs for our run_ids
            for j, run in enumerate(new_runs):
                run_db_id = str(run["databaseId"])

                # Show progress for larger searches
                if new_runs_count >= 50 and (j + 1) % 25 == 0:
                    print(f"    Checked {j + 1}/{new_runs_count} new runs...")

                # Fetch logs for this run
                log_result = subprocess.run(
                    ["gh", "run", "view", run_db_id, "--log"],
                    capture_output=True,
                    text=True,
                )

                if log_result.returncode != 0:
                    # Cache as None to avoid re-fetching
                    run_logs_cache[run_db_id] = None
                    continue

                # Cache the log content
                run_logs_cache[run_db_id] = log_result.stdout

                # Check if this run matches any of our remaining run_ids
                for run_id in list(remaining_ids):
                    if f"RUN_ID={run_id}" in log_result.stdout:
                        print(f"  ✅ Found run {run_db_id} for run_id={run_id}")
                        found_runs[run_id] = run_db_id
                        remaining_ids.remove(run_id)
                        break  # This run can only match one run_id

            # If we still have unfound IDs and more limits to try
            if remaining_ids and i < len(limits) - 1:
                print(f"  Still looking for {len(remaining_ids)} run(s), expanding search...")
                time.sleep(1)

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error: {e}")
            if i < len(limits) - 1:
                time.sleep(2)
                continue
            else:
                raise

    # Report any unfound runs
    if remaining_ids:
        print(f"  ❌ Could not find {len(remaining_ids)} run(s): {', '.join(remaining_ids)}")

    return found_runs


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


def save_run_ids(run_ids_by_branch: dict[str, list[str]], script_dir: str) -> str:
    """Save run IDs to a JSON file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(sys.argv[0]) if sys.argv else "benchmark_checks_workflow.py"

    filename = f"runs_{timestamp}.json"
    filepath = os.path.join(script_dir, filename)
    with open(filepath, "w") as f:
        json.dump(run_ids_by_branch, f, indent=2)

    print(f"\n💾 Saved run IDs to: {filepath}")
    print(f"   To re-analyze these runs later: python {script_name} -f {filename}")
    return filepath


def load_run_ids(filepath: str) -> dict[str, list[str]]:
    """Load run IDs from a JSON file

    Expected JSON format:
    ```
    {
      "main": ["17507770748", "17507770859", "17507771006"],
      "robb/0905-matrix": ["17507771120", "17507771263"]
    }
    ```
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Run file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Ensure all run IDs are strings
    result = {}
    for branch, run_ids in data.items():
        result[branch] = [str(run_id) for run_id in run_ids]

    return result


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

                    # Check if we have a special step name mapping
                    if worst_case["value"] in TEST_STEP_NAMES:
                        run_tests_time = get_step_timing(job_id, TEST_STEP_NAMES[worst_case["value"]])
                    else:
                        # Default pattern for matrix jobs
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

    # Process all runs for each branch in batch
    for branch, entries in triggered_runs.items():
        # Extract all run_ids for this branch
        run_ids = [uuid_tag for uuid_tag, _ in entries]

        if not run_ids:
            continue

        try:
            # Find all runs for this branch in one go
            found_runs = find_workflow_runs_batch(branch, run_ids)

            # Maintain order and handle missing runs
            for uuid_tag, _ in entries:
                if uuid_tag in found_runs:
                    run_number = found_runs[uuid_tag]
                    print(f"🎯 Resolved run_id={uuid_tag} → {run_number}")
                    resolved_by_branch[branch].append(run_number)
                else:
                    print(f"❌ Failed to resolve run_id {uuid_tag} for `{branch}`")

        except Exception as e:
            print(f"❌ Failed to resolve runs for `{branch}`: {e}")

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
                        f"{format_duration(matrix_data['mean_duration'])}"
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

        # Check if this branch uses matrix jobs with more than 1 job
        has_matrix = False
        for timing in detailed_timings:
            if "unit-tests" in timing.matrix_aggregates:
                if timing.matrix_aggregates["unit-tests"]["count"] > 1:
                    has_matrix = True
                    break

        if has_matrix:
            # Collect matrix aggregates across all runs
            worst_durations = []
            mean_durations = []
            worst_job_names = []
            job_counts = []

            for timing in detailed_timings:
                if "unit-tests" in timing.matrix_aggregates:
                    aggregate = timing.matrix_aggregates["unit-tests"]
                    worst_durations.append(aggregate["worst_duration"])
                    mean_durations.append(aggregate["mean_duration"])
                    worst_job_names.append(aggregate["worst_case"]["value"])
                    job_counts.append(aggregate["count"])

            if worst_durations:
                # Find which job was worst most often
                from collections import Counter

                job_counter = Counter(worst_job_names)
                most_common_worst = job_counter.most_common(1)[0]

                # Get the count from the first timing's aggregate
                job_count = job_counts[0] if job_counts else 1

                print(f"\n{branch} (uses {'matrix' if job_count > 1 else 'single job'} strategy):")
                print("  unit-tests:")
                print(
                    f"    Worst-case duration: min={format_duration(min(worst_durations)):>6}  "
                    f"mean={format_duration(statistics.mean(worst_durations)):>6}  "
                    f"max={format_duration(max(worst_durations)):>6}"
                )

                if job_count > 1:
                    print(f"    Average across matrix: {format_duration(statistics.mean(mean_durations)):>6}")
                    print(
                        f"    Most often slowest: {most_common_worst[0]} "
                        f"({most_common_worst[1]}/{len(worst_job_names)} times)"
                    )
                else:
                    # Single job - the "worst case" is the only case
                    print(f"    Single job: {most_common_worst[0]}")
        else:
            # Check if we have any unit-tests data at all
            has_unit_tests = any("unit-tests" in timing.matrix_aggregates for timing in detailed_timings)
            if has_unit_tests:
                # We have unit tests but in single job configuration
                worst_durations = []
                job_names = []

                for timing in detailed_timings:
                    if "unit-tests" in timing.matrix_aggregates:
                        aggregate = timing.matrix_aggregates["unit-tests"]
                        worst_durations.append(aggregate["worst_duration"])
                        job_names.append(aggregate["worst_case"]["value"])

                if worst_durations:
                    print(f"\n{branch} (uses single job strategy):")
                    print("  unit-tests:")
                    print(
                        f"    Duration: min={format_duration(min(worst_durations)):>6}  "
                        f"mean={format_duration(statistics.mean(worst_durations)):>6}  "
                        f"max={format_duration(max(worst_durations)):>6}"
                    )
                    print(f"    Single job: {job_names[0]}")
            else:
                print(f"\n{branch} (no matrix jobs found)")

    # Unit test step timing comparison - THE KEY COMPARISON
    print("\n🧪 UNIT TEST STEP TIMING COMPARISON:")
    print("=" * 80)

    comparison_data = {}

    for branch, results in results_by_branch.items():
        detailed_timings = results["detailed_timings"]
        if not detailed_timings:
            continue

        # Determine strategy type - check if actually using multiple matrix jobs
        has_matrix = False
        for timing in detailed_timings:
            if "unit-tests" in timing.matrix_aggregates:
                # Only consider it a matrix strategy if there's more than 1 job
                if timing.matrix_aggregates["unit-tests"]["count"] > 1:
                    has_matrix = True
                    break
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
                    print(f"✅ Matrix worst-case is {1 / speedup:.2f}x slower than single job")

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

    parser = argparse.ArgumentParser(
        description="Benchmark GitHub workflow across branches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new benchmarks on branches:
  %(prog)s main feature/branch -n 5

  # Re-analyze existing runs from file:
  %(prog)s -f runs_20240105_143022.json

  # Run benchmarks without saving run IDs:
  %(prog)s main dev --no-save

Expected JSON format for -f/--file:
{
  "main": ["17507770748", "17507770859", "17507771006"],
  "robb/0905-matrix": ["17507771120", "17507771263"]
}
        """,
    )

    # Create mutually exclusive group for branches vs loading from file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("branches", nargs="*", default=[], help="Branches to benchmark")
    input_group.add_argument("-f", "--file", type=str, help="Load run IDs from JSON file")

    parser.add_argument("-n", "--repeats", type=int, default=10, help="Number of runs per branch (ignored with --file)")
    parser.add_argument("--no-save", action="store_true", help="Don't save run IDs to file")

    args = parser.parse_args()

    # Get script directory for saving run files
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.file:
        # Load and analyze existing runs
        print(f"📂 Loading run IDs from: {args.file}")
        try:
            run_ids = load_run_ids(args.file)
            print(f"✅ Loaded {sum(len(ids) for ids in run_ids.values())} runs across {len(run_ids)} branches")
            for branch, ids in run_ids.items():
                print(f"   • {branch}: {len(ids)} runs")

            # Proceed directly to analysis
            results = wait_for_all_runs(run_ids)
            summarize(results)

        except Exception as e:
            print(f"❌ Failed to load run file: {e}")
            exit(1)
    else:
        # Normal mode: trigger new runs
        if not args.branches:
            parser.error("Please provide branches to benchmark or use --file to load existing runs")

        triggered = trigger_all_runs(args.branches, args.repeats)
        resolved = resolve_run_numbers(triggered)

        # Save run IDs unless --no-save is specified
        if not args.no_save:
            save_run_ids(resolved, script_dir)

        results = wait_for_all_runs(resolved)
        summarize(results)
